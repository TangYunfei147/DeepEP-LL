// Copyright (c) 2026, Lu Lu
#pragma once
#include "kernel_operator.h"
#include "ll_stage_timer_layout.h"
#include "ll_stage_timer_profile.h"
#include "ll_core_config.h"
#include "ll_overlap_stage.h"
namespace LlStageTimer {
// Scalar boundaries preserve the existing pipeline dependencies. Repeated metrics
// are cumulative intervals, may nest, and are never a partition of kernel time.
// Each core writes its own 1024B row once after WORK_DONE. No extra UB or events.
class DeviceTimer {
public:
    __aicore__ inline void Init(GM_ADDR address,uint32_t operation,uint32_t mode) {
#ifdef DEEPEP_LL_STAGE_TIMER
        if(address && (mode!=1U || (operation!=1U && operation!=2U) ||
                       DispatchDedup::kLlOverlapExperimentStage!=3U)) { AscendC::Trap(); return; }
        address_=address; operation_=operation; mode_=mode;
        if(address_) {
            valid_=0U; metricValid_=0U; lanes_=0U;
            for(uint32_t i=0;i<kCycleMetricCount;++i) { metrics_[i][0]=0U; metrics_[i][1]=0U; }
        }
#else
        (void)address; (void)operation; (void)mode;
#endif
    }
    __aicore__ inline bool Enabled() const {
#ifdef DEEPEP_LL_STAGE_TIMER
        return address_!=nullptr;
#else
        return false;
#endif
    }
    template<Point point> __aicore__ inline void MarkAt(uint64_t tick) {
        static_assert(uint32_t(point)<kPointCount,"invalid timestamp point");
#ifdef DEEPEP_LL_STAGE_TIMER
        constexpr uint64_t bit=uint64_t(1U)<<uint32_t(point);
        if(address_ && !(valid_&bit)) { ticks_[uint32_t(point)]=tick; valid_|=bit; }
#else
        (void)tick;
#endif
    }
    template<Point point> __aicore__ inline void Mark() {
        static_assert(uint32_t(point)<kPointCount,"invalid timestamp point");
#ifdef DEEPEP_LL_STAGE_TIMER
        constexpr uint64_t bit=uint64_t(1U)<<uint32_t(point);
        if(address_ && !(valid_&bit)) { ticks_[uint32_t(point)]=AscendC::GetSystemCycle(); valid_|=bit; }
#endif
    }
    template<Point observed,Point point> __aicore__ inline void MarkIfSeen() {
        static_assert(uint32_t(observed)<kPointCount,"invalid prerequisite timestamp");
#ifdef DEEPEP_LL_STAGE_TIMER
        if(address_ && (valid_&(uint64_t(1U)<<uint32_t(observed)))) Mark<point>();
#endif
    }
    template<Stage stage> __aicore__ inline uint64_t Begin() const {
#ifdef DEEPEP_LL_STAGE_TIMER
        if constexpr(CycleMetricSelected(stage) && !CycleMetricCounter(stage)) {
            if(address_) return AscendC::GetSystemCycle();
        }
#endif
        return 0U;
    }
    template<Stage stage> __aicore__ inline void End(uint64_t begin) {
#ifdef DEEPEP_LL_STAGE_TIMER
        if constexpr(CycleMetricSelected(stage) && !CycleMetricCounter(stage)) {
            if(address_) {
                constexpr uint32_t index=CycleMetricIndex(stage);
                metrics_[index][0]+=AscendC::GetSystemCycle()-begin;
                ++metrics_[index][1]; metricValid_|=uint64_t(1U)<<index;
            }
        }
#else
        (void)begin;
#endif
    }
    template<Stage stage> __aicore__ inline void Count(uint64_t count) {
#ifdef DEEPEP_LL_STAGE_TIMER
        if constexpr(CycleMetricCounter(stage)) {
            if(address_) {
                constexpr uint32_t index=CycleMetricIndex(stage);
                metrics_[index][1]+=count; metricValid_|=uint64_t(1U)<<index;
            }
        }
#else
        (void)count;
#endif
    }
    __aicore__ inline void SendLanes(uint32_t lanes,bool overlap) {
#ifdef DEEPEP_LL_STAGE_TIMER
        if(address_) {
            const uint32_t bit=overlap ? 0x10000U:0U;
            if((lanes_&0xffffU) && (lanes_&0x10000U)!=bit) { AscendC::Trap(); return; }
            lanes_=(lanes_&0x1ff00U)|(lanes&0xffU)|bit;
        }
#else
        (void)lanes; (void)overlap;
#endif
    }
    __aicore__ inline void RecvLanes(uint32_t lanes,bool overlap) {
#ifdef DEEPEP_LL_STAGE_TIMER
        if(address_) {
            const uint32_t bit=overlap ? 0x10000U:0U;
            if((lanes_&0xffffU) && (lanes_&0x10000U)!=bit) { AscendC::Trap(); return; }
            lanes_=(lanes_&0x100ffU)|((lanes&0xffU)<<8U)|bit;
        }
#else
        (void)lanes; (void)overlap;
#endif
    }
    __aicore__ inline void Flush() {
#ifdef DEEPEP_LL_STAGE_TIMER
        if(!address_) return;
        const uint32_t core=AscendC::GetBlockIdx();
        auto* words=reinterpret_cast<__gm__ uint64_t*>(address_)+core*kWordsPerCore;
        const uint32_t role=operation_==1U ? (core<DispatchDedup::kLlSendCoreNum ? 3U:2U)
                                          : (core<DispatchDedup::kLlSendCoreNum ? 1U:4U);
        words[0]=kMagic; words[1]=kSchema; words[2]=operation_; words[3]=AscendC::GetBlockNum();
        words[4]=core; words[5]=mode_|(uint64_t(DispatchDedup::kLlSendCoreNum)<<8U)|
            (uint64_t(role)<<16U)|(uint64_t(kProfileId)<<24U);
        words[6]=valid_;
        // lanes[0:15], overlap[16], reserved zero[17:23], metric-valid[24:63].
        words[7]=uint64_t(lanes_)|(metricValid_<<24U);
        for(uint32_t i=0;i<kPointCount;++i)
            words[kRecordWordOffset+i]=(valid_&(uint64_t(1U)<<i)) ? ticks_[i]:0U;
        for(uint32_t i=0;i<kCycleMetricCount;++i) {
            words[kCycleMetricWordOffset+i*2U]=metrics_[i][0];
            words[kCycleMetricWordOffset+i*2U+1U]=metrics_[i][1];
        }
        // Host waits for kernel completion; no diagnostic SyncAll or PIPE_ALL.
        for(uint32_t offset=0;offset<kExportWords;offset+=8U) {
            AscendC::GlobalTensor<uint32_t> tensor;
            tensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(words+offset),16U);
            AscendC::DataCacheCleanAndInvalid<uint32_t,AscendC::CacheLine::SINGLE_CACHE_LINE,
                AscendC::DcciDst::CACHELINE_OUT>(tensor);
        }
#endif
    }
private:
#ifdef DEEPEP_LL_STAGE_TIMER
    GM_ADDR address_=nullptr;
    uint32_t operation_=0U,mode_=1U,lanes_=0U;
    uint64_t valid_=0U,metricValid_=0U;
    uint64_t ticks_[kPointCount];
    uint64_t metrics_[kCycleMetricCount][2];
#endif
};
template<Stage stage,bool Selected=CycleMetricSelected(stage) && !CycleMetricCounter(stage)>
class MetricScope {
public:
    __aicore__ inline explicit MetricScope(DeviceTimer*) {}
};
template<Stage stage> class MetricScope<stage,true> {
public:
    __aicore__ inline explicit MetricScope(DeviceTimer* timer):timer_(timer) {
        if(timer_ && timer_->Enabled()) begin_=timer_->Begin<stage>();
        else timer_=nullptr;
    }
    __aicore__ inline ~MetricScope() { if(timer_) timer_->End<stage>(begin_); }
private:
    DeviceTimer* timer_=nullptr;
    uint64_t begin_=0U;
};
}
#define LL_CYCLE_JOIN_IMPL(a,b) a##b
#define LL_CYCLE_JOIN(a,b) LL_CYCLE_JOIN_IMPL(a,b)
#ifdef DEEPEP_LL_STAGE_TIMER
#define LL_SEG_SCOPE(stage) LlStageTimer::MetricScope<LlStageTimer::stage> LL_CYCLE_JOIN(llCycleScope_,__COUNTER__)(timer_)
#define LL_SEG_BEGIN(name,stage) const uint64_t name=(timer_ ? timer_->Begin<LlStageTimer::stage>():0U)
#define LL_SEG_END(name,stage) do { if(timer_) timer_->End<LlStageTimer::stage>(name); } while(false)
#define LL_SEG_COUNT(stage,count) do { if constexpr(LlStageTimer::CycleMetricCounter(LlStageTimer::stage)) { if(timer_ && timer_->Enabled()) timer_->Count<LlStageTimer::stage>(count); } } while(false)
#define LL_MTE3_SEND_LANES(lanes,overlap) do { if(timer_ && timer_->Enabled()) timer_->SendLanes(lanes,overlap); } while(false)
#define LL_MTE3_RECV_LANES(lanes,overlap) do { if(timer_ && timer_->Enabled()) timer_->RecvLanes(lanes,overlap); } while(false)
#define LL_SPARSE_MARK(point) do { if(timer_) timer_->Mark<LlStageTimer::point>(); } while(false)
#define LL_SPARSE_WHEN(condition,point) do { if(timer_ && timer_->Enabled() && (condition)) timer_->Mark<LlStageTimer::point>(); } while(false)
#define LL_SPARSE_AFTER(observed,point) do { if(timer_) timer_->MarkIfSeen<LlStageTimer::observed,LlStageTimer::point>(); } while(false)
#define LL_CYCLE_MARK(point) LL_SPARSE_MARK(point)
#define LL_CYCLE_LOCAL_MARK(timer,point) (timer).Mark<LlStageTimer::point>()
#else
#define LL_SEG_SCOPE(stage) do {} while(false)
#define LL_SEG_BEGIN(name,stage) do {} while(false)
#define LL_SEG_END(name,stage) do {} while(false)
#define LL_SEG_COUNT(stage,count) do {} while(false)
#define LL_MTE3_SEND_LANES(lanes,overlap) do {} while(false)
#define LL_MTE3_RECV_LANES(lanes,overlap) do {} while(false)
#define LL_SPARSE_MARK(point) do {} while(false)
#define LL_SPARSE_WHEN(condition,point) do {} while(false)
#define LL_SPARSE_AFTER(observed,point) do {} while(false)
#define LL_CYCLE_MARK(point) do {} while(false)
#define LL_CYCLE_LOCAL_MARK(timer,point) do {} while(false)
#endif
// Historical probes remain labels only; arguments are never evaluated.
#define LL_SEG_RELAY_COMPLETION(pipe) do {} while(false)
#define LL_SEG_FIRST(stage) do {} while(false)
#define LL_MTE3_WAIT_BEGIN(name) do {} while(false)
#define LL_MTE3_WAIT_END(name,metric) do {} while(false)
#define LL_MTE3_DRAIN_WHEN(condition,metric,mode,pipe) do {} while(false)
#define LL_MTE3_DRAIN(metric,mode,pipe) do {} while(false)
#define LL_BW_BEGIN(stage,pipe) do {} while(false)
#define LL_BW_END(stage,pipe) do {} while(false)
#define LL_BW_TRANSFER(stage,peer,bytes,payload) do {} while(false)
#define LL_BW_SPLIT_INPUT(input,pipe) do {} while(false)
