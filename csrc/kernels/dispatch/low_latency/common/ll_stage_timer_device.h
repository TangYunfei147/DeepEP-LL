// Copyright (c) 2026, Lu Lu
#pragma once
#include "kernel_operator.h"
#include "ll_stage_timer_layout.h"
#include "ll_stage_timer_profile.h"
#include "ll_core_config.h"
namespace LlStageTimer {
// Purpose-specific Scalar waits plus first-visit milestones. Observe adds no
// dependency; explicit TX/RX modes expose queued MTE3 completion on S cores.
// All GM writes occur at exit; OFF/null-buffer paths have no diagnostic clocks.
class DeviceTimer {
public:
    __aicore__ inline void Init(GM_ADDR address, uint32_t operation, uint32_t mode) {
#ifdef DEEPEP_LL_STAGE_TIMER
        if (address && ((mode != 1U && mode != 4U && mode != 5U) ||
                        (operation != 1U && mode != 1U))) { AscendC::Trap(); return; }
        address_=address; operation_=operation; mode_=mode;
        if (address_) {
            valid_=0U; lanes_=0U;
            for(uint32_t i=0;i<kWaitMetricCount;++i)
                for(uint32_t field=0;field<kWaitMetricFields;++field) waits_[i][field]=0U;
        }
#else
        (void)address; (void)operation; (void)mode;
#endif
    }
    __aicore__ inline bool Enabled() const {
#ifdef DEEPEP_LL_STAGE_TIMER
        return address_ != nullptr;
#else
        return false;
#endif
    }
    template<Point point> __aicore__ inline void MarkAt(uint64_t tick) {
        static_assert(uint32_t(point)<kPointCount,"invalid sparse timestamp point");
#ifdef DEEPEP_LL_STAGE_TIMER
        constexpr uint64_t bit=uint64_t(1U)<<uint32_t(point);
        if(address_ && !(valid_ & bit)) { ticks_[uint32_t(point)]=tick; valid_ |= bit; }
#else
        (void)tick;
#endif
    }
    template<Point point> __aicore__ inline void Mark() {
        static_assert(uint32_t(point)<kPointCount,"invalid sparse timestamp point");
#ifdef DEEPEP_LL_STAGE_TIMER
        constexpr uint64_t bit=uint64_t(1U)<<uint32_t(point);
        if (address_ && !(valid_ & bit)) {
            ticks_[uint32_t(point)]=AscendC::GetSystemCycle();
            valid_ |= bit;
        }
#endif
    }
    template<Point observed, Point point> __aicore__ inline void MarkIfSeen() {
        static_assert(uint32_t(observed)<kPointCount,"invalid prerequisite timestamp");
#ifdef DEEPEP_LL_STAGE_TIMER
        if(address_ && (valid_ & (uint64_t(1U)<<uint32_t(observed)))) Mark<point>();
#endif
    }
    __aicore__ inline uint64_t WaitBegin() const {
#ifdef DEEPEP_LL_STAGE_TIMER
        return address_ ? AscendC::GetSystemCycle() : 0U;
#else
        return 0U;
#endif
    }
    template<WaitMetric metric> __aicore__ inline void WaitEnd(uint64_t start) {
        static_assert(uint32_t(metric)<4U,"only existing Scalar waits use WaitEnd");
#ifdef DEEPEP_LL_STAGE_TIMER
        if(address_) Record<metric>(AscendC::GetSystemCycle()-start,0U);
#else
        (void)start;
#endif
    }
    __aicore__ inline void SendLanes(uint32_t lanes, bool overlap) {
#ifdef DEEPEP_LL_STAGE_TIMER
        if(address_) {
            const uint32_t bit=overlap ? 0x10000U : 0U;
            if((lanes_&0xffffU) && (lanes_&0x10000U)!=bit) { AscendC::Trap(); return; }
            lanes_=(lanes_&0x1ff00U)|(lanes&0xffU)|bit;
        }
#else
        (void)lanes; (void)overlap;
#endif
    }
    __aicore__ inline void RecvLanes(uint32_t lanes, bool overlap) {
#ifdef DEEPEP_LL_STAGE_TIMER
        if(address_) {
            const uint32_t bit=overlap ? 0x10000U : 0U;
            if((lanes_&0xffffU) && (lanes_&0x10000U)!=bit) { AscendC::Trap(); return; }
            lanes_=(lanes_&0x100ffU)|((lanes&0xffU)<<8U)|bit;
        }
#else
        (void)lanes; (void)overlap;
#endif
    }
    template<WaitMetric metric, uint32_t mode>
    __aicore__ inline void DrainMte3(AscendC::TPipe* pipe) {
        static_assert((mode==4U && (metric==TX_REUSE_MTE3 || metric==TX_END_MTE3)) ||
                      (mode==5U && (metric==RX_REUSE_MTE3 || metric==RX_END_MTE3)),
                      "MTE3 intervention/metric mismatch");
#ifdef DEEPEP_LL_STAGE_TIMER
        if(!address_ || operation_!=1U || mode_!=mode ||
           AscendC::GetBlockIdx()>=DispatchDedup::kLlSendCoreNum) return;
        // MTE3_S is separate from the lane event domains. Fetch only for this
        // immediate Set/Wait pair: do not occupy or reuse an in-flight lease.
        const auto event=pipe->FetchEventID(AscendC::HardEvent::MTE3_S);
        const uint64_t begin=AscendC::GetSystemCycle();
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(event);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(event);
        const uint64_t completed=AscendC::GetSystemCycle();
        // Includes queued MTE3 and its preceding producer dependencies, not
        // pure DMA/network latency. The second pair submits no new MTE3 work.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(event);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(event);
        const uint64_t emptyCompleted=AscendC::GetSystemCycle();
        Record<metric>(completed-begin,emptyCompleted-completed);
#else
        (void)pipe;
#endif
    }
    __aicore__ inline void Flush() {
#ifdef DEEPEP_LL_STAGE_TIMER
        if (!address_) return;
        const uint32_t core=AscendC::GetBlockIdx();
        auto* words=reinterpret_cast<__gm__ uint64_t*>(address_)+core*kWordsPerCore;
        const uint32_t role=operation_==1U ? (core<DispatchDedup::kLlSendCoreNum ? 3U:2U)
                                          : (core<DispatchDedup::kLlSendCoreNum ? 1U:4U);
        words[0]=kMagic; words[1]=kSchema; words[2]=operation_; words[3]=AscendC::GetBlockNum();
        words[4]=core; words[5]=mode_ | (uint64_t(DispatchDedup::kLlSendCoreNum)<<8U) |
            (uint64_t(role)<<16U) | (uint64_t(kProfileId)<<24U);
        // Header7: send lanes[0:7], receive lanes[8:15], actual overlap[16].
        // Higher bits remain zero; empty cores may have no configured lanes.
        words[6]=valid_; words[7]=lanes_;
        for(uint32_t i=0;i<kPointCount;++i)
            words[kRecordWordOffset+i]=(valid_ & (uint64_t(1U)<<i)) ? ticks_[i]:0U;
        for(uint32_t i=kRecordWordOffset+kPointCount;i<kWaitMetricWordOffset;++i) words[i]=0U;
        for(uint32_t i=0;i<kWaitMetricCount;++i)
            for(uint32_t field=0;field<kWaitMetricFields;++field)
                words[kWaitMetricWordOffset+i*kWaitMetricFields+field]=waits_[i][field];
        // Host reads only after kernel completion. Each core owns separate lines;
        // flush the contiguous used range once, with no all-core diagnostic sync.
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
    template<WaitMetric metric> __aicore__ inline void Record(uint64_t ticks,uint64_t emptyTicks) {
        auto* stat=waits_[uint32_t(metric)];
        stat[0]+=ticks;
        if(ticks>stat[1]) stat[1]=ticks;
        ++stat[2];
        stat[3]+=emptyTicks;
    }
    GM_ADDR address_=nullptr;
    uint32_t operation_=0U, mode_=1U, lanes_=0U;
    uint64_t valid_=0U;
    uint64_t waits_[kWaitMetricCount][kWaitMetricFields];
    uint64_t ticks_[kPointCount]; // Missing points are never read; Flush writes zero.
#endif
};
}
// Retain source labels without evaluating their arguments or creating scopes.
#define LL_SEG_SCOPE(stage) do {} while(false)
#define LL_SEG_BEGIN(name, stage) do {} while(false)
#define LL_SEG_END(name, stage) do {} while(false)
#define LL_SEG_COUNT(stage, count) do {} while(false)
#define LL_SEG_RELAY_COMPLETION(pipe) do {} while(false)
#define LL_SEG_FIRST(stage) do {} while(false)
#ifdef DEEPEP_LL_STAGE_TIMER
#define LL_MTE3_WAIT_BEGIN(name) const uint64_t name=timer_ ? timer_->WaitBegin() : 0U
#define LL_MTE3_WAIT_END(name,metric) do { if(timer_) timer_->WaitEnd<LlStageTimer::metric>(name); } while(false)
#define LL_MTE3_DRAIN_WHEN(condition,metric,mode,pipe) do { if(timer_ && (condition)) timer_->DrainMte3<LlStageTimer::metric,mode>(pipe); } while(false)
#define LL_MTE3_DRAIN(metric,mode,pipe) do { if(timer_) timer_->DrainMte3<LlStageTimer::metric,mode>(pipe); } while(false)
#define LL_MTE3_SEND_LANES(lanes,overlap) do { if(timer_) timer_->SendLanes(lanes,overlap); } while(false)
#define LL_MTE3_RECV_LANES(lanes,overlap) do { if(timer_) timer_->RecvLanes(lanes,overlap); } while(false)
#define LL_SPARSE_MARK(point) do { if(timer_) timer_->Mark<LlStageTimer::point>(); } while(false)
#define LL_SPARSE_WHEN(condition,point) do { if(timer_ && (condition)) timer_->Mark<LlStageTimer::point>(); } while(false)
#define LL_SPARSE_AFTER(observed,point) do { if(timer_) timer_->MarkIfSeen<LlStageTimer::observed,LlStageTimer::point>(); } while(false)
#else
#define LL_MTE3_WAIT_BEGIN(name) do {} while(false)
#define LL_MTE3_WAIT_END(name,metric) do {} while(false)
#define LL_MTE3_DRAIN_WHEN(condition,metric,mode,pipe) do {} while(false)
#define LL_MTE3_DRAIN(metric,mode,pipe) do {} while(false)
#define LL_MTE3_SEND_LANES(lanes,overlap) do {} while(false)
#define LL_MTE3_RECV_LANES(lanes,overlap) do {} while(false)
#define LL_SPARSE_MARK(point) do {} while(false)
#define LL_SPARSE_WHEN(condition,point) do {} while(false)
#define LL_SPARSE_AFTER(observed,point) do {} while(false)
#endif
