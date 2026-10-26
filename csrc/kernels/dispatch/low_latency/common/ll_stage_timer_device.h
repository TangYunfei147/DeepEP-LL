// Copyright (c) 2026, Lu Lu
#pragma once
#include "kernel_operator.h"
#include "ll_stage_timer_layout.h"
#include "ll_stage_timer_profile.h"
#include "ll_core_config.h"
#include "ll_overlap_stage.h"
namespace LlStageTimer {
// Exact dispatch copy accounting plus first-visit milestones. Observe adds no
// dependency; explicit completion mode isolates each local send-stage interval.
// All GM writes occur at exit; OFF/null-buffer paths have no diagnostic clocks.
class DeviceTimer {
public:
    __aicore__ inline void Init(GM_ADDR address, uint32_t operation, uint32_t mode) {
#ifdef DEEPEP_LL_STAGE_TIMER
        if (address && ((mode != 1U && mode != 6U) ||
                        (operation != 1U && mode != 1U) ||
                        (mode == 6U && DispatchDedup::kLlOverlapExperimentStage != 3U))) { AscendC::Trap(); return; }
        address_=address; operation_=operation; mode_=mode;
        if (address_) {
            valid_=0U; lanes_=0U; activeStage_=kBandwidthStageCount; intervalFirst_=0U; intervalOps_=0U;
            for(uint32_t i=0;i<kBandwidthStageCount;++i)
                for(uint32_t field=0;field<kBandwidthRecordWords;++field) bandwidth_[i][field]=0U;
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
    __aicore__ inline bool CompletionEnabled() const {
#ifdef DEEPEP_LL_STAGE_TIMER
        return address_ && operation_==1U && mode_==6U;
#else
        return false;
#endif
    }
    __aicore__ inline void BeginStage(BandwidthStage stage,AscendC::TPipe* pipe) {
#ifdef DEEPEP_LL_STAGE_TIMER
        if(!CompletionEnabled()) return;
        if(uint32_t(stage)>=kBandwidthStageCount || activeStage_!=kBandwidthStageCount) { AscendC::Trap(); return; }
        activeStage_=uint32_t(stage); intervalFirst_=0U; intervalOps_=0U;
        const auto event=pipe->FetchEventID(AscendC::HardEvent::MTE3_S);
        const uint64_t begin=AscendC::GetSystemCycle();
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(event);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(event);
        bandwidth_[uint32_t(stage)][PRE_DRAIN_TICKS]+=AscendC::GetSystemCycle()-begin;
#else
        (void)stage; (void)pipe;
#endif
    }
    __aicore__ inline void Transfer(BandwidthStage stage,bool peer,uint64_t bytes,uint64_t payloadBytes) {
#ifdef DEEPEP_LL_STAGE_TIMER
        if(!address_ || operation_!=1U) return;
        if(uint32_t(stage)>=kBandwidthStageCount || !bytes || payloadBytes>bytes ||
           (CompletionEnabled() && activeStage_!=uint32_t(stage))) { AscendC::Trap(); return; }
        auto* stat=bandwidth_[uint32_t(stage)];
        const uint64_t issue=AscendC::GetSystemCycle();
        if(!(stat[PEER_OPERATIONS]+stat[SELF_OPERATIONS])) stat[FIRST_ISSUE_TICK]=issue;
        stat[LAST_ISSUE_TICK]=issue;
        stat[peer ? PEER_BYTES:SELF_BYTES]+=bytes;
        stat[peer ? PEER_PAYLOAD_BYTES:SELF_PAYLOAD_BYTES]+=payloadBytes;
        ++stat[peer ? PEER_OPERATIONS:SELF_OPERATIONS];
        if(CompletionEnabled()) {
            if(!intervalOps_) intervalFirst_=issue;
            ++intervalOps_;
        }
#else
        (void)stage; (void)peer; (void)bytes; (void)payloadBytes;
#endif
    }
    __aicore__ inline void EndStage(BandwidthStage stage,AscendC::TPipe* pipe) {
#ifdef DEEPEP_LL_STAGE_TIMER
        if(!CompletionEnabled()) return;
        if(uint32_t(stage)>=kBandwidthStageCount || activeStage_!=uint32_t(stage)) { AscendC::Trap(); return; }
        if(intervalOps_) {
            auto* stat=bandwidth_[uint32_t(stage)];
            // A separate event domain, fetched only for immediately consumed
            // Set/Wait pairs. Never steals a packet lane's in-flight event.
            const auto event=pipe->FetchEventID(AscendC::HardEvent::MTE3_S);
            const uint64_t begin=AscendC::GetSystemCycle();
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(event);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(event);
            const uint64_t done=AscendC::GetSystemCycle();
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(event);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(event);
            const uint64_t emptyDone=AscendC::GetSystemCycle();
            stat[LAST_COMPLETION_TICK]=done;
            stat[COMPLETED_ELAPSED_TICKS]+=done-intervalFirst_;
            ++stat[COMPLETED_INTERVALS];
            stat[END_DRAIN_TICKS]+=done-begin;
            stat[EMPTY_CONTROL_TICKS]+=emptyDone-done;
            stat[COMPLETED_OPERATIONS]+=intervalOps_;
        }
        activeStage_=kBandwidthStageCount; intervalFirst_=0U; intervalOps_=0U;
#else
        (void)stage; (void)pipe;
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
    __aicore__ inline void Flush() {
#ifdef DEEPEP_LL_STAGE_TIMER
        if (!address_) return;
        if(activeStage_!=kBandwidthStageCount) { AscendC::Trap(); return; }
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
        for(uint32_t i=kRecordWordOffset+kPointCount;i<kBandwidthWordOffset;++i) words[i]=0U;
        for(uint32_t i=0;i<kBandwidthStageCount;++i)
            for(uint32_t field=0;field<kBandwidthRecordWords;++field)
                words[kBandwidthWordOffset+i*kBandwidthRecordWords+field]=bandwidth_[i][field];
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
    GM_ADDR address_=nullptr;
    uint32_t operation_=0U, mode_=1U, lanes_=0U;
    uint64_t valid_=0U;
    uint32_t activeStage_=kBandwidthStageCount;
    uint64_t intervalFirst_=0U,intervalOps_=0U;
    uint64_t bandwidth_[kBandwidthStageCount][kBandwidthRecordWords];
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
// Prior MTE3 wait probes are intentionally inactive in this purpose-specific schema.
#define LL_MTE3_WAIT_BEGIN(name) do {} while(false)
#define LL_MTE3_WAIT_END(name,metric) do {} while(false)
#define LL_MTE3_DRAIN_WHEN(condition,metric,mode,pipe) do {} while(false)
#define LL_MTE3_DRAIN(metric,mode,pipe) do {} while(false)
#ifdef DEEPEP_LL_STAGE_TIMER
#define LL_BW_BEGIN(stage,pipe) do { if(timer_ && timer_->CompletionEnabled()) timer_->BeginStage((stage),(pipe)); } while(false)
#define LL_BW_END(stage,pipe) do { if(timer_ && timer_->CompletionEnabled()) timer_->EndStage((stage),(pipe)); } while(false)
#define LL_BW_TRANSFER(stage,peer,bytes,payload) do { if(timer_ && timer_->Enabled()) timer_->Transfer((stage),(peer),(bytes),(payload)); } while(false)
// Isolate the final queued first-hop input before closing its completion span.
// In OFF/observe the original one-lookahead stream remains continuous.
#define LL_BW_SPLIT_INPUT(input,pipe) do { if(timer_ && timer_->CompletionEnabled()) { EndInputPrefetch(input); timer_->EndStage(LlStageTimer::REMOTE_FIRST_HOP,(pipe)); timer_->BeginStage(LlStageTimer::SOURCE_DIRECT,(pipe)); BeginInputPrefetch(); } } while(false)
#define LL_MTE3_SEND_LANES(lanes,overlap) do { if(timer_) timer_->SendLanes(lanes,overlap); } while(false)
#define LL_MTE3_RECV_LANES(lanes,overlap) do { if(timer_) timer_->RecvLanes(lanes,overlap); } while(false)
#define LL_SPARSE_MARK(point) do { if(timer_) timer_->Mark<LlStageTimer::point>(); } while(false)
#define LL_SPARSE_WHEN(condition,point) do { if(timer_ && (condition)) timer_->Mark<LlStageTimer::point>(); } while(false)
#define LL_SPARSE_AFTER(observed,point) do { if(timer_) timer_->MarkIfSeen<LlStageTimer::observed,LlStageTimer::point>(); } while(false)
#else
#define LL_BW_BEGIN(stage,pipe) do {} while(false)
#define LL_BW_END(stage,pipe) do {} while(false)
#define LL_BW_TRANSFER(stage,peer,bytes,payload) do {} while(false)
#define LL_BW_SPLIT_INPUT(input,pipe) do {} while(false)
#define LL_MTE3_SEND_LANES(lanes,overlap) do {} while(false)
#define LL_MTE3_RECV_LANES(lanes,overlap) do {} while(false)
#define LL_SPARSE_MARK(point) do {} while(false)
#define LL_SPARSE_WHEN(condition,point) do {} while(false)
#define LL_SPARSE_AFTER(observed,point) do {} while(false)
#endif
