// Copyright (c) 2026, Lu Lu
#pragma once
#include "kernel_operator.h"
#include "ll_stage_timer_layout.h"
#include "ll_stage_timer_profile.h"
#include "ll_core_config.h"
namespace LlStageTimer {
// No per-packet metrics, stage lookup, interval arrays, or diagnostic barriers.
// A point reads the clock only on its first visit. All GM writes occur at exit.
class DeviceTimer {
public:
    __aicore__ inline void Init(GM_ADDR address, uint32_t operation, uint32_t mode) {
#ifdef DEEPEP_LL_STAGE_TIMER
        if (address && mode != 1U) { AscendC::Trap(); return; }
        address_=address; operation_=operation;
        if (address_) valid_=0U;
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
    __aicore__ inline void Flush() {
#ifdef DEEPEP_LL_STAGE_TIMER
        if (!address_) return;
        const uint32_t core=AscendC::GetBlockIdx();
        auto* words=reinterpret_cast<__gm__ uint64_t*>(address_)+core*kWordsPerCore;
        const uint32_t role=operation_==1U ? (core<DispatchDedup::kLlSendCoreNum ? 3U:2U)
                                          : (core<DispatchDedup::kLlSendCoreNum ? 1U:4U);
        words[0]=kMagic; words[1]=kSchema; words[2]=operation_; words[3]=AscendC::GetBlockNum();
        words[4]=core; words[5]=1U | (uint64_t(DispatchDedup::kLlSendCoreNum)<<8U) |
            (uint64_t(role)<<16U) | (uint64_t(kProfileId)<<24U);
        words[6]=valid_; words[7]=0U;
        for(uint32_t i=0;i<kPointCount;++i)
            words[kRecordWordOffset+i]=(valid_ & (uint64_t(1U)<<i)) ? ticks_[i]:0U;
        for(uint32_t i=kRecordWordOffset+kPointCount;i<kExportWords;++i) words[i]=0U;
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
    uint32_t operation_=0U;
    uint64_t valid_=0U;
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
#define LL_SPARSE_MARK(point) do { if(timer_) timer_->Mark<LlStageTimer::point>(); } while(false)
#define LL_SPARSE_AFTER(observed,point) do { if(timer_) timer_->MarkIfSeen<LlStageTimer::observed,LlStageTimer::point>(); } while(false)
#else
#define LL_SPARSE_MARK(point) do {} while(false)
#define LL_SPARSE_AFTER(observed,point) do {} while(false)
#endif
