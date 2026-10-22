// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// Adapted from graph_dc_debug_seg@e58ece1, mc2/common/op_kernel/dc_seg_profiler.h.
// Licensed under CANN Open Software License Agreement Version 2.0;
// see docs/licenses/CANN-Open-Software-License-2.0.txt. Provided AS IS without warranties.
#pragma once
#include "kernel_operator.h"
#include "ll_stage_timer_layout.h"
// Bisheng also parses these non-template device classes during its Host pass.
// Match helper qualifiers to this header's device API, not pass-local macros.
#define DEEPEP_LL_TIMER_DEVICE_CONTEXT 1
#include "ll_stage_timer_profile.h"
#undef DEEPEP_LL_TIMER_DEVICE_CONTEXT
#include "ll_core_config.h"
namespace LlStageTimer {
// Reference semantics: no boundary barriers, local accumulation, first two
// intervals per stage, reserved KERNEL pair, and GM export after all-core sync.
class DeviceTimer {
public:
    __aicore__ inline void Init(GM_ADDR address, uint32_t operation, uint32_t mode) {
#ifdef DEEPEP_LL_STAGE_TIMER
        if (address && ((mode != 1U && mode != 3U) || (mode == 3U && (operation != 1U || (kProfileId != 0U && kProfileId != 5U))))) { AscendC::Trap(); return; }
        address_ = address; operation_ = operation; mode_ = mode;
        // Ordinary calls in an ON wheel pass null. Do not zero the diagnostic
        // arrays on that path, and allow the object to be reused safely.
        if (address_) {
            count_=0U; omitted_=0U; kernelBegin_=0U;
            for (uint32_t i=0;i<kProfileMetrics;++i) { sums_[i]=0U; calls_[i]=0U; }
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
    __aicore__ inline bool RelayCompletion() const {
#ifdef DEEPEP_LL_STAGE_TIMER
        return Enabled() && mode_ == 3U;
#else
        return false;
#endif
    }
    __aicore__ inline uint64_t Begin(Stage stage) {
        (void)stage;
#ifdef DEEPEP_LL_STAGE_TIMER
        const uint64_t tick=Enabled() && ProfileEnabled(stage) ? AscendC::GetSystemCycle() : 0U;
        if (Enabled() && stage==KERNEL) kernelBegin_=tick;
        return tick;
#else
        return 0U;
#endif
    }
    __aicore__ inline void Count(Stage stage, uint32_t count = 1U) {
#ifdef DEEPEP_LL_STAGE_TIMER
        if (Enabled() && ProfileEnabled(stage)) calls_[ProfileIndex(stage)] += count;
#else
        (void)stage; (void)count;
#endif
    }
    __aicore__ inline void End(Stage stage, uint64_t begin) {
#ifdef DEEPEP_LL_STAGE_TIMER
        if (!Enabled() || !ProfileEnabled(stage)) return;
        const uint64_t end = AscendC::GetSystemCycle();
        const uint32_t index=ProfileIndex(stage);
        sums_[index] += end - begin;
        ++calls_[index];
        uint32_t reserved=0U;
        const Stage important[]={KERNEL,SEND_SUBMIT_END,BEFORE_FIRST_SEND,BEFORE_FIRST_RECV_POLL};
        for (uint32_t i=0;i<(operation_==1U?4U:1U);++i)
            if (stage!=important[i] && ProfileEnabled(important[i]) && calls_[ProfileIndex(important[i])]==0U) reserved+=2U;
        if (calls_[index] <= 2U && count_ + 2U <= kTraceRecords-reserved) {
            events_[count_] = uint32_t(stage) * 2U; cycles_[count_++] = begin;
            events_[count_] = uint32_t(stage) * 2U + 1U; cycles_[count_++] = end;
        } else { ++omitted_; }
#else
        (void)stage; (void)begin;
#endif
    }
    // One real elapsed interval from this core's KERNEL start. Empty workers
    // have no milestone; never infer zero or use cross-core clock subtraction.
    __aicore__ inline void First(Stage stage) {
#ifdef DEEPEP_LL_STAGE_TIMER
        if (Enabled() && ProfileEnabled(stage) && calls_[ProfileIndex(stage)]==0U) End(stage,kernelBegin_);
#else
        (void)stage;
#endif
    }
    __aicore__ inline void Flush() {
#ifdef DEEPEP_LL_STAGE_TIMER
        if (!Enabled()) return;
        const uint32_t core = AscendC::GetBlockIdx();
        auto* words = reinterpret_cast<__gm__ uint64_t*>(address_) + core * kWordsPerCore;
        for (uint32_t i = 0; i < count_; ++i) Write(words + kRecordWordOffset + i * 2U, events_[i], cycles_[i]);
        uint32_t records = count_, index = 0U;
        for (uint32_t stage = 0; stage < kMetrics; ++stage) {
            if (!ProfileEnabled(Stage(stage))) continue;
            if (calls_[index]) Write(words + kRecordWordOffset + records++ * 2U,
                (uint64_t(calls_[index]) << 32U) | (0x8000U + stage), sums_[index]);
            ++index;
        }
        // Commit header last. No device-side GM record writes occur before Flush.
        const uint32_t role = operation_ == 1U ? (core < DispatchDedup::kLlSendCoreNum ? 3U : 2U)
                                               : (core < DispatchDedup::kLlSendCoreNum ? 1U : 4U);
        words[0]=kMagic; words[1]=kSchema; words[2]=operation_; words[3]=AscendC::GetBlockNum();
        words[4]=core; words[5]=mode_ | (uint64_t(DispatchDedup::kLlSendCoreNum) << 8U) |
            (uint64_t(role) << 16U) | (uint64_t(kProfileId) << 24U);
        words[6]=records; words[7]=omitted_;
        FlushLine(words);
#endif
    }
private:
#ifdef DEEPEP_LL_STAGE_TIMER
    __aicore__ inline void FlushLine(__gm__ uint64_t* words) {
        AscendC::GlobalTensor<uint32_t> tensor;
        tensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(words), 16U);
        AscendC::DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                          AscendC::DcciDst::CACHELINE_OUT>(tensor);
    }
    __aicore__ inline void Write(__gm__ uint64_t* words, uint64_t event, uint64_t value) {
        words[0]=event; words[1]=value; FlushLine(words);
    }
    GM_ADDR address_ = nullptr;
    uint32_t mode_ = 1U, operation_ = 0, count_ = 0, omitted_ = 0;
    uint64_t kernelBegin_=0U;
    uint64_t sums_[kProfileMetrics];
    uint32_t calls_[kProfileMetrics];
    uint64_t cycles_[kTraceRecords];
    uint32_t events_[kTraceRecords];
#endif
};
class Scope {
public:
    __aicore__ inline Scope(DeviceTimer* timer, Stage stage)
        : timer_(timer), stage_(stage), begin_(timer ? timer->Begin(stage) : 0U) {}
    __aicore__ inline ~Scope() { if (timer_) timer_->End(stage_, begin_); }
private:
    DeviceTimer* timer_; Stage stage_; uint64_t begin_;
};
template<Stage stage, bool enabled = ProfileEnabled(stage)> class ProfileScope {
public:
    __aicore__ inline ProfileScope(DeviceTimer*) {}
};
template<Stage stage> class ProfileScope<stage,true> {
public:
    __aicore__ inline ProfileScope(DeviceTimer* timer)
        : timer_(timer), begin_(timer ? timer->Begin(stage) : 0U) {}
    __aicore__ inline ~ProfileScope() { if (timer_) timer_->End(stage,begin_); }
private:
    DeviceTimer* timer_; uint64_t begin_;
};
// Construct BEFORE the issue scope: destruction then follows issue End.
// Ordinary observe captures do not execute either synchronization.
class RelayCompletionScope {
public:
    __aicore__ inline RelayCompletionScope(DeviceTimer* timer, AscendC::TPipe* pipe)
        : timer_(timer),pipe_(pipe),active_(ProfileEnabled(RELAY_COMPLETE) && timer && timer->RelayCompletion()) {
        if (active_) {
            const uint64_t prior=timer_->Begin(RELAY_PRE_DRAIN);
            Drain();
            timer_->End(RELAY_PRE_DRAIN,prior);
            begin_=timer_->Begin(RELAY_COMPLETE);
        }
    }
    __aicore__ inline ~RelayCompletionScope() {
        if (active_) {
            Drain();
            timer_->End(RELAY_COMPLETE,begin_);
        }
    }
private:
    __aicore__ inline void Drain() {
        const auto event=pipe_->FetchEventID(AscendC::HardEvent::MTE3_S);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(event);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(event);
    }
    DeviceTimer* timer_;
    AscendC::TPipe* pipe_;
    bool active_;
    uint64_t begin_=0U;
};
}
#define LL_SEG_JOIN_IMPL(a,b) a##b
#define LL_SEG_JOIN(a,b) LL_SEG_JOIN_IMPL(a,b)
#define LL_SEG_SCOPE(stage) LlStageTimer::ProfileScope<LlStageTimer::stage> LL_SEG_JOIN(llSegScope,__LINE__)(timer_)
#define LL_SEG_BEGIN(name, stage) const uint64_t name = LlStageTimer::ProfileEnabled(LlStageTimer::stage) && timer_ ? timer_->Begin(LlStageTimer::stage) : 0U
#define LL_SEG_END(name, stage) do { if constexpr (LlStageTimer::ProfileEnabled(LlStageTimer::stage)) { if (timer_) timer_->End(LlStageTimer::stage, name); } } while (false)
#define LL_SEG_COUNT(stage, count) do { if constexpr (LlStageTimer::ProfileEnabled(LlStageTimer::stage)) { if (timer_) timer_->Count(LlStageTimer::stage, count); } } while (false)
#define LL_SEG_RELAY_COMPLETION(pipe) LlStageTimer::RelayCompletionScope LL_SEG_JOIN(llRelayScope,__LINE__)(timer_, pipe)

#define LL_SEG_FIRST(stage) do { if constexpr (LlStageTimer::ProfileEnabled(LlStageTimer::stage)) { if (timer_) timer_->First(LlStageTimer::stage); } } while (false)
