// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// Adapted from graph_dc_debug_seg@e58ece1, mc2/common/op_kernel/dc_seg_profiler.h.
// Licensed under CANN Open Software License Agreement Version 2.0;
// see docs/licenses/CANN-Open-Software-License-2.0.txt. Provided AS IS without warranties.
#pragma once
#include "kernel_operator.h"
#include "ll_stage_timer_layout.h"
#include "ll_core_config.h"
namespace LlStageTimer {
// Reference semantics: no boundary barriers, local accumulation, first two
// intervals per stage, reserved KERNEL pair, and GM export after all-core sync.
class DeviceTimer {
public:
    __aicore__ inline void Init(GM_ADDR address, uint32_t operation, uint32_t mode) {
#ifdef DEEPEP_LL_STAGE_TIMER
        if (address && mode != 1U) { AscendC::Trap(); return; }
        address_ = address; operation_ = operation;
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
    __aicore__ inline uint64_t Begin(Stage stage) const {
        (void)stage;
#ifdef DEEPEP_LL_STAGE_TIMER
        return Enabled() ? AscendC::GetSystemCycle() : 0U;
#else
        return 0U;
#endif
    }
    __aicore__ inline void Count(Stage stage, uint32_t count = 1U) {
#ifdef DEEPEP_LL_STAGE_TIMER
        if (Enabled()) calls_[stage] += count;
#else
        (void)stage; (void)count;
#endif
    }
    __aicore__ inline void End(Stage stage, uint64_t begin) {
#ifdef DEEPEP_LL_STAGE_TIMER
        if (!Enabled()) return;
        const uint64_t end = AscendC::GetSystemCycle();
        sums_[stage] += end - begin;
        ++calls_[stage];
        if (calls_[stage] <= 2U && count_ + 2U <= kTraceRecords - (stage == KERNEL ? 0U : 2U)) {
            events_[count_] = uint32_t(stage) * 2U; cycles_[count_++] = begin;
            events_[count_] = uint32_t(stage) * 2U + 1U; cycles_[count_++] = end;
        } else { ++omitted_; }
#else
        (void)stage; (void)begin;
#endif
    }
    __aicore__ inline void Flush() {
#ifdef DEEPEP_LL_STAGE_TIMER
        if (!Enabled()) return;
        const uint32_t core = AscendC::GetBlockIdx();
        auto* words = reinterpret_cast<__gm__ uint64_t*>(address_) + core * kWordsPerCore;
        for (uint32_t i = 0; i < count_; ++i) Write(words + kRecordWordOffset + i * 2U, events_[i], cycles_[i]);
        uint32_t records = count_;
        for (uint32_t stage = 0; stage < kMetrics; ++stage) {
            if (calls_[stage]) Write(words + kRecordWordOffset + records++ * 2U,
                (uint64_t(calls_[stage]) << 32U) | (0x8000U + stage), sums_[stage]);
        }
        // Commit header last. No device-side GM record writes occur before Flush.
        const uint32_t role = operation_ == 1U ? (core < DispatchDedup::kLlSendCoreNum ? 3U : 2U)
                                               : (core < DispatchDedup::kLlSendCoreNum ? 1U : 4U);
        words[0]=kMagic; words[1]=kSchema; words[2]=operation_; words[3]=AscendC::GetBlockNum();
        words[4]=core; words[5]=1U | (uint64_t(DispatchDedup::kLlSendCoreNum) << 8U) | (uint64_t(role) << 16U);
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
    uint32_t operation_ = 0, count_ = 0, omitted_ = 0;
    uint64_t sums_[kMetrics]{};
    uint32_t calls_[kMetrics]{};
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
}
#define LL_SEG_JOIN_IMPL(a,b) a##b
#define LL_SEG_JOIN(a,b) LL_SEG_JOIN_IMPL(a,b)
#define LL_SEG_SCOPE(stage) LlStageTimer::Scope LL_SEG_JOIN(llSegScope,__LINE__)(timer_, LlStageTimer::stage)
#define LL_SEG_BEGIN(name, stage) const uint64_t name = timer_ ? timer_->Begin(LlStageTimer::stage) : 0U
#define LL_SEG_END(name, stage) do { if (timer_) timer_->End(LlStageTimer::stage, name); } while (false)
#define LL_SEG_COUNT(stage, count) do { if (timer_) timer_->Count(LlStageTimer::stage, count); } while (false)
