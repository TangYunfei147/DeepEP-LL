#pragma once
#include "kernel_operator.h"
#include "ll_stage_timer_layout.h"

namespace LlStageTimer {
// 默认编译关闭。observe不插入额外流水屏障；drain显式串行化边界，只用于诊断。
// Schema 2 GM阶段布局：[first_start,start_or_last_end,sum_ticks,count|active]。
// 重复区间只累计同一核自己的记录；first..last包含间隙，不应作为有效阶段时间。
class DeviceTimer {
public:
    __aicore__ inline void Init(GM_ADDR address, uint32_t operation, uint32_t mode)
    {
#ifdef DEEPEP_LL_STAGE_TIMER
        address_ = address;
        mode_ = mode;
        if (!address_) return;
        const uint32_t core = AscendC::GetBlockIdx();
        data_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(address) + core * kWordsPerCore);
        data_(0) = kMagic;
        data_(1) = kSchema;
        data_(2) = operation;
        data_(3) = AscendC::GetBlockNum();
        data_(4) = core;
        data_(5) = mode;
        data_(6) = 0;
        data_(7) = StageCount;
#endif
    }
    __aicore__ inline void Start(Stage stage)
    {
#ifdef DEEPEP_LL_STAGE_TIMER
        if (!address_) return;
        if (mode_ == 2U) AscendC::PipeBarrier<PIPE_ALL>();
        const uint64_t cycle = AscendC::GetSystemCycle();
        const uint32_t row = 8U + (uint32_t(stage) - 1U) * 4U;
        if (data_(row + 3U) == 0U) data_(row) = cycle;
        data_(row + 1U) = cycle;
        data_(row + 3U) = data_(row + 3U) | kActive;
#endif
    }
    __aicore__ inline void End(Stage stage)
    {
#ifdef DEEPEP_LL_STAGE_TIMER
        if (!address_) return;
        if (mode_ == 2U) AscendC::PipeBarrier<PIPE_ALL>();
        const uint64_t cycle = AscendC::GetSystemCycle();
        const uint32_t row = 8U + (uint32_t(stage) - 1U) * 4U;
        const uint64_t start = data_(row + 1U);
        data_(row + 1U) = cycle;
        data_(row + 2U) = data_(row + 2U) + cycle - start;
        data_(row + 3U) = (data_(row + 3U) & ~kActive) + 1U;
#endif
    }
    __aicore__ inline void Finish()
    {
#ifdef DEEPEP_LL_STAGE_TIMER
        if (address_) data_(6) = 1U;
#endif
    }

private:
#ifdef DEEPEP_LL_STAGE_TIMER
    GM_ADDR address_ = nullptr;
    uint32_t mode_ = 1U;
    AscendC::GlobalTensor<uint64_t> data_;
#endif
};
}  // namespace LlStageTimer
