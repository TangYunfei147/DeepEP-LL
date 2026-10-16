#include "kernel_operator.h"
#include "../../csrc/kernels/dispatch/low_latency/common/ll_stage_timer_device.h"
using namespace AscendC;
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void ll_timer_smoke(GM_ADDR output, uint32_t mode)
{
    LlStageTimer::DeviceTimer timer;
    timer.Init(output, 1U, mode);
    timer.Start(LlStageTimer::Kernel);
    // 嵌套与重复区间，验证每核独立槽、累计次数及关闭模式无写入。
    for (uint32_t i = 0; i < 3; ++i) {
        timer.Start(LlStageTimer::RecvWait);
        volatile uint64_t cycle = GetSystemCycle();
        (void)cycle;
        timer.End(LlStageTimer::RecvWait);
    }
    for (uint32_t stage = LlStageTimer::Magic; stage < LlStageTimer::StageCount; ++stage) {
        if (stage == LlStageTimer::RecvWait) continue;
        timer.Start(static_cast<LlStageTimer::Stage>(stage));
        volatile uint64_t cycle = GetSystemCycle();
        (void)cycle;
        timer.End(static_cast<LlStageTimer::Stage>(stage));
    }
    timer.End(LlStageTimer::Kernel);
    timer.Finish();
}
extern "C" void ll_timer_smoke_do(void* stream, uint8_t* output, uint32_t mode)
{
    ll_timer_smoke<<<16, nullptr, stream>>>(output, mode);
}
