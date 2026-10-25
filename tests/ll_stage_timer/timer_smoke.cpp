#include "kernel_operator.h"
#include "../../csrc/kernels/dispatch/low_latency/common/ll_stage_timer_device.h"
using namespace AscendC;
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void ll_timer_smoke(GM_ADDR output, uint32_t mode)
{
#ifdef DEEPEP_LL_STAGE_TIMER
    const uint64_t sparseEntry=output ? GetSystemCycle():0U;
#endif
    LlStageTimer::DeviceTimer timer;
    timer.Init(output, 1U, mode);
#ifdef DEEPEP_LL_STAGE_TIMER
    timer.MarkAt<LlStageTimer::ENTRY>(sparseEntry);
#endif
    timer.Mark<LlStageTimer::INIT_DONE>();
    for (uint32_t i=0;i<3;++i) {
        timer.Mark<LlStageTimer::FIRST_RECV_PROBE>();
        volatile uint64_t cycle=GetSystemCycle(); (void)cycle;
    }
    timer.Mark<LlStageTimer::FIRST_RECV_READY>();
    timer.Mark<LlStageTimer::RECV_DONE>();
    timer.Mark<LlStageTimer::WORK_DONE>();
    // Per-core independent output is consumed only after kernel completion.
    timer.Flush();
}
extern "C" void ll_timer_smoke_do(void* stream, uint8_t* output, uint32_t mode)
{
    ll_timer_smoke<<<DispatchDedup::kLlActiveCoreNum, nullptr, stream>>>(output, mode);
}
