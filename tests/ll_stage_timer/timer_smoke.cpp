#include "kernel_operator.h"
#include "../../csrc/kernels/dispatch/low_latency/common/ll_stage_timer_device.h"
using namespace AscendC;
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void ll_timer_smoke(GM_ADDR output, uint32_t mode)
{
    LlStageTimer::DeviceTimer timer;
    timer.Init(output, 1U, mode);
    const uint64_t kernelBegin=timer.Begin(LlStageTimer::KERNEL);
    for (uint32_t i=0;i<3;++i) {
        const uint64_t begin=timer.Begin(LlStageTimer::RECV_POLL);
        volatile uint64_t cycle=GetSystemCycle(); (void)cycle;
        timer.End(LlStageTimer::RECV_POLL,begin);
    }
    timer.Count(LlStageTimer::POLL_RECV_COUNTS_ITERATIONS,3U);
    const uint64_t drainBegin=timer.Begin(LlStageTimer::FINAL_DRAIN);
    PipeBarrier<PIPE_ALL>();
    timer.End(LlStageTimer::FINAL_DRAIN,drainBegin);
    timer.End(LlStageTimer::KERNEL,kernelBegin);
    // This primitive smoke has no configured inter-core FFTS context. Production
    // entries perform SyncAll<true>() here; this smoke checks buffer/record logic.
    timer.Flush();
}
extern "C" void ll_timer_smoke_do(void* stream, uint8_t* output, uint32_t mode)
{
    ll_timer_smoke<<<DispatchDedup::kLlActiveCoreNum, nullptr, stream>>>(output, mode);
}
