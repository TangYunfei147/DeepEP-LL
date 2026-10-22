// Copyright (c) 2026, Lu Lu
// Modified by ryan_li 2026

#include <cstdint>

#include "kernel_operator.h"
#include "shmem.h"

#include "../launch.hpp"

#if CATLASS_ARCH == 3510
#include "low_latency/combine_server_dedup_kernel.h"
#include "low_latency/ep_memory_server_dedup_combine_tiling.h"
#include "../dispatch/low_latency/device/dispatch_server_dedup_magic.h"
#include "low_latency/launch.hpp"

// 中文审计：ElasticBuffer._combine_low_latency → csrc/ops/low_latency.cpp
// → combine_server_dedup_kernel_do → 本入口。输入依赖匹配dispatch handle的路由/计数/权重。
// profiler中的combine_server_dedup_kernel包含下层准备、控制通知、回传与最终归约。
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void combine_server_dedup_kernel(
    GM_ADDR commArgs, GM_ADDR expertOut, GM_ADDR destinationIndex, GM_ADDR sendCounts, GM_ADDR expertScales,
    GM_ADDR tokenType, GM_ADDR relayReadIndex, GM_ADDR sourceMask, GM_ADDR yOut, GM_ADDR tilingData, GM_ADDR stageTimer,
    uint32_t timerMode)
{
    const auto* args = reinterpret_cast<const __gm__ DispatchDedup::CommArgs*>(commArgs);
    util_set_ffts_config(args->fftsVal);
    LlStageTimer::DeviceTimer timer;
    timer.Init(stageTimer, 2U, timerMode);
    timer.Start(LlStageTimer::Kernel);
    timer.Start(LlStageTimer::Magic);
    uint32_t magic;
    {
        AscendC::TPipe pipe;
        // 与dispatch共用每核窗口轮转规则；即使本核没有有效token也必须参加本次轮转。
        magic = DispatchDedup::RotateDispatchServerDedupMagic(args, AscendC::GetBlockIdx(), &pipe);
    }
    timer.End(LlStageTimer::Magic);
    timer.Start(LlStageTimer::WorkerInit);
    const auto* td = reinterpret_cast<const volatile __gm__ CombineDedup::EpServerDedupCombineTilingData*>(tilingData);
    const bool enableDedup = DispatchDedup::EpServerDedupEnabled(td->numMaxTokensPerRank, td->bs, uint32_t(args->rankSize));
    if (!DispatchDedup::LlTilingKeyValid(args->rankSize, td->tilingKey)) { AscendC::Trap(); return; }
#define COMBINE_RUN(Group) \
    CombineDedup::CombineMemoryServerDedupKernel<Group> kernel; \
    kernel.Init(commArgs, expertOut, destinationIndex, sendCounts, expertScales, tokenType, relayReadIndex, sourceMask, \
                yOut, td->bs, td->h, td->topK, td->moeExpertNum, td->dtype, 0, magic, enableDedup, &timer); \
    timer.End(LlStageTimer::WorkerInit); \
    kernel.Process()
    if (td->tilingKey == uint8_t(DispatchDedup::LlTilingKey::Group4)) { COMBINE_RUN(4U); }
    else { COMBINE_RUN(16U); }
#undef COMBINE_RUN
    timer.End(LlStageTimer::Kernel);
    timer.Finish();
}
extern "C" DEEPEP_EXPORT void combine_server_dedup_kernel_do(void* stream,
                                                             const deepep::EpServerDedupCombineKernelArgs* args)
{
    const auto& a = *args;
    combine_server_dedup_kernel<<<a.coreNum, DEEPEP_KERNEL_LAUNCH_UB(0), stream>>>(
        a.commArgs, a.expertOut, a.destinationIndex, a.sendCounts, a.expertScales, a.tokenType, a.relayReadIndex,
        a.sourceMask, a.yOut, a.tilingData, a.stageTimer, a.timerMode);
}
#endif
