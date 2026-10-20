// Copyright (c) 2026, Lu Lu
// Modified by ryan_li 2026

#include <cstdint>

#include "kernel_operator.h"
#include "shmem.h"

#include "../launch.hpp"

#if CATLASS_ARCH == 3510
#include "low_latency/dispatch_server_dedup_kernel.h"
#include "low_latency/common/dispatch_dedup_types.h"
#include "low_latency/common/ep_memory_server_dedup_tiling.h"

// 中文审计：完整dispatch性能入口见tests/elastic/test_ep_low_latency.py。
// 调用链：ElasticBuffer._dispatch_low_latency → csrc/ops/low_latency.cpp
// → dispatch_server_dedup_kernel_do → 本设备入口 → DispatchMemoryServerDedupKernel。
// profiler匹配本入口任务名；下层计数/发送/接收是同一kernel内的阶段，不是独立task。
// One production entry and a queried pure-AIV launch with a 1:1 send/recv split.
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void dispatch_server_dedup_kernel(
    GM_ADDR commArgs, GM_ADDR x, GM_ADDR expertIds, GM_ADDR expertScales, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,
    GM_ADDR expertTokenNumsOut, GM_ADDR sendCountsOut, GM_ADDR tokenTypeOut, GM_ADDR destinationIndexOut,
    GM_ADDR relayReadIndexOut, GM_ADDR sourceMaskOut, GM_ADDR expertScalesOut, GM_ADDR tilingData,
    int64_t reservedMagic, GM_ADDR inputScales, GM_ADDR stageTimer, uint32_t timerMode, GM_ADDR expertTimer)
{
    util_set_ffts_config(reinterpret_cast<const __gm__ DispatchDedup::CommArgs*>(commArgs)->fftsVal);
    LlStageTimer::DeviceTimer timer;
    timer.Init(stageTimer, 1U, timerMode);
    const uint64_t initBegin = timer.Begin(LlStageTimer::INIT);
    // Host owns validation; each core reads its configuration once before work.
    const auto* tiling = reinterpret_cast<const __gm__ DispatchDedup::EpServerDedupTilingData*>(tilingData);
    const uint32_t bs = tiling->bs;
    const uint32_t h = tiling->h;
    const bool enableDedup = DispatchDedup::EpServerDedupEnabled(tiling->numMaxTokensPerRank);
    const uint32_t topK = tiling->topK;
    const uint32_t moeExpertNum = tiling->moeExpertNum;
    (void)expertTimer; // Native host rejects legacy per-expert instrumentation.
    const uint32_t expertTokenNumsType = tiling->expertTokenNumsType;
    const uint8_t quantMode = tiling->quantMode;
    const uint8_t dtype = tiling->dtype;
    const uint8_t expandXOutDtype = tiling->expandXOutDtype;
    const auto inputMode = static_cast<DispatchDedup::EpServerDedupInputMode>(tiling->inputMode);
    AscendC::TPipe pipe;
    // 下列模板分支决定输入类型、线上FP8格式与是否已预量化；比较性能时须固定这些条件。
#define DISPATCH_SERVER_DEDUP_RUN(XType, WireFp8, E4M3, Prequantized)                                                  \
    DispatchDedup::DispatchMemoryServerDedupKernel<XType, WireFp8, E4M3, Prequantized> kernel;                         \
    kernel.Init(commArgs, x, expertIds, expertScales, expandXOut, dynamicScalesOut, expertTokenNumsOut, sendCountsOut, \
                tokenTypeOut, destinationIndexOut, relayReadIndexOut, sourceMaskOut, expertScalesOut, bs, h, topK,     \
                moeExpertNum, expertTokenNumsType, &pipe, inputScales, enableDedup, &timer, nullptr);                           \
    timer.End(LlStageTimer::INIT, initBegin);                                                                               \
    const uint64_t kernelBegin = timer.Begin(LlStageTimer::KERNEL); \
    kernel.Process(); \
    timer.End(LlStageTimer::KERNEL, kernelBegin)
    if (inputMode == DispatchDedup::EpServerDedupInputMode::PrequantizedFp8Packs && quantMode == 4 &&
        dtype == DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_FP8E4M3 &&
        expandXOutDtype == DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_FP8E4M3) {
        DISPATCH_SERVER_DEDUP_RUN(uint8_t, true, true, true);
    } else if (dtype == DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_FP16) {
        if (quantMode == 0) {
            DISPATCH_SERVER_DEDUP_RUN(half, false, false, false);
        } else if (expandXOutDtype == DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_FP8E4M3) {
            DISPATCH_SERVER_DEDUP_RUN(half, true, true, false);
        } else {
            DISPATCH_SERVER_DEDUP_RUN(half, true, false, false);
        }
    } else if (dtype == DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_BFP16) {
        if (quantMode == 0) {
            DISPATCH_SERVER_DEDUP_RUN(bfloat16_t, false, false, false);
        } else if (expandXOutDtype == DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_FP8E4M3) {
            DISPATCH_SERVER_DEDUP_RUN(bfloat16_t, true, true, false);
        } else {
            DISPATCH_SERVER_DEDUP_RUN(bfloat16_t, true, false, false);
        }
    }
#undef DISPATCH_SERVER_DEDUP_RUN
    if (timer.Enabled()) { AscendC::SyncAll<true>(); timer.Flush(); }
}

// Same in-library launch mechanism as all other ascendEP kernels.
#include "low_latency/launch.hpp"
extern "C" DEEPEP_EXPORT uint32_t deepep_ll_stage_timer_enabled()
{
#ifdef DEEPEP_LL_STAGE_TIMER
    return 1U;
#else
    return 0U;
#endif
}
extern "C" DEEPEP_EXPORT uint32_t deepep_ll_ranks_per_group()
{
    return DispatchDedup::kLlRanksPerGroup;
}

extern "C" DEEPEP_EXPORT uint32_t deepep_ll_core_protocol_version()
{
    return DispatchDedup::kLlCoreProtocolVersion;
}

extern "C" DEEPEP_EXPORT void dispatch_server_dedup_kernel_do(void* stream, const deepep::EpServerDedupKernelArgs* args)
{
    const auto& a = *args;
    dispatch_server_dedup_kernel<<<a.coreNum, DEEPEP_KERNEL_LAUNCH_UB(0), stream>>>(
        a.commArgs, a.x, a.expertIds, a.expertScales, a.expandXOut, a.dynamicScalesOut, a.expertTokenNumsOut,
        a.sendCountsOut, a.tokenTypeOut, a.destinationIndexOut, a.relayReadIndexOut, a.sourceMaskOut, a.expertScalesOut,
        a.tilingData, a.reservedMagic, a.inputScales, a.stageTimer, a.timerMode, a.expertTimer);
}

#endif
