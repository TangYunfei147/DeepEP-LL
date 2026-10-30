#pragma once
// Copyright (c) 2026, Lu Lu
// Modified by ryan_li 2026

#include <cstdint>

#include "kernel_operator.h"
#include "shmem.h"

#include "../../launch.hpp"

#if CATLASS_ARCH == 3510
#include "common/ll_dispatch_variant.h"
#include "launch.hpp"
#include "common/ll_reference_profile_device.h"
#if DEEPEP_LL_DISPATCH_CB59 == 4
#include "cb59_small/dispatch_server_dedup_kernel.h"
#elif DEEPEP_LL_DISPATCH_CB59 == 3
#include "cb59_pair/kernel.h"
#elif DEEPEP_LL_DISPATCH_CB59 == 2
#include "cb59_stream/kernel.h"
#elif DEEPEP_LL_DISPATCH_CB59
#include "cb59/dispatch_server_dedup_kernel.h"
#else
#include "dispatch_server_dedup_kernel.h"
#endif
#include "common/dispatch_dedup_types.h"
#include "common/ep_memory_server_dedup_tiling.h"

#define LL_VARIANT_JOIN_(a,b) a##b
#define LL_VARIANT_JOIN(a,b) LL_VARIANT_JOIN_(a,b)
#define LL_VARIANT_KERNEL LL_VARIANT_JOIN(dispatch_server_dedup_kernel_,DEEPEP_LL_VARIANT_NAME)
#define LL_VARIANT_LAUNCH LL_VARIANT_JOIN(dispatch_server_dedup_launch_reference7_,DEEPEP_LL_VARIANT_NAME)
namespace {
#if DEEPEP_LL_VARIANT_MODE == 0
using LlVariantInput = uint8_t;
constexpr bool kWireFp8 = true, kE4M3 = true, kPrequantized = true;
#elif DEEPEP_LL_VARIANT_MODE == 1
using LlVariantInput = half;
constexpr bool kWireFp8 = false, kE4M3 = false, kPrequantized = false;
#elif DEEPEP_LL_VARIANT_MODE == 2
using LlVariantInput = half;
constexpr bool kWireFp8 = true, kE4M3 = true, kPrequantized = false;
#elif DEEPEP_LL_VARIANT_MODE == 3
using LlVariantInput = half;
constexpr bool kWireFp8 = true, kE4M3 = false, kPrequantized = false;
#elif DEEPEP_LL_VARIANT_MODE == 4
using LlVariantInput = bfloat16_t;
constexpr bool kWireFp8 = false, kE4M3 = false, kPrequantized = false;
#elif DEEPEP_LL_VARIANT_MODE == 5
using LlVariantInput = bfloat16_t;
constexpr bool kWireFp8 = true, kE4M3 = true, kPrequantized = false;
#elif DEEPEP_LL_VARIANT_MODE == 6
using LlVariantInput = bfloat16_t;
constexpr bool kWireFp8 = true, kE4M3 = false, kPrequantized = false;
#else
#error Invalid LL dispatch variant
#endif
}

// One specialization per translation unit; unchanged algorithm and reference-only diagnostics.
// 调用链：ElasticBuffer._dispatch_low_latency → csrc/ops/low_latency.cpp
// → dispatch_server_dedup_kernel_do_reference7 → 本设备入口 → DispatchMemoryServerDedupKernel。
// profiler匹配本入口任务名；下层计数/发送/接收是同一kernel内的阶段，不是独立task。
// Host selects one pure-AIV entry; the configured S/R split is unchanged.
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void LL_VARIANT_KERNEL(
    GM_ADDR commArgs, GM_ADDR x, GM_ADDR expertIds, GM_ADDR expertScales, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,
    GM_ADDR expertTokenNumsOut, GM_ADDR sendCountsOut, GM_ADDR tokenTypeOut, GM_ADDR destinationIndexOut,
    GM_ADDR relayReadIndexOut, GM_ADDR sourceMaskOut, GM_ADDR expertScalesOut, GM_ADDR tilingData,
    int64_t reservedMagic, GM_ADDR inputScales, GM_ADDR referenceProfile)
{
    const uint64_t referenceEntry = LlReferenceProfile::DeviceProfile::EntryClock(referenceProfile);
    util_set_ffts_config(reinterpret_cast<const __gm__ DispatchDedup::CommArgs*>(commArgs)->fftsVal);
    LlReferenceProfile::DeviceProfile profile;
    profile.Init(referenceProfile);
    profile.Value<0>(referenceEntry);
    profile.Mark<1>();


    // Host owns validation; each core reads its configuration once before work.
    const auto* tiling = reinterpret_cast<const __gm__ DispatchDedup::EpServerDedupTilingData*>(tilingData);
    const uint32_t bs = tiling->bs;
    const uint32_t h = tiling->h;
    const uint32_t maxTokens = tiling->numMaxTokensPerRank;
    const bool enableDedup = DispatchDedup::EpServerDedupEnabled(maxTokens);
    const uint32_t topK = tiling->topK;
    const uint32_t moeExpertNum = tiling->moeExpertNum;
    const uint32_t expertTokenNumsType = tiling->expertTokenNumsType;
    const uint8_t quantMode = tiling->quantMode;
    const uint8_t dtype = tiling->dtype;
    const uint8_t expandXOutDtype = tiling->expandXOutDtype;
    const auto inputMode = static_cast<DispatchDedup::EpServerDedupInputMode>(tiling->inputMode);
    if (DispatchDedup::LlDispatchVariantForTiling(dtype,quantMode,expandXOutDtype,
            static_cast<uint8_t>(inputMode)) != DEEPEP_LL_VARIANT_MODE) {
        AscendC::Trap(); return;
    }
#if DEEPEP_LL_DISPATCH_CB59 == 4
    // Temporary small-only experiment: reject excluded modes instead of
    // interpreting a large-call tiling through the compiled direct path.
    if (bs == 0U || bs > 32U || maxTokens < bs || maxTokens > 32U) {
        AscendC::Trap(); return;
    }
#endif

    AscendC::TPipe pipe;

    {
#if DEEPEP_LL_DISPATCH_CB59 == 4
        namespace Algorithm = DispatchDedup::Cb59Small;
#elif DEEPEP_LL_DISPATCH_CB59 == 3
        namespace Algorithm = DispatchDedup::Cb59Pair;
#elif DEEPEP_LL_DISPATCH_CB59 == 2
        namespace Algorithm = DispatchDedup::Cb59Stream;
#elif DEEPEP_LL_DISPATCH_CB59
        namespace Algorithm = DispatchDedup::Cb59Replay;
#else
        namespace Algorithm = DispatchDedup;
#endif
        Algorithm::DispatchMemoryServerDedupKernel<LlVariantInput,kWireFp8,kE4M3,kPrequantized> kernel;

        kernel.SetProfile(&profile);
        kernel.Init(commArgs,x,expertIds,expertScales,expandXOut,dynamicScalesOut,expertTokenNumsOut,sendCountsOut,
                    tokenTypeOut,destinationIndexOut,relayReadIndexOut,sourceMaskOut,expertScalesOut,bs,h,topK,
                    moeExpertNum,expertTokenNumsType,&pipe,inputScales,enableDedup);

        kernel.Process();
        kernel.ExportProfile();

    }
    profile.Finish(uint32_t(reinterpret_cast<const __gm__ DispatchDedup::CommArgs*>(commArgs)->rank),
                   uint32_t(reinterpret_cast<const __gm__ DispatchDedup::CommArgs*>(commArgs)->rankSize), bs);
}

extern "C" DEEPEP_EXPORT void LL_VARIANT_LAUNCH(void* stream, const deepep::EpServerDedupKernelArgs* args)
{
    const auto& a = *args;
    LL_VARIANT_KERNEL<<<a.coreNum, DEEPEP_KERNEL_LAUNCH_UB(0), stream>>>(
        a.commArgs, a.x, a.expertIds, a.expertScales, a.expandXOut, a.dynamicScalesOut, a.expertTokenNumsOut,
        a.sendCountsOut, a.tokenTypeOut, a.destinationIndexOut, a.relayReadIndexOut, a.sourceMaskOut, a.expertScalesOut,
        a.tilingData, a.reservedMagic, a.inputScales, a.referenceProfile);
}

#endif
