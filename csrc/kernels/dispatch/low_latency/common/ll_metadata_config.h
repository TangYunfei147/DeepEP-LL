#pragma once
#include <cstdint>
#ifndef DEEPEP_LL_METADATA_CORES
#define DEEPEP_LL_METADATA_CORES 4
#endif
#if DEEPEP_LL_METADATA_CORES < 2 || DEEPEP_LL_METADATA_CORES > 70 || (DEEPEP_LL_METADATA_CORES % 2)
#error DEEPEP_LL_METADATA_CORES must be an even integer in [2,70]
#endif
namespace DispatchDedup {
constexpr uint32_t kLlMetadataCores = DEEPEP_LL_METADATA_CORES;
#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
#define LL_META_INLINE __aicore__ inline
#else
#define LL_META_INLINE inline
#endif
LL_META_INLINE constexpr bool MetadataSplitValid(uint32_t cores) {
    return cores >= kLlMetadataCores + 2U && cores <= 72U && cores % 2U == 0U;
}
LL_META_INLINE constexpr uint32_t PayloadCores(uint32_t cores) { return (cores-kLlMetadataCores)/2U; }
LL_META_INLINE constexpr uint32_t ReceiveBegin(uint32_t cores) { return kLlMetadataCores+PayloadCores(cores); }
LL_META_INLINE constexpr uint32_t ForwardSlots(uint32_t slots,uint32_t core,uint32_t cores) {
    return slots > core ? (slots-1U-core)/cores+1U : 0U;
}
LL_META_INLINE constexpr uint32_t BalancedBegin(uint32_t total,uint32_t core,uint32_t cores) {
    return uint32_t(uint64_t(total)*core/cores);
}
#undef LL_META_INLINE
}
