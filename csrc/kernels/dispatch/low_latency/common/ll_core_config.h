#ifndef DEEPEP_LL_CORE_CONFIG_H
#define DEEPEP_LL_CORE_CONFIG_H
#include <cstdint>
#include "ll_metadata_config.h"
namespace DispatchDedup {
#ifdef DEEPEP_LL_COUNT_HISTOGRAM
constexpr uint32_t kLlCoreProtocolVersion = 41001U + 2U * kLlMetadataCores; // same wire, distinct histogram build identity
#else
constexpr uint32_t kLlCoreProtocolVersion = 41000U + 2U * kLlMetadataCores; // local and remote count targets round-robin across metadata cores
#endif
// Capacity is not participation. Pure AIV launches use the queried even count.
constexpr uint32_t kLlMaxCoreNum = 72U;
constexpr uint32_t kLlMaxRecvCoreNum = kLlMaxCoreNum / 2U;
constexpr uint32_t kLlDefaultPlanCoreNum = 64U; // CPU-only legacy helper default
constexpr uint32_t kLlNotifyCoreNum = 8U;
#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
__aicore__ inline
#else
inline
#endif
constexpr bool LlCoreCountValid(uint32_t cores) {
    return cores >= 16U && cores <= kLlMaxCoreNum && cores % 2U == 0U;
}
}

#endif
