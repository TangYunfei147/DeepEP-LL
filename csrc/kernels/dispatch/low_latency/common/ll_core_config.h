#ifndef DEEPEP_LL_CORE_CONFIG_H
#define DEEPEP_LL_CORE_CONFIG_H
#include <cstdint>
namespace DispatchDedup {
#ifndef DEEPEP_LL_SEND_CORES
#define DEEPEP_LL_SEND_CORES 29
#endif
#ifndef DEEPEP_LL_RECV_CORES
#define DEEPEP_LL_RECV_CORES 29
#endif
static_assert(DEEPEP_LL_SEND_CORES > 0 && DEEPEP_LL_RECV_CORES > 0, "LL core counts must be positive");
static_assert(DEEPEP_LL_SEND_CORES + DEEPEP_LL_RECV_CORES == 58, "LL send + receive cores must equal 58");
constexpr uint32_t kLlSendCoreNum = DEEPEP_LL_SEND_CORES;
constexpr uint32_t kLlRecvCoreNum = DEEPEP_LL_RECV_CORES;
constexpr uint32_t kLlActiveCoreNum = 58U;
// Encode the split so mismatched Host/device objects and rank identities differ.
constexpr uint32_t kLlCoreProtocolVersion = 6000U + kLlSendCoreNum;
// Storage capacity remains separate from the fixed 58-core launch.
constexpr uint32_t kLlMaxCoreNum = 72U;
constexpr uint32_t kLlMaxRecvCoreNum = 58U;
constexpr uint32_t kLlDefaultPlanCoreNum = kLlActiveCoreNum; // Host/device planning uses the compiled launch
constexpr uint32_t kLlNotifyCoreNum = 8U;
#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
__aicore__ inline
#else
inline
#endif
constexpr bool LlCoreCountValid(uint32_t cores) {
    return cores == kLlActiveCoreNum;
}
}

#endif
