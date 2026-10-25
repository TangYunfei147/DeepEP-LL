#ifndef DEEPEP_LL_CORE_CONFIG_H
#define DEEPEP_LL_CORE_CONFIG_H
#include <cstdint>
#include "ll_dispatch_implementation.h"
namespace DispatchDedup {
#ifndef DEEPEP_LL_SEND_CORES
#define DEEPEP_LL_SEND_CORES 28
#endif
#ifndef DEEPEP_LL_RECV_CORES
#define DEEPEP_LL_RECV_CORES 28
#endif
static_assert(DEEPEP_LL_SEND_CORES > 0 && DEEPEP_LL_RECV_CORES > 0, "LL core counts must be positive");
static_assert(DEEPEP_LL_SEND_CORES + DEEPEP_LL_RECV_CORES == 56, "LL send + receive cores must equal 56");
static_assert(!kLlCb59Replay || (DEEPEP_LL_SEND_CORES == 28 && DEEPEP_LL_RECV_CORES == 28),
              "cb59 replay requires the original 1:1 split on the fixed 56-core comparison launch");
constexpr uint32_t kLlSendCoreNum = DEEPEP_LL_SEND_CORES;
constexpr uint32_t kLlRecvCoreNum = DEEPEP_LL_RECV_CORES;
constexpr uint32_t kLlActiveCoreNum = 56U;
constexpr uint32_t kLlDispatchUbBytes = 230U * 1024U;
// Dispatch senders join receive after their own sends; combine retains S/R.
constexpr uint32_t kLlDispatchRecvCoreNum = kLlCb59Replay ? kLlRecvCoreNum : kLlActiveCoreNum;
// Encode the split so mismatched Host/device objects and rank identities differ.
constexpr uint32_t kLlCoreProtocolVersion = (kLlCb59Small ? 24000U : (kLlCb59Pair ? 23000U :
    (kLlCb59Stream ? 22000U : (kLlCb59Replay ? 21000U : 19000U)))) + kLlSendCoreNum;
// Storage capacity remains separate from the fixed 56-core launch.
constexpr uint32_t kLlMaxCoreNum = 72U;
constexpr uint32_t kLlMaxRecvCoreNum = 56U;
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
