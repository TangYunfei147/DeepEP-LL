#ifndef DEEPEP_LL_CORE_CONFIG_H
#define DEEPEP_LL_CORE_CONFIG_H
#include <cstdint>
#include "ll_shared_pull_probe_config.h"
#define DEEPEP_LL_PREFIX_READY_REVISION 1
namespace DispatchDedup {
constexpr uint32_t kLlPrefixReadyRevision = DEEPEP_LL_PREFIX_READY_REVISION;
#ifndef DEEPEP_LL_SEND_CORES
#define DEEPEP_LL_SEND_CORES 28
#endif
#ifndef DEEPEP_LL_RECV_CORES
#define DEEPEP_LL_RECV_CORES 28
#endif
static_assert(DEEPEP_LL_SEND_CORES > 0 && DEEPEP_LL_RECV_CORES > 0, "LL core counts must be positive");
static_assert(DEEPEP_LL_SEND_CORES + DEEPEP_LL_RECV_CORES == 56, "LL send + receive cores must equal 56");
constexpr uint32_t kLlSendCoreNum = DEEPEP_LL_SEND_CORES;
constexpr uint32_t kLlRecvCoreNum = DEEPEP_LL_RECV_CORES;
constexpr uint32_t kLlActiveCoreNum = 56U;
constexpr uint32_t kLlDispatchUbBytes = 230U * 1024U;
// Dispatch senders join receive after their own sends; combine retains S/R.
constexpr uint32_t kLlDispatchRecvCoreNum = kLlActiveCoreNum;
// Encode the split so mismatched Host/device objects and rank identities differ.
constexpr uint32_t kLlCoreProtocolVersion =
    (kLlSharedPullProbeMode==0U ? 25000U : (kLlSharedPullProbeMode==1U ? 26000U :
        (kLlSharedPullWireRevision==1U ? 27000U : 28000U))) + kLlSendCoreNum;
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
