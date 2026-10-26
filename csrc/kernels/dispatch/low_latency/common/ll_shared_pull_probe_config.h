#pragma once
#include <cstdint>
namespace DispatchDedup {
// Frozen per candidate commit: 0=original, 1=lifetime fence control, 2=shared pull.
#ifndef DEEPEP_LL_SHARED_PULL_PROBE_MODE
#define DEEPEP_LL_SHARED_PULL_PROBE_MODE 2
#endif
constexpr uint32_t kLlSharedPullProbeMode = DEEPEP_LL_SHARED_PULL_PROBE_MODE;
#define DEEPEP_LL_SHARED_PULL_WIRE_REVISION 2
constexpr uint32_t kLlSharedPullWireRevision = DEEPEP_LL_SHARED_PULL_WIRE_REVISION;
static_assert(kLlSharedPullProbeMode<=2U,"invalid shared-pull prototype");
constexpr uint32_t kLlSharedPullDescriptor = 0x50554c4cU;
#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
__aicore__ inline
#else
inline
#endif
bool LlSharedPullScope(bool prequantized,bool dedup,uint32_t world) {
    return kLlSharedPullProbeMode!=0U && prequantized && dedup && world==16U;
}
#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
__aicore__ inline
#else
inline
#endif
bool LlSharedPullAddressValid(uint32_t owner,uint32_t reader,uint32_t world,uint32_t group,
    uint64_t offset,uint64_t base,uint64_t bytes,uint64_t stride) {
    return world==16U && group!=0U && owner<world && reader<world &&
        owner/group==reader/group && stride!=0U && offset>=base &&
        bytes>=stride && offset-base<=bytes-stride && (offset-base)%stride==0U;
}
}
