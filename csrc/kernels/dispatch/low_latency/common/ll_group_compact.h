#pragma once
#include <stdint.h>
namespace DispatchDedup {
#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
#define LL_GROUP_COMPACT_INLINE __aicore__ inline
#else
#define LL_GROUP_COMPACT_INLINE inline
#endif
struct LlSharedGroupPlan {
    uint32_t descriptorBytes, headerBytes, tableBytes, readyBytes, cacheBytes, ubBytes, gmBytes;
};
LL_GROUP_COMPACT_INLINE uint32_t LlPlanAlign(uint32_t n) { return (n+31U)/32U*32U; }
LL_GROUP_COMPACT_INLINE LlSharedGroupPlan LlMakeSharedGroupPlan(uint32_t bs,uint32_t k,
                                                               uint32_t groups,uint32_t send) {
    const uint32_t descriptors=LlPlanAlign(((bs+send-1U)/send)*k*8U);
    const uint32_t header=LlPlanAlign((2U*groups+1U)*4U);
    const uint32_t table=header*send, ready=send*32U;
    const uint32_t cache=((bs*k+send-1U)/send)*32U;
    return {descriptors,header,table,ready,cache,
            2U*descriptors+table+ready+header+cache,table+descriptors*send};
}
struct LlOwnedGroupSlice { uint32_t first,count,nextOwner; };
// Select only this core's descriptors in a producer's contiguous bucket.
// Carry owner across producer buckets, destination groups and phases.
LL_GROUP_COMPACT_INLINE LlOwnedGroupSlice LlOwnedGroupTasks(uint32_t n,uint32_t owner,
                                                          uint32_t core,uint32_t send) {
    const uint32_t first=core>=owner ? core-owner : core+send-owner;
    return {first,first<n ? 1U+(n-1U-first)/send : 0U,(owner+n%send)%send};
}
// Word8 in each persistent 512B per-core magic slot; word0 is unchanged.
constexpr uint32_t kLlSharedPlanReadyOffset=32U;
constexpr uint64_t kLlSharedPlanControlLimit=1024ULL*1024ULL;
#undef LL_GROUP_COMPACT_INLINE
}
