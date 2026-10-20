#pragma once
#include <cstdint>
#include "ll_core_config.h"
#include "ep_memory_server_dedup_protocol.h"
namespace DispatchDedup {
#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
#define LL_PLAN_INLINE __aicore__ inline
#else
#define LL_PLAN_INLINE inline
#endif
struct LlRoutePlanLayout {
    uint64_t matrixBytes=0, totalsOffset=0, idsOffset=0, qOffset=0, masksOffset=0, bytes=0;
};
LL_PLAN_INLINE uint64_t LlPlanAlign(uint64_t n,uint64_t a=512U) { return (n+a-1U)/a*a; }
LL_PLAN_INLINE LlRoutePlanLayout LlRoutePlanMemory(uint64_t routes,uint32_t experts) {
    LlRoutePlanLayout p;
    p.matrixBytes=uint64_t(experts)*kLlActiveCoreNum*32U;
    p.totalsOffset=p.matrixBytes;
    p.idsOffset=LlPlanAlign(p.totalsOffset+uint64_t(experts)*32U);
    p.qOffset=p.idsOffset+LlPlanAlign(routes*4U);
    p.masksOffset=p.qOffset+LlPlanAlign(routes*4U);
    p.bytes=p.masksOffset+LlPlanAlign(routes*4U);
    return p;
}
// Remote combine returns can arrive before this rank finishes dispatch.
// Keep route metadata AFTER the entire fixed return region, never inside it.
LL_PLAN_INLINE uint64_t LlRoutePlanOffset(uint32_t bs,uint32_t h,uint32_t k,uint64_t sourceOffset) {
    const uint64_t stride=(uint64_t(h)*2U+479U)/480U*512U;
    return LlPlanAlign(sourceOffset+uint64_t(bs)*k*stride);
}
LL_PLAN_INLINE uint32_t LlPlanFirst(uint32_t mask) {
    uint32_t k=0U; while (!(mask&(uint32_t(1)<<k))) ++k; return k;
}
template<class Ids>
LL_PLAN_INLINE uint32_t LlChooseLeader(Ids ids,uint32_t k,uint32_t mask,uint32_t localExperts) {
    uint32_t counts[kLlRanksPerGroup]{};
    uint32_t first[kLlRanksPerGroup]{};
    uint32_t best=ids.GetValue(LlPlanFirst(mask))/localExperts%kLlRanksPerGroup;
    for (uint32_t j=0;j<k;++j) {
        if (!(mask&(uint32_t(1)<<j))) continue;
        const uint32_t rank=ids.GetValue(j)/localExperts%kLlRanksPerGroup;
        if (!counts[rank]) first[rank]=j;
        ++counts[rank];
        if (counts[rank]>counts[best] ||
            (counts[rank]==counts[best] && first[rank]<first[best])) best=rank;
    }
    return first[best];
}
LL_PLAN_INLINE uint64_t LlRoutePlanUbBytes(uint32_t bs,uint32_t k,uint32_t experts,uint32_t groups) {
    const uint64_t shard=(uint64_t(bs)+kLlActiveCoreNum-1U)/kLlActiveCoreNum*k;
    const uint64_t owned=(experts+kLlActiveCoreNum-1U)/kLlActiveCoreNum;
    return uint64_t(experts)*32U+3U*LlPlanAlign(shard*4U ? shard*4U : 1U,256U)+
           LlPlanAlign(uint64_t(groups)*8U,32U)+LlPlanAlign(owned*kLlActiveCoreNum*32U,32U)+
           LlPlanAlign(owned*32U,32U);
}
#undef LL_PLAN_INLINE
} // namespace DispatchDedup
