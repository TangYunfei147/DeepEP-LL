#pragma once
#include "ll_core_config.h"
#include "ep_memory_server_dedup_protocol.h"
namespace DispatchDedup {
#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
#define LL_GROUP_SEND_INLINE __aicore__ inline
#else
#define LL_GROUP_SEND_INLINE inline
#endif
struct LlGroupCountRange { uint32_t begin=0, end=0, record=0; };
struct LlGroupCountPlan {
    LlGroupCountRange ranges[2];
    uint32_t size=0, count=0;
    bool local=false;
};
// One owner per expert preserves both the total count and stable original q.
LL_GROUP_SEND_INLINE LlGroupCountPlan LlMakeGroupCountPlan(uint32_t experts, uint32_t ranks,
                                                          uint32_t rank, uint32_t core) {
    LlGroupCountPlan plan;
    const uint32_t groupExperts=experts/ranks*kLlRanksPerGroup;
    const uint32_t groupBegin=rank/kLlRanksPerGroup*groupExperts;
    plan.local=core>=kLlSendCoreNum;
    const uint32_t total=plan.local ? groupExperts : experts-groupExperts;
    const uint32_t workers=plan.local ? kLlRecvCoreNum : kLlSendCoreNum;
    const uint32_t worker=plan.local ? core-kLlSendCoreNum : core;
    const uint32_t base=total/workers, remainder=total%workers;
    const uint32_t begin=worker*base+(worker<remainder ? worker : remainder);
    plan.count=base+(worker<remainder ? 1U : 0U);
    const uint32_t end=begin+plan.count;
    if (!plan.count) return plan;
    if (plan.local) {
        plan.ranges[0]={groupBegin+begin,groupBegin+end,0U}; plan.size=1U;
    } else {
        const uint32_t frontEnd=end<groupBegin ? end : groupBegin;
        if (begin<frontEnd) plan.ranges[plan.size++]={begin,frontEnd,0U};
        const uint32_t backBegin=begin>groupBegin ? begin : groupBegin;
        if (backBegin<end) plan.ranges[plan.size++]={backBegin+groupExperts,end+groupExperts,backBegin-begin};
    }
    return plan;
}
LL_GROUP_SEND_INLINE uint32_t LlGroupCountMaxRecords(uint32_t experts, uint32_t ranks, bool local) {
    const uint32_t groupExperts=experts/ranks*kLlRanksPerGroup;
    const uint32_t total=local ? groupExperts : experts-groupExperts;
    const uint32_t workers=local ? kLlRecvCoreNum : kLlSendCoreNum;
    return (total+workers-1U)/workers;
}
#undef LL_GROUP_SEND_INLINE
} // namespace DispatchDedup
