#pragma once
#include "ll_core_config.h"
#include "ep_memory_server_dedup_protocol.h"
namespace DispatchDedup {
#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
#define LL_GROUP_SEND_INLINE __aicore__ inline
#else
#define LL_GROUP_SEND_INLINE inline
#endif
struct LlGroupCountRange { uint32_t begin=0, end=0, record=0; bool local=false; };
struct LlGroupCountPlan {
    LlGroupCountRange ranges[3];
    uint32_t size=0, count=0;
    bool local=false;
};
// One owner per expert preserves both the total count and stable original q.
LL_GROUP_SEND_INLINE LlGroupCountPlan LlMakeGroupCountPlan(uint32_t experts, uint32_t ranks,
                                                          uint32_t rank, uint32_t core) {
    LlGroupCountPlan plan;
    if (core<kLlSendCoreNum) return plan;
    const uint32_t groupExperts=experts/ranks*kLlRanksPerGroup;
    const uint32_t groupBegin=rank/kLlRanksPerGroup*groupExperts;
    const uint32_t workers=kLlRecvCoreNum, worker=core-kLlSendCoreNum;
    // Keep the existing balanced local descriptors. Distribute remote expert
    // ownership over the same R workers independently; S only prepares scratch.
    const uint32_t localBegin=worker*(groupExperts/workers)+(worker<groupExperts%workers ? worker:groupExperts%workers);
    const uint32_t localCount=groupExperts/workers+(worker<groupExperts%workers ? 1U:0U);
    if (localCount) {
        plan.ranges[plan.size++]={groupBegin+localBegin,groupBegin+localBegin+localCount,0U,true};
        plan.local=true; plan.count=localCount;
    }
    const uint32_t total=experts-groupExperts;
    const uint32_t base=total/workers, remainder=total%workers;
    const uint32_t begin=worker*base+(worker<remainder ? worker : remainder);
    const uint32_t remoteCount=base+(worker<remainder ? 1U : 0U);
    const uint32_t end=begin+remoteCount;
    if (remoteCount) {
        const uint32_t frontEnd=end<groupBegin ? end : groupBegin;
        if (begin<frontEnd) plan.ranges[plan.size++]={begin,frontEnd,localCount,false};
        const uint32_t backBegin=begin>groupBegin ? begin : groupBegin;
        if (backBegin<end) plan.ranges[plan.size++]={backBegin+groupExperts,end+groupExperts,localCount+backBegin-begin,false};
    }
    plan.count+=remoteCount;
    return plan;
}
LL_GROUP_SEND_INLINE uint32_t LlGroupCountMaxRecords(uint32_t experts, uint32_t ranks, bool local) {
    if (!local) return 0U; // S owns no source count records.
    const uint32_t groupExperts=experts/ranks*kLlRanksPerGroup;
    const uint32_t workers=kLlRecvCoreNum;
    return (groupExperts+workers-1U)/workers+(experts-groupExperts+workers-1U)/workers;
}
#undef LL_GROUP_SEND_INLINE
} // namespace DispatchDedup
