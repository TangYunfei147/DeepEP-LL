#pragma once
#include "ll_core_config.h"
#include "ep_memory_server_dedup_protocol.h"
namespace DispatchDedup {
#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
#define LL_GROUP_SEND_INLINE __aicore__ inline
#else
#define LL_GROUP_SEND_INLINE inline
#endif
struct LlGroupCountRange { uint32_t begin=0,end=0,record=0; bool local=false; };
struct LlGroupCountPlan {
    LlGroupCountRange ranges[3];
    uint32_t size=0,count=0;
    bool local=false;
};
// Only R cores publish counts. Keep their old local-expert ownership, and
// distribute the complementary expert ranges over the same R workers.
LL_GROUP_SEND_INLINE LlGroupCountPlan LlMakeGroupCountPlan(uint32_t experts,uint32_t ranks,
                                                          uint32_t rank,uint32_t core) {
    LlGroupCountPlan p{};
    if (core<kLlSendCoreNum) return p;
    const uint32_t worker=core-kLlSendCoreNum,workers=kLlRecvCoreNum;
    const uint32_t local=experts/ranks*kLlRanksPerGroup,first=rank/kLlRanksPerGroup*local;
    const uint32_t remote=experts-local;
    const uint32_t begin=worker*(remote/workers)+(worker<remote%workers?worker:remote%workers);
    const uint32_t end=begin+remote/workers+(worker<remote%workers);
    const uint32_t front=end<first?end:first;
    if (begin<front) p.ranges[p.size++]={begin,front,0U,false};
    const uint32_t back=begin>first?begin:first;
    if (back<end) p.ranges[p.size++]={back+local,end+local,back-begin,false};
    p.count=end-begin;
    const uint32_t lo=worker*(local/workers)+(worker<local%workers?worker:local%workers);
    const uint32_t hi=lo+local/workers+(worker<local%workers);
    if (lo<hi) {
        p.ranges[p.size++]={first+lo,first+hi,p.count,true};
        p.count+=hi-lo;p.local=true;
    }
    return p;
}
// local=false is the S budget (zero records); true bounds every R producer.
LL_GROUP_SEND_INLINE uint32_t LlGroupCountMaxRecords(uint32_t experts,uint32_t ranks,bool local) {
    if (!local) return 0U;
    const uint32_t n=experts/ranks*kLlRanksPerGroup;
    return (n+kLlRecvCoreNum-1U)/kLlRecvCoreNum+
           (experts-n+kLlRecvCoreNum-1U)/kLlRecvCoreNum;
}
#undef LL_GROUP_SEND_INLINE
}
