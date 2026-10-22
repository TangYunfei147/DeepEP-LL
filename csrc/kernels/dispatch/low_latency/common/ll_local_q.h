#pragma once
#include <cstdint>
namespace DispatchDedup {
#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
#define LL_Q_INLINE __aicore__ inline
#else
#define LL_Q_INLINE inline
#endif
struct LlQShard { uint32_t begin,count; };
LL_Q_INLINE LlQShard LlLocalQShard(uint32_t bs,uint32_t k,uint32_t cores,uint32_t core,bool dedup) {
    const uint32_t total=dedup?bs:bs*k,base=total/cores,extra=total%cores;
    const uint32_t begin=core*base+(core<extra?core:extra),count=base+(core<extra);
    return {dedup?begin*k:begin,dedup?count*k:count};
}
// All operands are UB views on device. Histogram is private to this core.
// Preserve the legacy first-token q=0 even for repeated expert IDs.
template<class Routes,class Histogram,class Table>
LL_Q_INLINE void LlBuildLocalQ(Routes routes,Histogram histogram,Table table,
                              uint32_t experts,uint32_t topK,LlQShard shard) {
    if (!shard.count) return;
    for(uint32_t e=0;e<experts;++e) histogram.SetValue(e,0U);
    for(uint32_t i=0;i<shard.begin;++i) {
        const uint32_t e=uint32_t(routes.GetValue(i));
        histogram.SetValue(e,histogram.GetValue(e)+1U);
    }
    for(uint32_t j=0;j<shard.count;++j) {
        const uint32_t i=shard.begin+j,e=uint32_t(routes.GetValue(i));
        const uint32_t q=histogram.GetValue(e);
        table.SetValue(j,i<topK?0U:q);
        histogram.SetValue(e,q+1U);
    }
}
#undef LL_Q_INLINE
}
