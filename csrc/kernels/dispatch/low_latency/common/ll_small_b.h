#pragma once
#include "ll_local_q.h"
namespace DispatchDedup {
#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
#define LL_SMALL_INLINE __aicore__ inline
#else
#define LL_SMALL_INLINE inline
#endif
LL_SMALL_INLINE bool LlUseSmallDirect(uint32_t bs,bool dedup) {
    return !dedup && bs<=32U;
}
// Only used for B2/4. A tiny route prefix costs less setup than a dense expert
// histogram; large inputs and every dedup invocation keep the original VF path.
template<class Routes,class Table>
LL_SMALL_INLINE void LlBuildTinyDirectQ(Routes routes,Table table,uint32_t topK,LlQShard shard) {
    for(uint32_t j=0;j<shard.count;++j) {
        const uint32_t index=shard.begin+j;
        uint32_t q=0;
        if(index>=topK) {
            const uint32_t expert=uint32_t(routes.GetValue(index));
            for(uint32_t i=0;i<index;++i)q+=uint32_t(routes.GetValue(i))==expert;
        }
        table.SetValue(j,q);
    }
}
#undef LL_SMALL_INLINE
}
