#pragma once
#include <cstdint>
namespace DispatchDedup {
#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
#define LL_PRIORITY_INLINE __aicore__ inline
#else
#define LL_PRIORITY_INLINE inline
#endif

// At most 16 snapshot positions, each encoded in four bits. No extra UB tile.
struct LlReceivePriority {
    uint64_t positions=0U;
    uint32_t count=0U;
    LL_PRIORITY_INLINE uint32_t At(uint32_t i) const {
        return uint32_t((positions>>(i*4U))&15U);
    }
};
// Caller supplies 0<=size<=16. Priority applies ONLY to already-ready packets.
// Keep the old descending order within each class.
LL_PRIORITY_INLINE LlReceivePriority LlMakeReceivePriority(uint32_t size,uint32_t ready,uint32_t priority) {
    LlReceivePriority order;
    ready&=(uint32_t(1)<<size)-1U;
    const uint32_t first=ready&priority;
    for (uint32_t pass=0;pass<2U;++pass) {
        const uint32_t mask=pass==0U ? first : ready&~first;
        if (!mask) continue;
        for (uint32_t n=size;n>0U;--n) {
            const uint32_t index=n-1U;
            if (mask&(uint32_t(1)<<index)) {
                order.positions|=uint64_t(index)<<(order.count*4U);
                ++order.count;
            }
        }
    }
    return order;
}
#undef LL_PRIORITY_INLINE
} // namespace DispatchDedup
