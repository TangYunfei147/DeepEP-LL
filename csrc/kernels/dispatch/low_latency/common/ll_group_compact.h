#pragma once
#include <stdint.h>

namespace DispatchDedup {
#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
#define LL_GROUP_COMPACT_INLINE __aicore__ inline
#else
#define LL_GROUP_COMPACT_INLINE inline
#endif

// Access exposes Index(i), the stable, ascending original route index selected
// for ONE destination group. Emit receives only packets owned by this core.
// All send cores visit identical groups/phases and advance owner identically.
// Carry owner across groups AND phases: restarting at zero would concentrate
// every short group on the same low-numbered cores.
template<class Access, class Emit>
LL_GROUP_COMPACT_INLINE void LlVisitCompactGroup(Access access, uint32_t size, uint32_t topK,
    bool dedup, bool firstHopPhase, uint32_t sendCores, uint32_t core,
    uint32_t& owner, Emit emit)
{
    for (uint32_t pos=0; pos<size;) {
        const uint32_t first=access.Index(pos++);
        uint32_t mask=0U;
        if (dedup) {
            const uint32_t token=first/topK;
            mask=uint32_t(1)<< (first%topK);
            while (pos<size && access.Index(pos)/topK==token) {
                mask |= uint32_t(1) << (access.Index(pos)%topK);
                ++pos;
            }
            const bool multiple=(mask & (mask-1U))!=0U;
            if (multiple!=firstHopPhase) continue;
            if (!multiple) mask=0U;
        }
        if (owner==core) emit(first,mask);
        if (++owner==sendCores) owner=0U;
    }
}
#undef LL_GROUP_COMPACT_INLINE
} // namespace DispatchDedup
