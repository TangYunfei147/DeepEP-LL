#pragma once
#include <cstdint>

namespace DispatchDedup {
#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
#define LL_MESH_INLINE __aicore__ inline
#else
#define LL_MESH_INLINE inline
#endif

// Confirmed EP32 deployment: each physical host contains four consecutive
// full-mesh groups of four ranks. Count actual expert deliveries, not distinct
// cards: each relay carries one equal-sized token for its target expert.
// Two scalar words suffice for K<=32; no UB array, route rescan or cross-core work.
class LlFullMeshFirstHop {
    uint32_t counts_ = 0U;
    uint32_t first_ = 0U;
public:
    LL_MESH_INLINE void Add(uint32_t destinationRank, uint32_t k) {
        const uint32_t shift = ((destinationRank & 15U) >> 2U) * 8U;
        counts_ += 1U << shift;
        if (((first_ >> shift) & 255U) == 0U) first_ |= (k + 1U) << shift;
    }
    LL_MESH_INLINE uint32_t FirstK() const {
        uint32_t bestCount = 0U, bestK = 32U;
        for (uint32_t group = 0U; group < 4U; ++group) {
            const uint32_t shift = group * 8U;
            const uint32_t count = (counts_ >> shift) & 255U;
            if (!count) continue;
            const uint32_t k = ((first_ >> shift) & 255U) - 1U;
            if (count > bestCount || (count == bestCount && k < bestK)) {
                bestCount = count;
                bestK = k;
            }
        }
        return bestCount ? bestK : 0U;
    }
};

LL_MESH_INLINE uint32_t LlRelayMemberMask(uint32_t mask, uint32_t firstK) {
    return mask & ~(1U << firstK);
}
#undef LL_MESH_INLINE
}
