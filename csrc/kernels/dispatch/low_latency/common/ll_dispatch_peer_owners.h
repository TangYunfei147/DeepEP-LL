#pragma once
#include "ll_rdma_layout.h"
#include "ll_route_exchange_layout.h"

namespace DispatchDedup {
#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
#define LL_PEER_OWNER_INLINE __aicore__ inline
#else
#define LL_PEER_OWNER_INLINE inline
#endif

// Exactly eight distinct owners PER PEER, drawn from the whole S-core pool.
// Preserve the control producer's QP owner so independently progressing count
// and payload phases cannot concurrently operate the same peer/QP.
LL_PEER_OWNER_INLINE uint32_t LlDispatchQpOwner(uint32_t rank, uint32_t world, uint32_t sendCores,
                                                uint32_t peer, uint32_t qp, uint32_t routes)
{
    const bool paired = world == 32U && routes <= kRouteExchangeMaxBytes / sizeof(int32_t);
    const uint32_t relative = (peer + world - rank) % world;
    const bool anchored = !paired || peer == (rank ^ 16U);
    if (!anchored) return (relative * LlRdma::kQpCount + qp) % sendCores;

    // Fallback count partitions are whole-rank ranges: E is divisible by EP,
    // and EP is divisible by eight. Its peer's count producer is peer/(EP/8).
    const uint32_t anchor = paired ? 0U : peer / (world / LlRdma::kQpCount);
    if (qp == anchor) return anchor;
    const uint32_t slot = qp < anchor ? qp : qp - 1U;
    const uint32_t compressed = (relative * (LlRdma::kQpCount - 1U) + slot) % (sendCores - 1U);
    return compressed < anchor ? compressed : compressed + 1U;
}
#undef LL_PEER_OWNER_INLINE
}
