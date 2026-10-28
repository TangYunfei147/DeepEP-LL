#pragma once
#include "ll_rdma_layout.h"
#include "ll_route_exchange_layout.h"

namespace DispatchDedup {
#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
#define LL_PEER_OWNER_INLINE __aicore__ inline
#else
#define LL_PEER_OWNER_INLINE inline
#endif

// All sender cores may issue to each peer; each owns its physical-core QP.
// Preserve the control producer's QP owner so independently progressing count
// and payload phases cannot concurrently operate the same peer/QP.
LL_PEER_OWNER_INLINE uint32_t LlDispatchQpOwner(uint32_t rank, uint32_t world, uint32_t sendCores,
                                                uint32_t peer, uint32_t qp, uint32_t routes)
{
    (void)rank; (void)world; (void)sendCores; (void)peer; (void)routes;
    return qp; // QP identity equals its unique physical sender core.

}
#undef LL_PEER_OWNER_INLINE
}
