#pragma once
#include "ll_rdma_layout.h"
#include "ll_route_exchange_layout.h"

namespace DispatchDedup {
#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
__aicore__ inline
#else
inline
#endif
uint32_t LlCountProducerCores(uint32_t cores, uint32_t bs, uint32_t world, uint64_t routes, bool rdma) {
    (void)world; (void)routes; (void)rdma;
    return bs < kEpServerDedupAllCoreCountBsThreshold ? cores : cores / 2U;
}
}
