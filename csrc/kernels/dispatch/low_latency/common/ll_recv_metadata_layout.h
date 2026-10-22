#pragma once

#include "ll_overlap_layout.h"

namespace DispatchDedup {
#if DISPATCH_DEDUP_ASCENDC_AICORE_COMPILE
#define LL_RECV_METADATA_INLINE FORCE_INLINE_AICORE
#else
#define LL_RECV_METADATA_INLINE inline
#endif

struct LlRecvMetadataBatchLayout {
    uint32_t rows = 0U;
    uint32_t destinationOffset = 0U;
    uint32_t bytes = 0U;
    bool enabled = false;
};

// Allocate only from otherwise unused budget; never recalculate/reduce the
// production send/receive lane count to make this candidate fit. totalBytes
// already includes worst-case task tables, retained TX/count UB and RX lanes.
// Actual rows determine staging size; the conservative old total remains the
// bound for every other allocation. No buffer or event lifetime is shortened.
LL_RECV_METADATA_INLINE LlRecvMetadataBatchLayout LlMakeRecvMetadataBatchLayout(
    uint32_t rows, uint32_t bs, bool enableDedup, const LlOverlapLayout& overlap)
{
    LlRecvMetadataBatchLayout result{};
    if (!enableDedup || bs <= 32U || !overlap.enabled || rows == 0U) return result;
    const uint64_t typeBytes = DispatchDedupAlign(uint64_t(rows) * 4U, 32U);
    const uint64_t destinationBytes = DispatchDedupAlign(uint64_t(rows) * 12U, 32U);
    const uint64_t bytes = typeBytes + destinationBytes;
    if (overlap.totalBytes > kLlDispatchUbBytes ||
        bytes > uint64_t(kLlDispatchUbBytes) - overlap.totalBytes) return result;
    result.rows = rows;
    result.destinationOffset = uint32_t(typeBytes);
    result.bytes = uint32_t(bytes);
    result.enabled = true;
    return result;
}

#undef LL_RECV_METADATA_INLINE
}  // namespace DispatchDedup
