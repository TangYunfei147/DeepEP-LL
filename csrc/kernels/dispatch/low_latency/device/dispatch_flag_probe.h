#pragma once

#include <cstdint>
#include "c_api/asc_simd.h"

namespace DispatchDedup {

// Specialized for eight 32B flag blocks per packet (64 floats), at most the
// existing sixteen-packet Probe batch. The caller retains the generic Sum path
// for other block counts. All flag words participate, including the layout
// marker; ready requires exactly 64 or 64.5, as PacketReceiveLayout::Decode.
//
// Input: the same contiguous flag-DMA buffer as the generic path. Output: two
// uint32 words, readyMask and keepFlagsMask, in the existing sum buffer. Input
// rows are 256B apart; both buffers retain their original 32B alignment. No
// additional UB allocation or intermediate per-packet scalar read is needed.
// Caller must preserve MTE2_V before this VF and V_S before reading its output.
__simd_vf__ inline void BuildReceiveFlagMasks(
    __ubuf__ float* flags, __ubuf__ uint32_t* output,
    uint32_t count, uint32_t blocks)
{
    const vector_bool all = asc_create_mask_b32(PAT_ALL);
    uint32_t firstCount = 1U;
    const vector_bool first = asc_update_mask_b32(firstCount);
    vector_uint32_t zero, readyBits, keepBits;
    asc_duplicate_scalar(zero, uint32_t(0));
    readyBits = zero;
    keepBits = zero;

    // Each row contains protocol values 0, 1 or 1.5. Their sums are exactly
    // representable, independent of the reduction tree. Equality deliberately
    // rejects any other sum; do not replace it with a threshold/interval test.
    #pragma unroll 1
    for (uint32_t i = 0U; i < count; ++i) {
        vector_float values, sum;
        asc_loadalign(values, flags + i * blocks * 8U);
        asc_reduce_sum(sum, values, all);
        const vector_bool compact = asc_eq_scalar(sum, 64.0f, first);
        const vector_bool keep = asc_eq_scalar(sum, 64.5f, first);
        vector_uint32_t bit, readyContribution, keepContribution;
        asc_duplicate_scalar(bit, uint32_t(1U << i));
        asc_select(readyContribution, bit, zero, compact);
        asc_select(readyContribution, bit, readyContribution, keep);
        asc_select(keepContribution, bit, zero, keep);
        // Every iteration contributes a distinct bit, so addition equals OR.
        asc_add(readyBits, readyBits, readyContribution, first);
        asc_add(keepBits, keepBits, keepContribution, first);
    }

    vector_uint32_t readyAll, keepAll, packed;
    asc_gather(readyAll, readyBits, zero);
    asc_gather(keepAll, keepBits, zero);
    asc_select(packed, readyAll, keepAll, first);
    uint32_t outputCount = 2U;
    const vector_bool outputMask = asc_update_mask_b32(outputCount);
    asc_storealign(output, packed, outputMask);
}

}  // namespace DispatchDedup
