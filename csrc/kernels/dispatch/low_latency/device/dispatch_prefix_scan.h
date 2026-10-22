#pragma once
#include <cstdint>
#include "c_api/asc_simd.h"

namespace DispatchDedup {
template<uint32_t Distance>
__simd_callee__ inline void PrefixScanStep(vector_uint32_t& values,
    vector_uint32_t indices, vector_uint32_t zero, vector_bool all)
{
    vector_uint32_t delta, shiftedIndex, safeIndex, shifted, selected, sum;
    asc_duplicate_scalar(delta,Distance);
    asc_sub(shiftedIndex,indices,delta,all);
    const vector_bool valid=asc_ge_scalar(indices,Distance,all);
    // Even masked-off lanes must have legal gather indices.
    asc_select(safeIndex,shiftedIndex,zero,valid);
    asc_gather(shifted,values,safeIndex);
    asc_select(selected,shifted,zero,valid);
    asc_add(sum,values,selected,all);
    values=sum;
}

// Inclusive integer scan. Caller provides ceil(count/64)*256B backing storage;
// stores are masked, and padded input lanes never feed a valid prefix lane.
// In-place safe: the current block is fully loaded before its store; the next
// block is disjoint. Carry remains in a vector register, with no UB scalar loop.
__simd_vf__ inline void R0PrefixScan(__ubuf__ uint32_t* data,uint32_t count)
{
    vector_int32_t signedIndices;
    asc_arange(signedIndices,int32_t(0));
    const vector_uint32_t indices=reinterpret_cast<vector_uint32_t&>(signedIndices);
    vector_uint32_t zero,carry;
    asc_duplicate_scalar(zero,uint32_t(0));
    carry=zero;
    const vector_bool all=asc_create_mask_b32(PAT_ALL);
    uint32_t left=count;
    for (uint32_t begin=0;begin<count;begin+=64U) {
        const uint32_t valid=count-begin<64U?count-begin:64U;
        vector_uint32_t values,total,lastIndex,nextCarry;
        asc_loadalign(values,data+begin);
        PrefixScanStep<1U>(values,indices,zero,all);
        PrefixScanStep<2U>(values,indices,zero,all);
        PrefixScanStep<4U>(values,indices,zero,all);
        PrefixScanStep<8U>(values,indices,zero,all);
        PrefixScanStep<16U>(values,indices,zero,all);
        PrefixScanStep<32U>(values,indices,zero,all);
        asc_add(total,values,carry,all);
        asc_duplicate_scalar(lastIndex,valid-1U);
        asc_gather(nextCarry,total,lastIndex);
        carry=nextCarry;
        const vector_bool mask=asc_update_mask_b32(left);
        asc_storealign(data+begin,total,mask);
    }
}
}
