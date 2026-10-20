#pragma once

#include <cstdint>
#include "c_api/asc_simd.h"

namespace DispatchDedup {
// Bound the *whole segment*, not each vector instruction. Even one hot bin
// stays <=32768. Keep a multiple of 64 for aligned int32 vector loads.
constexpr uint32_t kLlHistogramSegment = 32768U;
constexpr uint32_t kLlHistogramBins = 128U;

__simd_callee__ inline void AccumulateFrequency(
    vector_uint16_t& histogram, __ubuf__ uint32_t* input,
    uint32_t count, uint32_t firstExpert, uint32_t bins)
{
    vector_uint32_t base;
    asc_duplicate_scalar(base, firstExpert);
    uint32_t remaining=count;
    for (uint32_t i=0;i<count;i+=64U) {
        vector_uint32_t ids, relative;
        asc_loadalign(ids,input+i);
        vector_bool valid=asc_update_mask_b32(remaining);
        asc_sub(relative,ids,base,valid);
        // Unsigned subtraction also excludes IDs below this expert window.
        valid=asc_lt_scalar(relative,bins,valid);
        vector_uint16_t packed16;
        vector_uint8_t packed8;
        asc_pack_to_low(packed16,relative);
        asc_pack_to_low(packed8,packed16);
        // b32 predicates use one bit per four bytes; pack twice with data.
        valid=asc_pack_to_low(valid);
        valid=asc_pack_to_low(valid);
        asc_frequency_histogram_bin0(histogram,packed8,valid);
    }
}

// SingleSegment is selected once by the caller. Its instantiation has no
// segment loop, uint32 accumulator initialization, or accumulator additions.
template<bool SingleSegment>
__simd_vf__ inline void CountFrequency(
    __ubuf__ uint32_t* output, __ubuf__ uint32_t* input,
    uint32_t count, uint32_t firstExpert, uint32_t expertCount)
{
    for (uint32_t first=0;first<expertCount;first+=kLlHistogramBins) {
        const uint32_t bins=expertCount-first<kLlHistogramBins
                                ? expertCount-first : kLlHistogramBins;
        vector_uint32_t low,high;
        if constexpr (SingleSegment) {
            vector_uint16_t histogram;
            asc_duplicate_scalar(histogram,uint16_t(0));
            AccumulateFrequency(histogram,input,count,firstExpert+first,bins);
            asc_unpack_lower(low,histogram);
            asc_unpack_upper(high,histogram);
        } else {
            asc_duplicate_scalar(low,uint32_t(0));
            asc_duplicate_scalar(high,uint32_t(0));
            const vector_bool all=asc_create_mask_b32(PAT_ALL);
            for (uint32_t begin=0;begin<count;) {
                const uint32_t size=count-begin<kLlHistogramSegment
                                        ? count-begin : kLlHistogramSegment;
                vector_uint16_t histogram;
                vector_uint32_t partLow,partHigh;
                asc_duplicate_scalar(histogram,uint16_t(0));
                AccumulateFrequency(histogram,input+begin,size,firstExpert+first,bins);
                asc_unpack_lower(partLow,histogram);
                asc_unpack_upper(partHigh,histogram);
                asc_add(low,low,partLow,all);
                asc_add(high,high,partHigh,all);
                begin+=size;
            }
        }
        uint32_t left=bins;
        vector_bool storeMask=asc_update_mask_b32(left);
        asc_storealign(output+first,low,storeMask);
        if (bins>64U) {
            storeMask=asc_update_mask_b32(left);
            asc_storealign(output+first+64U,high,storeMask);
        }
    }
}
} // namespace DispatchDedup
