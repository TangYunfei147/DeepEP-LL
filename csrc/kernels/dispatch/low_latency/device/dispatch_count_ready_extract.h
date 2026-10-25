#pragma once
#include <cstdint>
#include "c_api/asc_simd.h"

namespace DispatchDedup {
// One immutable, packed 32B snapshot record per source/expert: word0 is the
// float1.0 ready bit pattern, word1 is an unconverted uint32 count. Both gathers
// use element indices and the same 32B-aligned base. Inactive tail lanes perform
// no UB read; count stores are masked and cannot modify the padded output tail.
// The caller owns all three buffers until V_S, including an unsuccessful poll.
__simd_vf__ inline void P1CountReadyAndExtract(__ubuf__ uint32_t* records,
    __ubuf__ uint32_t* counts,__ubuf__ uint32_t* readyResult,uint32_t count)
{
    const vector_bool all=asc_create_mask_b32(PAT_ALL);
    vector_int32_t signedIndices;
    asc_arange(signedIndices,int32_t(0));
    vector_uint32_t flagIndices=reinterpret_cast<vector_uint32_t&>(signedIndices);
    // Each source record contributes eight uint32 words to the packed snapshot.
    asc_add(flagIndices,flagIndices,flagIndices,all);
    asc_add(flagIndices,flagIndices,flagIndices,all);
    asc_add(flagIndices,flagIndices,flagIndices,all);
    vector_uint32_t one,zero,countIndices,ready;
    asc_duplicate_scalar(one,uint32_t(1));
    asc_duplicate_scalar(zero,uint32_t(0));
    asc_add(countIndices,flagIndices,one,all);
    ready=one;
    uint32_t left=count;
    for(uint32_t begin=0;begin<count;begin+=64U){
        const vector_bool valid=asc_update_mask_b32(left);
        vector_uint32_t flags,values,nextReady;
        asc_gather(flags,records+begin*8U,flagIndices,valid);
        asc_gather(values,records+begin*8U,countIndices,valid);
        const vector_bool matches=asc_eq_scalar(flags,uint32_t(0x3f800000U),valid);
        // Retain failures from preceding blocks; ignore inactive final lanes.
        asc_select(nextReady,ready,zero,matches);
        asc_select(ready,nextReady,ready,valid);
        asc_storealign(counts+begin,values,valid);
    }
    vector_uint32_t complete;
    asc_reduce_min(complete,ready,all);
    uint32_t oneLane=1U;
    const vector_bool resultMask=asc_update_mask_b32(oneLane);
    // ReduceMin returns its scalar in lane0. Other result lanes are not used.
    asc_storealign(readyResult,complete,resultMask);
}
}
