#pragma once
#include "kernel_operator.h"
#include <cstdint>
namespace DispatchDedup {
// histogram is zeroed and V->S ready; routes is the complete source route table.
// Only this core's token shard is materialized. Keep original route order even
// when FIRST_HOP/DIRECT issue order is rotated. No GM writes or cross-core state.
__aicore__ inline void LlBuildPrivateRouteQ(AscendC::LocalTensor<int32_t> routes,
    AscendC::LocalTensor<uint32_t> histogram,AscendC::LocalTensor<uint32_t> table,
    uint32_t begin,uint32_t count,uint32_t k) {
    for (uint32_t i=0;i<begin;++i) {
        const uint32_t expert=uint32_t(routes.GetValue(i));
        histogram.SetValue(expert,histogram.GetValue(expert)+1U);
    }
    for (uint32_t j=0;j<count;++j) {
        const uint32_t index=begin+j,expert=uint32_t(routes.GetValue(index));
        const uint32_t q=histogram.GetValue(expert);
        // Preserve the old first-token special case, including duplicate IDs.
        table.SetValue(j,index<k ? 0U:q);
        histogram.SetValue(expert,q+1U);
    }
}
}
