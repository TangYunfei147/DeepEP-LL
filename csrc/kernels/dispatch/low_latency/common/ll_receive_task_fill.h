#pragma once
#include "kernel_operator.h"
#include <cstdint>
namespace DispatchDedup {
// Long record interiors use aligned vector stores. Scalar edges never share
// a 32B block with an in-flight vector store from this or an adjacent record.
// Caller fences V->S once after ALL table construction, before GetValue/updates.
__aicore__ inline void LlFillTaskRange(AscendC::LocalTensor<uint32_t> dst,
                                      uint32_t offset,uint32_t first,uint32_t count) {
    if (count<32U || uint64_t(first)+count>uint64_t(INT32_MAX)+1U) {
        for(uint32_t i=0;i<count;++i) dst.SetValue(offset+i,first+i);
        return;
    }
    while ((offset&7U) && count) { dst.SetValue(offset++,first++); --count; }
    const uint32_t body=count/8U*8U;
    if (body) AscendC::CreateVecIndex(dst[offset].template ReinterpretCast<int32_t>(),int32_t(first),body);
    offset+=body; first+=body; count-=body;
    for(uint32_t i=0;i<count;++i) dst.SetValue(offset+i,first+i);
}
}
