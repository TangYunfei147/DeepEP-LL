#pragma once
#include "kernel_operator.h"
#include "../common/ll_receive_queue.h"
namespace DispatchDedup {
struct LlReceiveScalarAtomics {
    using Pointer=__gm__ uint32_t*;
    __aicore__ inline static uint32_t Add(Pointer p,uint32_t value) { return AscendC::AtomicAdd(p,value); }
    __aicore__ inline static uint32_t Cas(Pointer p,uint32_t expected,uint32_t value) {
        return AscendC::AtomicCas(p,expected,value);
    }
};
using NativeReceiveQueue=LlReadyReceiveQueue<LlReceiveScalarAtomics>;
} // namespace DispatchDedup
