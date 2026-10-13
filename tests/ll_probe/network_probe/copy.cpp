#include "kernel_operator.h"
using namespace AscendC;
extern "C" __global__ __aicore__ void network_mte_copy(GM_ADDR src, GM_ADDR dst) {
    TPipe pipe;
    TBuf<TPosition::VECCALC> buf;
    pipe.InitBuffer(buf, 32);
    auto ub = buf.Get<uint8_t>();
    GlobalTensor<uint8_t> in, out;
    in.SetGlobalBuffer(src); out.SetGlobalBuffer(dst);
    DataCopy(ub, in, 32);
    SetFlag<HardEvent::MTE2_MTE3>(0);
    WaitFlag<HardEvent::MTE2_MTE3>(0);
    DataCopy(out, ub, 32);
    PipeBarrier<PIPE_ALL>();
}
extern "C" void network_mte_copy_do(void* stream, void* src, void* dst) {
    network_mte_copy<<<1, nullptr, stream>>>(static_cast<uint8_t*>(src), static_cast<uint8_t*>(dst));
}
