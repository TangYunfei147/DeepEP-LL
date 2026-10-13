// Standalone AscendC transport probe. dst must be a valid local or imported
// peer-GM mapping owned by the harness; this kernel never resolves peer ranks.
#include <cstdint>
#include "kernel_operator.h"
using namespace AscendC;

// mode: 0=empty launch, 1=UB allocation/control only,
//       2=preloaded UB -> destination GM, 3=source GM -> UB -> destination GM.
// Host contract: totalBytes%32==0; chunkBytes%32==0; 0<chunkBytes<=32768;
// 1<=lanes<=4; mode<=3; iterations>=1; allocation length>=totalBytes.
// Each core owns disjoint 32-byte blocks. Every iteration writes the same
// region; remote readback must finish before the next host launch.
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void ll_transfer_probe(
    GM_ADDR src, GM_ADDR dst, uint64_t totalBytes, uint32_t chunkBytes,
    uint32_t lanes, uint32_t iterations, uint32_t mode)
{
    if (mode == 0U) return;
    TPipe pipe;
    TBuf<TPosition::VECCALC> storage;
    pipe.InitBuffer(storage, chunkBytes * lanes);
    auto buffer = storage.Get<uint8_t>();
    // Rank-independent deterministic pattern, checked by receiving harness.
    Duplicate<uint8_t>(buffer, uint8_t(0x5a), chunkBytes * lanes);
    for (uint32_t lane = 0; lane < lanes; ++lane)
        SetFlag<HardEvent::MTE3_V>(lane);
    if (mode == 1U) {
        for (uint32_t lane = 0; lane < lanes; ++lane)
            WaitFlag<HardEvent::MTE3_V>(lane);
        PipeBarrier<PIPE_ALL>();
        return;
    }
    GlobalTensor<uint8_t> input, output;
    input.SetGlobalBuffer(src);
    output.SetGlobalBuffer(dst);
    const uint64_t blocks = totalBytes / 32U;
    const uint32_t core = GetBlockIdx(), cores = GetBlockNum();
    const uint64_t owned = blocks / cores + (core < blocks % cores ? 1U : 0U);
    const uint64_t first = core * (blocks / cores) +
                          (core < blocks % cores ? core : blocks % cores);
    uint64_t packet = 0;
    for (uint32_t repeat = 0; repeat < iterations; ++repeat) {
        for (uint64_t offset = 0; offset < owned * 32U; offset += chunkBytes) {
            const uint32_t bytes = uint32_t(owned * 32U - offset < chunkBytes
                                              ? owned * 32U - offset : chunkBytes);
            const uint32_t lane = uint32_t(packet++ % lanes);
            auto tile = buffer[lane * chunkBytes];
            // Prior send must complete before a lane can be overwritten.
            WaitFlag<HardEvent::MTE3_V>(lane);
            if (mode == 3U) {
                SetFlag<HardEvent::V_MTE2>(lane);
                WaitFlag<HardEvent::V_MTE2>(lane);
                DataCopy(tile, input[first * 32U + offset], bytes);
                SetFlag<HardEvent::MTE2_MTE3>(lane);
                WaitFlag<HardEvent::MTE2_MTE3>(lane);
            } else {
                SetFlag<HardEvent::V_MTE3>(lane);
                WaitFlag<HardEvent::V_MTE3>(lane);
            }
            // Same transfer primitive as LL SendPacket; dst can be peer GM.
            DataCopy(output[first * 32U + offset], tile, bytes);
            SetFlag<HardEvent::MTE3_V>(lane);
        }
    }
    for (uint32_t lane = 0; lane < lanes; ++lane)
        WaitFlag<HardEvent::MTE3_V>(lane);
    PipeBarrier<PIPE_ALL>();
}

extern "C" void ll_transfer_probe_do(uint32_t cores, void* stream, GM_ADDR src,
    GM_ADDR dst, uint64_t totalBytes, uint32_t chunkBytes, uint32_t lanes,
    uint32_t iterations, uint32_t mode)
{
    ll_transfer_probe<<<cores, nullptr, stream>>>(src,
        dst, totalBytes, chunkBytes, lanes, iterations, mode);
}
