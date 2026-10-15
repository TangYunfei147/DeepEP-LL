#pragma once
#include "../common/ll_rdma_layout.h"
#include "kernel_operator.h"
#ifdef DEEPEP_LL_RDMA
#include "device/gm2gm/engine/shmem_device_rdma.h"
#endif
namespace LlTransport {
using namespace AscendC;
using Args = DispatchDedup::CommArgs;
template<HardEvent E> __aicore__ inline void Sync(uint32_t id) {
    SetFlag<E>(id);
    WaitFlag<E>(id);
}
// Same raw-cycle bound as the pinned SDK's Ascend950 CQ timeout. Disabled for MTE.
class PollDeadline {
    bool enabled_;
    uint64_t start_;
public:
    __aicore__ explicit PollDeadline(bool enabled): enabled_(enabled), start_(enabled ? GetSystemCycle() : 0U) {}
    __aicore__ inline void Check() {
        if (enabled_ && GetSystemCycle()-start_ > 60000000000ULL) {
            AscendC::printf("LL RDMA receive timeout on core %u\n", GetBlockIdx());
            Trap();
        }
    }
};
__aicore__ inline bool Rdma(const __gm__ Args* a) { return a->reservedConfig[0] == LlRdma::kBackend; }
__aicore__ inline void Check(const __gm__ Args* a, uint32_t peer, uint64_t offset, uint64_t bytes) {
    if (peer >= uint32_t(a->rankSize) || offset > a->peerMemBytes || bytes > a->peerMemBytes - offset) Trap();
}
__aicore__ inline GM_ADDR Address(const __gm__ Args* a, uint32_t peer, uint64_t offset) {
    return Rdma(a) ? a->reservedAddresses[0] + uint64_t(peer) * 1024U + offset : a->peerMems[peer] + offset;
}
__aicore__ inline GlobalTensor<uint8_t> Tensor(GM_ADDR p, bool uncached=false) {
    GlobalTensor<uint8_t> t; t.SetGlobalBuffer(p);
    if (uncached) t.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    return t;
}
__aicore__ inline GM_ADDR Stage(const __gm__ Args* a, uint32_t bytes) {
    if (GetBlockNum() != a->reservedConfig[1] || GetBlockIdx() >= a->reservedConfig[1] || bytes > LlRdma::kCoreBytes) Trap();
    return a->reservedAddresses[1] + uint64_t(GetBlockIdx()) * LlRdma::kCoreBytes;
}
__aicore__ inline void Put(const __gm__ Args* a, uint32_t peer, GM_ADDR dst, GM_ADDR src,
                           uint32_t bytes, LocalTensor<uint8_t> scratch, uint32_t syncId=7U) {
#ifdef DEEPEP_LL_RDMA
    aclshmemx_roce_qp_put_nbi(dst, src, (__ubuf__ uint8_t*)scratch.GetPhyAddr(), bytes, peer, GetBlockIdx(), syncId);
#else
    Trap();
#endif
}
__aicore__ inline void Quiet(uint32_t peer, LocalTensor<uint8_t> scratch, uint32_t syncId=7U) {
#ifdef DEEPEP_LL_RDMA
    aclshmemx_roce_qp_quiet(peer, GetBlockIdx(), (__ubuf__ uint8_t*)scratch.GetPhyAddr(), syncId);
#else
    Trap();
#endif
}
// A packet is a bounded batch: all 480B bodies complete before any 32B flag
// writes are submitted. No cross-QP ordering is assumed. The second quiet owns
// staging reuse. This conservative first implementation serializes packets;
// its completion and WQE costs must be measured before introducing a ring.
__aicore__ inline void Packet(const __gm__ Args* a, uint32_t peer, uint64_t offset,
                              LocalTensor<uint8_t> packet, uint32_t bytes, uint32_t syncId=7U) {
    Check(a, peer, offset, bytes);
    auto dst = Address(a, peer, offset);
    if (!Rdma(a) || peer == uint32_t(a->rank)) { DataCopy(Tensor(dst,Rdma(a)), packet, bytes); return; }
    if (bytes == 0U || bytes % 512U || syncId >= 8U) { Trap(); return; }
    auto stage = Stage(a, bytes);
    PipeBarrier<PIPE_ALL>();
    DataCopy(Tensor(stage,true), packet, bytes);
    Sync<HardEvent::MTE3_S>(syncId); // registered GM complete before Scalar posts NIC WQEs
    for (uint32_t b = 0; b < bytes; b += 512U) Put(a, peer, dst + b, stage + b, 480U, packet, syncId);
    Quiet(peer, packet, syncId);
    for (uint32_t b = 0; b < bytes; b += 512U) Put(a, peer, dst + b + 480U, stage + b + 480U, 32U, packet, syncId);
    Quiet(peer, packet, syncId);
    // SDK workspace borrows the first 128B after staging. Restore it because
    // relay callers reuse this packet for several destinations.
    DataCopy(packet, Tensor(stage,true), 128U);
    Sync<HardEvent::MTE2_S>(syncId);
    PipeBarrier<PIPE_ALL>();
}
// Strided 32B records. Count records publish only word 0 after words 1..7
// complete. Ready records carry no payload and are published after local stage
// producers have completed. scratch is a retired, independent >=128B UB view.
__aicore__ inline void Records(const __gm__ Args* a, uint32_t peer, uint64_t offset,
                               LocalTensor<uint8_t> records, uint32_t count, uint32_t stride,
                               bool counts, LocalTensor<uint8_t> scratch) {
    if (!count) return;
    Check(a, peer, offset, uint64_t(count - 1U) * stride + 32U);
    auto dst = Address(a, peer, offset);
    if (!Rdma(a) || peer == uint32_t(a->rank)) {
        DataCopyPad(Tensor(dst,Rdma(a)), records, {uint16_t(count), 32U, 0U, stride - 32U, 0U}); return;
    }
    auto stage = Stage(a, count * 32U);
    PipeBarrier<PIPE_ALL>();
    DataCopy(Tensor(stage,true), records, count * 32U);
    Sync<HardEvent::MTE3_S>(7U);
    if (counts) {
        for (uint32_t i = 0; i < count; ++i) Put(a, peer, dst + uint64_t(i)*stride + 4U, stage + i*32U + 4U, 28U, scratch);
        Quiet(peer, scratch);
    }
    for (uint32_t i = 0; i < count; ++i) Put(a, peer, dst + uint64_t(i)*stride, stage + i*32U, counts ? 4U : 32U, scratch);
    Quiet(peer, scratch);
}
__aicore__ inline void Get(const __gm__ Args* a, uint32_t peer, uint64_t offset,
                           LocalTensor<uint16_t> output, uint32_t bytes) {
    Check(a, peer, offset, bytes);
    auto src = Address(a, peer, offset);
    if (Rdma(a) && peer != uint32_t(a->rank)) {
        auto stage = Stage(a, bytes);
        PipeBarrier<PIPE_ALL>();
        Sync<HardEvent::V_S>(7U); // input-lane Cast must retire before Scalar overwrites SDK scratch
#ifdef DEEPEP_LL_RDMA
        aclshmemx_roce_qp_get_nbi(stage, src, (__ubuf__ uint8_t*)output.GetPhyAddr(), bytes, peer, GetBlockIdx(), 7U);
        Quiet(peer, output.ReinterpretCast<uint8_t>());
#else
        Trap();
#endif
        src = stage;
    }
    DataCopyPad(output.ReinterpretCast<uint8_t>(), Tensor(src,Rdma(a)), {1U, bytes, 0U, 0U, 0U}, {false, 0U, 0U, uint8_t(0)});
    // The next Get may reuse this core's GM slot.
    if (Rdma(a)) { Sync<HardEvent::MTE2_S>(7U); PipeBarrier<PIPE_ALL>(); }
}
}
