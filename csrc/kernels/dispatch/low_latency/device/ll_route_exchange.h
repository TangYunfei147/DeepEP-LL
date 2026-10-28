#pragma once
#include "ll_transport.h"
#include "../common/ll_route_exchange_layout.h"

namespace DispatchDedup {
// Only a gateway's own count cores consume its inbox. They reconstruct counts
// for the paired source and MTE-publish them to local expert ranks. Successive
// dispatches alternate banks and must consume the peer's next epoch before
// reusing a bank. That peer cannot launch its next epoch until its prior local
// readers retire. Intervening combine/fallback calls retain their existing
// completion dependencies; no separate route-reader acknowledgement is needed.
class PairedRouteExchange {
    const __gm__ CommArgs* args_ = nullptr;
    uint64_t bank_ = 0;
    uint32_t bytes_ = 0, bodyBytes_ = 0, epoch_ = 0;
    __aicore__ inline GM_ADDR Local(uint64_t offset) const {
        return args_->peerMems[args_->rank] + offset;
    }
    __aicore__ inline uint32_t Read(uint64_t offset) const {
        AscendC::GlobalTensor<uint32_t> value;
        value.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(Local(offset)));
        value.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        AscendC::DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                        AscendC::DcciDst::CACHELINE_OUT>(value);
        return value.GetValue(0);
    }
    __aicore__ inline void Write(uint64_t offset, uint32_t sequence) const {
        AscendC::GlobalTensor<uint32_t> value;
        value.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(Local(offset)));
        value.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        value.SetValue(0, sequence);
        AscendC::DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                        AscendC::DcciDst::CACHELINE_OUT>(value);
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    }
    __aicore__ inline void WaitFrame() const {
        const uint64_t tail = bank_ + kRouteRx + bodyBytes_;
        for (;;) {
            bool ready = true;
            for (uint32_t word = 0; word < kRouteFlagBytes / sizeof(uint32_t); ++word) {
                if (Read(tail + word * sizeof(uint32_t)) != (word % 2U ? epoch_ : kRouteFlagMagic)) {
                    ready = false;
                    break;
                }
            }
            if (ready) return;
        }
    }
public:
    __aicore__ inline void Init(const __gm__ CommArgs* args, uint32_t routes, uint64_t bank) {
        args_=args;bank_=bank;bytes_=routes*sizeof(int32_t);
        bodyBytes_=(bytes_+31U)/32U*32U;
        if (args->rankSize!=32 || !routes || bytes_>kRouteExchangeMaxBytes || !LlTransport::Rdma(args)) {
            AscendC::Trap(); return;
        }
        const uint64_t counter=kRouteEpoch+uint64_t(AscendC::GetBlockIdx())*512U;
        const uint32_t previous=Read(counter);
        if (previous==UINT32_MAX) { AscendC::Trap(); return; }
        epoch_=previous+1U;Write(counter,epoch_);
    }
    __aicore__ inline void WaitReady() const { WaitFrame(); }
    __aicore__ inline uint32_t Peer() const { return uint32_t(args_->rank)^16U; }
    __aicore__ inline void PostFrame(AscendC::LocalTensor<int32_t> routes, AscendC::LocalTensor<uint8_t> scratch) {
        if (AscendC::GetBlockIdx()!=0U) return;
        // Routes reside in registered symmetric GM; no source points to UB.
        AscendC::DataCopy(LlTransport::Tensor(Local(bank_+kRouteTx),true), routes.ReinterpretCast<uint8_t>(),
                         bodyBytes_);
        // Existing count-work scratch supplies the 32B tail; no extra UB.
        auto flag = scratch.ReinterpretCast<uint32_t>();
        for (uint32_t word = 0; word < kRouteFlagBytes / sizeof(uint32_t); ++word)
            flag.SetValue(word, word % 2U ? epoch_ : kRouteFlagMagic);
        LlTransport::Sync<AscendC::HardEvent::S_MTE3>(7U);
        AscendC::DataCopy(LlTransport::Tensor(Local(bank_+kRouteTx+bodyBytes_),true), scratch, kRouteFlagBytes);
        LlTransport::Sync<AscendC::HardEvent::MTE3_S>(7U);
        LlTransport::Put(args_,Peer(),LlTransport::Address(args_,Peer(),bank_+kRouteRx),
                         Local(bank_+kRouteTx),bodyBytes_+kRouteFlagBytes,scratch);
        // Ordinary HNS Put submits immediately and retires its UB workspace
        // before returning. Only the registered GM source remains live: count
        // computation may reuse scratch while the SO frame progresses on NIC.
    }
    // Acquire the complete SO tail and issue the body load. The caller owns the
    // MTE2_V (or MTE2_S) dependency before consuming/reusing this UB view.
    __aicore__ inline void ReadPeerAsync(AscendC::LocalTensor<int32_t> routes) {
        WaitFrame();
        LlTransport::Sync<AscendC::HardEvent::S_MTE2>(7U);
        AscendC::DataCopy(routes.ReinterpretCast<uint8_t>(), LlTransport::Tensor(Local(bank_+kRouteRx),true),
                         bodyBytes_);
    }
    __aicore__ inline void FinishSend(AscendC::LocalTensor<uint8_t> scratch) {
        if (AscendC::GetBlockIdx()!=0U) return;
        // SO publishes readiness, not source reuse: retire the whole frame and
        // QP0 before core0 can send payload. Never join peer/local readers.
        LlTransport::Quiet(Peer(),scratch);
    }
};
}
