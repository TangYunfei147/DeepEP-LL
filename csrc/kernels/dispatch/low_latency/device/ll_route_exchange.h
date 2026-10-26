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
    uint32_t bytes_ = 0, epoch_ = 0;
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
    __aicore__ inline void Wait(uint64_t offset, uint32_t sequence) const {
        LlTransport::PollDeadline deadline(true);
        while (Read(offset) != sequence) deadline.Check();
    }
public:
    __aicore__ inline void Init(const __gm__ CommArgs* args, uint32_t routes, uint64_t bank) {
        args_=args;bank_=bank;bytes_=routes*sizeof(int32_t);
        if (args->rankSize!=32 || !routes || bytes_>kRouteExchangeMaxBytes || !LlTransport::Rdma(args)) {
            AscendC::Trap(); return;
        }
        const uint64_t counter=kRouteEpoch+uint64_t(AscendC::GetBlockIdx())*512U;
        const uint32_t previous=Read(counter);
        if (previous==UINT32_MAX) { AscendC::Trap(); return; }
        epoch_=previous+1U;Write(counter,epoch_);
    }
    __aicore__ inline uint32_t Peer() const { return uint32_t(args_->rank)^16U; }
    __aicore__ inline void PostBody(AscendC::LocalTensor<int32_t> routes, AscendC::LocalTensor<uint8_t> scratch) {
        if (AscendC::GetBlockIdx()!=0U) return;
        // Routes reside in registered symmetric GM; no source points to UB.
        AscendC::DataCopy(LlTransport::Tensor(Local(bank_+kRouteTx),true), routes.ReinterpretCast<uint8_t>(),
                         (bytes_+31U)/32U*32U);
        LlTransport::Sync<AscendC::HardEvent::MTE3_S>(7U);
        Write(bank_+kRouteSequence,epoch_);
        LlTransport::Put(args_,Peer(),LlTransport::Address(args_,Peer(),bank_+kRouteRx),
                         Local(bank_+kRouteTx),bytes_,scratch);
        // Ordinary HNS Put submits immediately and retires its UB workspace
        // before returning. Only the registered GM source remains live: count
        // computation may reuse scratch while the body progresses on the NIC.
    }
    __aicore__ inline void PublishCommit(AscendC::LocalTensor<uint8_t> scratch) {
        if (AscendC::GetBlockIdx()!=0U) return;
        // Never expose a route commit before successful body completion.
        LlTransport::Quiet(Peer(),scratch);
        LlTransport::Put(args_,Peer(),LlTransport::Address(args_,Peer(),bank_+kRouteCommit),
                         Local(bank_+kRouteSequence),sizeof(uint32_t),scratch);
        // Commit progresses without quiet. FinishSend retires its GM sequence
        // source and CQ before this core leaves the count phase.
    }
    __aicore__ inline void ReadPeer(AscendC::LocalTensor<int32_t> routes) {
        Wait(bank_+kRouteCommit,epoch_);
        LlTransport::Sync<AscendC::HardEvent::S_MTE2>(7U);
        AscendC::DataCopy(routes.ReinterpretCast<uint8_t>(), LlTransport::Tensor(Local(bank_+kRouteRx),true),
                         (bytes_+31U)/32U*32U);
        LlTransport::Sync<AscendC::HardEvent::MTE2_S>(7U);
    }
    __aicore__ inline void FinishSend(AscendC::LocalTensor<uint8_t> scratch) {
        if (AscendC::GetBlockIdx()!=0U) return;
        // This waits only for our outbound commit, never for peer/local readers.
        LlTransport::Quiet(Peer(),scratch);
    }
};
}
