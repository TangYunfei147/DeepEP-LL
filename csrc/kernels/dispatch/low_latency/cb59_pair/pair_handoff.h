// Copyright (c) 2026, Lu Lu
#pragma once
#include "kernel_operator.h"
#include "../cb59_stream/stream_counts.h"
#ifndef DEEPEP_LL_PAIR_DEVICE_CONTEXT
#define DEEPEP_LL_PAIR_DEVICE_CONTEXT 1
#define DEEPEP_LL_PAIR_HANDOFF_DEFINED_CONTEXT 1
#endif
#include "../common/ll_pair_receive.h"
#ifdef DEEPEP_LL_PAIR_HANDOFF_DEFINED_CONTEXT
#undef DEEPEP_LL_PAIR_DEVICE_CONTEXT
#undef DEEPEP_LL_PAIR_HANDOFF_DEFINED_CONTEXT
#endif

namespace DispatchDedup::Cb59Pair {
using namespace AscendC;

// A single consumer owns each mailbox. This class never reuses a StreamCounts
// DMA source and never resets its pool. All three regions of packedBuf_ remain
// allocated until the caller's original counts.Finish() drain.
class PairHandoff {
public:
    __aicore__ inline void Init(const __gm__ CommArgs* args, GM_ADDR sendCountsOut,
                                uint32_t experts, uint32_t magic, TPipe* pipe,
                                LlStageTimer::DeviceTimer* timer = nullptr) {
        pipe_ = pipe;
        (void)timer;
        experts_ = experts;
        magic_ = magic;
        receivers_ = GetBlockNum() / 2U;
        receiver_ = GetBlockIdx() - receivers_;
        if (!receivers_ || receivers_ % 2U || GetBlockNum() != kLlActiveCoreNum ||
            GetBlockIdx() < receivers_ || receiver_ >= receivers_ || args->rankSize != 16 ||
            experts < 16U || experts > kEpServerDedupMaxExperts || experts % 16U) {
            AscendC::Trap(); return;
        }
        partner_ = receiver_ ^ 1U;
        ownSlice_ = LlStreamRecordSlice(experts, receiver_, receivers_);
        partnerSlice_ = LlStreamRecordSlice(experts, partner_, receivers_);
        // EP16 with E divisible by16 has 16 or28 active owners, both even.
        // An empty record slice therefore always has an empty partner.
        if (bool(ownSlice_.Count()) != bool(partnerSlice_.Count())) { AscendC::Trap(); return; }
        const auto layout = Cb59Replay::DispatchServerDedupCountLayout(experts, GetBlockNum());
        auto* ready = args->peerMems[args->rank] + DispatchServerDedupWindowOffset(magic) +
                      layout.cntOffset + layout.cumsumFlagOffset;
        // The old full-prefix flag matrix is unused by StreamCounts. Never
        // alias its live upper-triangular sum matrix for this second protocol.
        incoming_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ready +
            (uint64_t(receiver_) * receivers_ + partner_) * 32U));
        outgoing_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ready +
            (uint64_t(partner_) * receivers_ + receiver_) * 32U));
        publicPrefix_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(sendCountsOut));
        prefixBias_ = LlPairPrefixBias(partnerSlice_.begin);
    }

    template<class Pool>
    __aicore__ inline void Prepare(Pool& pool, Cb59Stream::StreamCounts& counts) {
        if (!ownSlice_.Count()) return;
        if (counts.RecordBegin() != ownSlice_.begin || counts.RecordCount() != ownSlice_.Count()) {
            AscendC::Trap(); return;
        }
        ubBytes_ = uint32_t(LlPairExtraBufferBytes(experts_, receiver_, receivers_));
        pool.InitBuffer(packedBuf_, ubBytes_);
        auto publish = packedBuf_.Get<uint32_t>();
        publish.SetValue(0U, 0x3f800000U);
        publish.SetValue(1U, counts.OutputBegin());
        publish.SetValue(2U, counts.Count());
        publish.SetValue(3U, counts.RecordBegin());
        publish.SetValue(4U, counts.RecordCount());
        publish.SetValue(5U, LlPairMailboxIdentity(experts_, receiver_, magic_));
        publish.SetValue(6U, 0U);
        publish.SetValue(7U, 0U);
        Sync<HardEvent::S_MTE3>();
        // StreamCounts queued the public prefix before returning. Order that
        // data before readiness without a Scalar/full-pipeline completion wait.
        PipeBarrier<PIPE_MTE3>();
        DataCopy(outgoing_, publish, 8U);

        auto snapshot = packedBuf_.Get<uint32_t>()[8U];
        while (true) {
            DataCopy(snapshot, incoming_, 8U);
            Sync<HardEvent::MTE2_S>();
            if (snapshot.GetValue(0U) == 0x3f800000U) break;
            Sync<HardEvent::S_MTE2>();
        }
        partnerOutputBegin_ = snapshot.GetValue(1U);
        partnerCount_ = snapshot.GetValue(2U);
        if (snapshot.GetValue(3U) != partnerSlice_.begin || snapshot.GetValue(4U) != partnerSlice_.Count() ||
            snapshot.GetValue(5U) != LlPairMailboxIdentity(experts_, partner_, magic_) ||
            snapshot.GetValue(6U) != 0U || snapshot.GetValue(7U) != 0U) {
            AscendC::Trap(); return;
        }
        plan_ = LlMakePairReceivePlan(receiver_, counts.OutputBegin(), counts.Count(),
                                     partnerOutputBegin_, partnerCount_, receivers_);
        if (!plan_.valid) { AscendC::Trap(); return; }
        if (plan_.needsPartnerPrefix) {
            auto prefix = packedBuf_.Get<uint32_t>()[16U];
            const uint32_t aligned = LlPairAlignedPrefixRecord(partnerSlice_.begin);
            const uint32_t words = prefixBias_ + partnerSlice_.Count();
            // Only leading words are borrowed for alignment. Do not round the
            // GM tail up into a later producer or past the public tensor end.
            DataCopyPad(prefix, publicPrefix_[aligned],
                        DataCopyExtParams{1U, words * uint32_t(sizeof(uint32_t)), 0U, 0U, 0U},
                        DataCopyPadExtParams<uint32_t>{false, 0U, 0U, 0U});
            Sync<HardEvent::MTE2_S>();
        }
        // Metadata and any needed prefix have been acquired. The snapshot is
        // now the dedicated immutable zero source; clear ALL header words so a
        // later legacy full-record reduction cannot observe stale count fields.
        for (uint32_t word = 0U; word < 8U; ++word) snapshot.SetValue(word, 0U);
        Sync<HardEvent::S_MTE3>();
        DataCopy(incoming_, snapshot, 8U);
    }

    __aicore__ inline uint32_t Begin() const { return plan_.begin; }
    __aicore__ inline uint32_t Count() const { return plan_.Count(); }
    __aicore__ inline uint32_t PartnerRecordBegin() const { return partnerSlice_.begin; }
    __aicore__ inline uint32_t PartnerRecordCount() const { return partnerSlice_.Count(); }
    __aicore__ inline uint32_t PartnerOutputBegin() const { return partnerOutputBegin_; }
    __aicore__ inline uint32_t PartnerCount() const { return partnerCount_; }
    __aicore__ inline uint32_t PartnerTotalRows() const { return partnerCount_; }
    // Raw aligned view: partner record i is at PrefixBias()+i. Call only when
    // NeedsPartnerPrefix() is true; no buffer exists for empty record pairs.
    __aicore__ inline LocalTensor<uint32_t> LocalPrefixAbsolute() { return packedBuf_.Get<uint32_t>()[16U]; }
    __aicore__ inline uint32_t PrefixBias() const { return prefixBias_; }
    __aicore__ inline bool NeedsPartnerPrefix() const { return plan_.needsPartnerPrefix; }
    __aicore__ inline uint32_t UbBytes() const { return ubBytes_; }

private:
    template<HardEvent event>
    __aicore__ inline void Sync() {
        const auto id = pipe_->FetchEventID(event);
        SetFlag<event>(id);
        WaitFlag<event>(id);
    }
    TPipe* pipe_ = nullptr;
    uint32_t experts_ = 0U, magic_ = 0U, receiver_ = 0U, receivers_ = 0U, partner_ = 0U;
    uint32_t partnerOutputBegin_ = 0U, partnerCount_ = 0U, prefixBias_ = 0U, ubBytes_ = 0U;
    LlStreamSlice ownSlice_{}, partnerSlice_{};
    LlPairReceivePlan plan_{};
    GlobalTensor<uint32_t> incoming_, outgoing_, publicPrefix_;
    TBuf<> packedBuf_;
};
} // namespace DispatchDedup::Cb59Pair
