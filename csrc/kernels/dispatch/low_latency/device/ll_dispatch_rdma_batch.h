#pragma once
#include "ll_transport.h"

namespace LlTransport {
// One SO-ordered WQE per packet. Several immutable registered GM sources
// amortize CQ retirement per distinct peer; each remote sender exclusively owns its QP.
class DispatchRdmaBatch {
    static constexpr uint32_t kMaxPackets = 8U;
    const __gm__ Args* args_ = nullptr;
    uint32_t bytes_ = 0, capacity_ = 0, count_ = 0;
    GM_ADDR stageBase_ = nullptr;
    uint32_t firstPeers_ = 0U;
    uint32_t peers_[kMaxPackets]{};
public:
    __aicore__ inline void Init(const __gm__ Args* args, uint32_t bytes) {
        if (count_) { Trap(); return; }
        args_ = args; bytes_ = bytes; count_ = 0;
        if (!bytes || bytes % 512U || bytes > LlRdma::kCoreBytes) { Trap(); return; }
        const uint32_t slots = LlRdma::kCoreBytes / bytes_;
        capacity_ = slots < kMaxPackets ? slots : kMaxPackets;
        // EP16 has no RoCE staging and never enters Enqueue.
        if (uint32_t(args_->rankSize) > DispatchDedup::kLlPhysicalRanksPerHost)
            stageBase_ = Stage(args_, capacity_ * bytes_);
    }
    __aicore__ inline bool Pending() const { return count_ != 0U; }

    // Caller retired this UB lane's old readers. SDK scratch may overwrite its
    // prefix after staging; NIC reads only the stable registered GM copy.
    __aicore__ inline void Enqueue(uint32_t peer, uint64_t offset, LocalTensor<uint8_t> packet) {
        Check(args_, peer, offset, bytes_);
        auto stage = stageBase_ + uint64_t(count_) * bytes_;

        DataCopy(Tensor(stage, true), packet, bytes_);
        Sync<HardEvent::MTE3_S>(7U);

        peers_[count_] = peer;
        bool first = true;
        for (uint32_t i = 0; i < count_; ++i) if (peers_[i] == peer) { first = false; break; }
        if (first) firstPeers_ |= uint32_t(1) << count_;

        // All-flag receive checks include the last existing 32B SO tail.
        // Compact/KeepFlags markers and the shared MTE wire remain unchanged.
        Put(args_, peer, RemoteAddress(args_, peer, offset), stage, bytes_, packet);

        ++count_;
        // Stage BEFORE Flush: its SDK scratch can destroy the current UB packet.
        if (count_ == capacity_) Flush(packet);
    }

    __aicore__ inline void Flush(LocalTensor<uint8_t> scratch) {
        if (!count_) return;

        for (uint32_t i = 0; i < count_; ++i) if (firstPeers_ & (uint32_t(1) << i)) Quiet(peers_[i], scratch);

        // Arrival is not a source-reuse grant. All QPs must retire first.
        count_ = 0;
        firstPeers_ = 0U;
    }
};
}
