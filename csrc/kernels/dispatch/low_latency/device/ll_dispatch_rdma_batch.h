#pragma once
#include "ll_transport.h"

namespace LlTransport {
// Sender-only batching. Counts have already completed their RDMA use of this
// core's staging area. Receiver relay and combine retain the synchronous API.
class DispatchRdmaBatch {
    static constexpr uint32_t kMaxPackets = 8U;
    const __gm__ Args* args_ = nullptr;
    uint32_t bytes_ = 0, capacity_ = 0, count_ = 0;
    uint32_t peers_[kMaxPackets]{};
    uint64_t offsets_[kMaxPackets]{};
    LlStageTimer::DeviceTimer* timer_ = nullptr;

    __aicore__ inline bool FirstPeer(uint32_t index) const {
        for (uint32_t i = 0; i < index; ++i) if (peers_[i] == peers_[index]) return false;
        return true;
    }
public:
    __aicore__ inline void Init(const __gm__ Args* args, uint32_t bytes, LlStageTimer::DeviceTimer* timer) {
        args_ = args; bytes_ = bytes; timer_ = timer; count_ = 0;
        if (!bytes || bytes % 512U || bytes > LlRdma::kCoreBytes) { Trap(); return; }
        const uint32_t slots = LlRdma::kCoreBytes / bytes;
        capacity_ = slots < kMaxPackets ? slots : kMaxPackets;
    }
    __aicore__ inline bool Pending() const { return count_ != 0U; }

    // Caller has consumed V_MTE3 for this packet and retired its previous lane
    // use. Stage completion releases UB; NIC reads only this immutable GM slot.
    // SDK scratch may clobber the packet's first 128B after staging: sender
    // repacks all body bytes on reuse, while flags at 480+512*n remain intact.
    __aicore__ inline void Enqueue(uint32_t peer, uint64_t offset, LocalTensor<uint8_t> packet) {
        if (!capacity_ || count_ >= capacity_ || !RemoteWrite(args_, peer)) { Trap(); return; }
        Check(args_, peer, offset, bytes_);
        auto base = Stage(args_, capacity_ * bytes_);
        auto stage = base + uint64_t(count_) * bytes_;
        if (timer_) timer_->Start(LlStageTimer::SendRdmaStage);
        DataCopy(Tensor(stage, true), packet, bytes_);
        Sync<HardEvent::MTE3_S>(7U);
        if (timer_) timer_->End(LlStageTimer::SendRdmaStage);
        peers_[count_] = peer; offsets_[count_] = offset;
        if (timer_) timer_->Start(LlStageTimer::SendRdmaBodyPost);
        auto dst = Address(args_, peer, offset);
        for (uint32_t b = 0; b < bytes_; b += 512U)
            Put(args_, peer, dst + b, stage + b, 480U, packet);
        if (timer_) timer_->End(LlStageTimer::SendRdmaBodyPost);
        ++count_;
        // Flush only AFTER staging the current packet. Flushing beforehand
        // would let SDK scratch overwrite a not-yet-staged payload.
        if (count_ == capacity_) Flush(packet);
    }

    // Caller supplies a retired UB lane. Every distinct peer's bodies complete
    // before any flags, and every flag completes before any slot is recycled.
    // No cross-peer or cross-QP completion ordering is assumed.
    __aicore__ inline void Flush(LocalTensor<uint8_t> scratch) {
        if (!count_) return;
        if (timer_) timer_->Start(LlStageTimer::SendRdmaBodyWait);
        for (uint32_t i = 0; i < count_; ++i) if (FirstPeer(i)) Quiet(peers_[i], scratch);
        if (timer_) timer_->End(LlStageTimer::SendRdmaBodyWait);
        auto base = Stage(args_, capacity_ * bytes_);
        if (timer_) timer_->Start(LlStageTimer::SendRdmaFlagPost);
        for (uint32_t i = 0; i < count_; ++i) {
            auto dst = Address(args_, peers_[i], offsets_[i]);
            auto src = base + uint64_t(i) * bytes_;
            for (uint32_t b = 0; b < bytes_; b += 512U)
                Put(args_, peers_[i], dst + b + 480U, src + b + 480U, 32U, scratch);
        }
        if (timer_) timer_->End(LlStageTimer::SendRdmaFlagPost);
        if (timer_) timer_->Start(LlStageTimer::SendRdmaFlagWait);
        for (uint32_t i = 0; i < count_; ++i) if (FirstPeer(i)) Quiet(peers_[i], scratch);
        if (timer_) timer_->End(LlStageTimer::SendRdmaFlagWait);
        count_ = 0;
    }
};
}
