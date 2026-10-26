#pragma once
#include "ll_transport.h"

namespace LlTransport {
// Sender-only batching. Counts have already completed their RDMA use of this
// core's staging area. Receiver relay and combine retain the synchronous API.
class DispatchRdmaBatch {
    static constexpr uint32_t kMaxPackets = 8U;
    const __gm__ Args* args_ = nullptr;
    uint32_t bytes_ = 0, slotBytes_ = 0, capacity_ = 0, count_ = 0;
    bool gated_ = false;
    GM_ADDR stageBase_ = nullptr;
    uint32_t peers_[kMaxPackets]{};
    GM_ADDR destinations_[kMaxPackets]{};
    LlStageTimer::DeviceTimer* timer_ = nullptr;

    __aicore__ inline bool FirstPeer(uint32_t index) const {
        for (uint32_t i = 0; i < index; ++i) if (peers_[i] == peers_[index]) return false;
        return true;
    }
    __aicore__ inline void ClearBodyGate(GM_ADDR stage) {
        GlobalTensor<uint32_t> gate;
        gate.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(stage + 480U));
        gate.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        // Preserve the neighboring 32B payload in the same scalar cacheline:
        // its MTE3 staging write has retired before this read/modify/write.
        DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(gate);
        for (uint32_t word = 0; word < 8U; ++word) gate.SetValue(word, 0U);
        DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(gate);
        DataSyncBarrier<MemDsbT::DDR>();
    }
public:
    __aicore__ inline void Init(const __gm__ Args* args, uint32_t bytes, LlStageTimer::DeviceTimer* timer) {
        if (count_) { Trap(); return; } // never discard in-flight slots on reconfiguration
        args_ = args; bytes_ = bytes; timer_ = timer; count_ = 0;
        if (!bytes || bytes % 512U || bytes > LlRdma::kCoreBytes) { Trap(); return; }
        // A private 32B GM gate copy preserves Compact/KeepFlags exactly while
        // the large body carries a zero first flag block. No UB flag mutation.
        slotBytes_ = bytes + 32U;
        gated_ = slotBytes_ <= LlRdma::kCoreBytes;
        const uint32_t slots = gated_ ? LlRdma::kCoreBytes / slotBytes_ : 1U;
        capacity_ = slots < kMaxPackets ? slots : kMaxPackets;
        // EP16 has no ROCE staging allocation and never enqueues remote writes.
        // The multi-host stage/QP contract and base are fixed for this kernel.
        if (uint32_t(args_->rankSize) > DispatchDedup::kLlPhysicalRanksPerHost)
            stageBase_ = Stage(args_, gated_ ? capacity_ * slotBytes_ : bytes_);
    }
    __aicore__ inline bool Pending() const { return count_ != 0U; }

    // Caller has consumed V_MTE3 for this packet and retired its previous lane
    // use. Stage completion releases UB; NIC reads only this immutable GM slot.
    // SDK scratch may clobber the packet's first 128B after staging: sender
    // repacks all body bytes on reuse, while flags at 480+512*n remain intact.
    __aicore__ inline void Enqueue(uint32_t peer, uint64_t offset, LocalTensor<uint8_t> packet) {
        // Preserve the original helper's extreme 128KiB input capability when
        // no gate copy fits. Admitted sender UB layouts are smaller than this.
        if (!gated_) { Packet(args_, peer, offset, packet, bytes_, 7U, timer_); return; }
        // The sender already selected RemoteWrite; Flush at capacity owns the
        // private slot count. Only the route-dependent range needs checking.
        Check(args_, peer, offset, bytes_);
        auto stage = stageBase_ + uint64_t(count_) * slotBytes_;
        if (timer_) timer_->Start(LlStageTimer::SendRdmaStage);
        DataCopy(Tensor(stage, true), packet, bytes_);
        DataCopy(Tensor(stage + bytes_, true), packet[480U], 32U);
        Sync<HardEvent::MTE3_S>(7U);
        ClearBodyGate(stage);
        if (timer_) timer_->End(LlStageTimer::SendRdmaStage);
        peers_[count_] = peer;
        if (timer_) timer_->Start(LlStageTimer::SendRdmaBodyPost);
        auto dst = RemoteAddress(args_, peer, offset);
        destinations_[count_] = dst;
        // Receiver sums every 32B flag block. With the first block zero, even
        // an arbitrarily reordered/partially visible body cannot signal ready.
        Put(args_, peer, dst, stage, bytes_, packet);
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
        if (timer_) timer_->Start(LlStageTimer::SendRdmaFlagPost);
        for (uint32_t i = 0; i < count_; ++i) {
            auto dst = destinations_[i];
            auto gate = stageBase_ + uint64_t(i) * slotBytes_ + bytes_;
            // Only the original first flag is published after all bodies quiet.
            // Other flags were transported in the body; final wire bytes match.
            Put(args_, peers_[i], dst + 480U, gate, 32U, scratch);
        }
        if (timer_) timer_->End(LlStageTimer::SendRdmaFlagPost);
        if (timer_) timer_->Start(LlStageTimer::SendRdmaFlagWait);
        for (uint32_t i = 0; i < count_; ++i) if (FirstPeer(i)) Quiet(peers_[i], scratch);
        if (timer_) timer_->End(LlStageTimer::SendRdmaFlagWait);
        count_ = 0;
    }
};
}
