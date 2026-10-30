#pragma once
#include "kernel_operator.h"
#include "ll_reference_profile_layout.h"
#include "ll_core_config.h"
#include "ll_small_b_prefix_config.h"
#include "ll_overlap_stage.h"
#ifdef DEEPEP_LL_REFERENCE_PROFILE
#include "shmem.h"
#endif

namespace LlReferenceProfile {
// Ported from kernel/ll-ndr-metadata-cores c3d8640. Each fixed point performs
// one clock + one private scalar GM store. Empty polls never read the clock.
// MTE schema7 records actual S/R/combined roles, not fictitious M cores.
class DeviceProfile {
public:
    __aicore__ static inline uint64_t EntryClock(GM_ADDR address) {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        return address ? AscendC::GetSystemCycle() : 0U;
#else
        return 0U;
#endif
    }
    __aicore__ inline void Init(GM_ADDR address) {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        record_ = address ? reinterpret_cast<__gm__ uint64_t*>(address) +
                            AscendC::GetBlockIdx() * kWordsPerCore : nullptr;
#endif
    }
    __aicore__ inline bool Enabled() const {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        return record_ != nullptr;
#else
        return false;
#endif
    }
    __aicore__ inline uint64_t Clock() const {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        return record_ ? AscendC::GetSystemCycle() : 0U;
#else
        return 0U;
#endif
    }
    template<uint32_t Word> __aicore__ inline void Value(uint64_t value) {
        static_assert(Word <= 21U || (Word >= 24U && Word <= 43U) || Word == 57U, "reference word");
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if (!record_) return;
        record_[Word] = value;
        valid_ |= uint64_t(1) << Word;
#endif
    }
    template<uint32_t Word> __aicore__ inline void Mark() {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if (record_) Value<Word>(AscendC::GetSystemCycle());
#endif
    }
    template<uint32_t Word> __aicore__ inline bool Has() const {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        return record_ && (valid_ & (uint64_t(1) << Word));
#else
        return false;
#endif
    }
    // Cooperative receive may return to TX between empty probes. Reuse one
    // waiting interval until a ready batch, without clocks on failed scans.
    __aicore__ inline void PollBegin() {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if (record_ && !waiting_) { waitBegin_ = Clock(); waiting_ = true; }
#endif
    }
    __aicore__ inline void BatchReady() {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if (!record_) return;
        ready_ = Clock(); pollTicks_ += ready_ - waitBegin_;
        if (!batches_) Value<42>(ready_);
        ++batches_;
#endif
    }
    __aicore__ inline void BatchDone() {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if (!record_) return;
        consumeTicks_ += Clock() - ready_; waiting_ = false;
#endif
    }
    __aicore__ inline void ReceiveTotals(uint32_t tasks) {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if (!record_) return;
        Value<33>(pollTicks_); Value<34>(consumeTicks_);
        Value<37>(tasks); Value<41>(batches_);
#endif
    }
    // Two classes are peer-MTE and self-MTE, not RDMA/local groups. No CQ,
    // quiet or RDMA staging time is manufactured for this backend.
    __aicore__ inline void SendBegin(bool peer, uint32_t token, uint32_t k,
                                    uint32_t expert, uint32_t destination) {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if (!record_) return;
        peer_ = peer;
        const uint64_t order = ++send_[24];
        if (peer) {
            const uint64_t n = ++send_[1];
            if (n == 1U) { send_[3] = Clock(); send_[20] = order; }
            send_[21] = order;
            if (n == 2U) {
                send_[7] = Clock(); send_[12] = token; send_[13] = k;
                send_[14] = expert; send_[15] = destination;
            }
        } else {
            if (++send_[2] == 1U) { send_[5] = Clock(); send_[22] = order; }
            send_[23] = order;
        }
#endif
    }
    __aicore__ inline bool SecondPeer() const {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        return record_ && peer_ && send_[1] == 2U;
#else
        return false;
#endif
    }
    __aicore__ inline void PacketPrepared() {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if (SecondPeer()) send_[8] = Clock();
#endif
    }
    __aicore__ inline void PutBegin() {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if (SecondPeer()) send_[9] = Clock();
#endif
    }
    __aicore__ inline void PutReturned() {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if (!record_) return;
        const uint64_t now = Clock();
        if (peer_) { send_[4] = now; if (SecondPeer()) send_[10] = now; }
        else send_[6] = now;
#endif
    }
    __aicore__ inline void MemberQ(uint64_t ticks) {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if (SecondPeer()) send_[11] += ticks;
#endif
    }
    __aicore__ inline void SendTotals(uint32_t lanes) {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if (record_) { send_[25] = lanes; Value<43>(send_[24]); }
#endif
    }
    __aicore__ inline void Detail(uint32_t word, uint64_t value) {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if (record_) record_[word] = value;
#endif
    }
    __aicore__ inline GM_ADDR PacketLogAddress() const {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        return record_ ? reinterpret_cast<GM_ADDR>(record_ + kPacketLogWord) : nullptr;
#else
        return nullptr;
#endif
    }
    __aicore__ inline void Finish(uint32_t rank, uint32_t world, uint32_t bs) {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if (!record_) return;
        Mark<57>(); // Packet-log export has finished; header/cache publication follows.
        send_[0] = 2U; // MTE packet detail contract, distinct from NDR detail1.
        for (uint32_t i = 0; i < 32U; ++i) record_[64U+i] = send_[i];
        record_[44] = DispatchDedup::kLlSendCoreNum;
        record_[45] = DispatchDedup::kLlRecvCoreNum;
        record_[46] = DispatchDedup::kLlDispatchRecvCoreNum;
        record_[47] = DispatchDedup::kLlDispatchImplementationId;
        record_[48] = DispatchDedup::kLlCb59Replay ? 0U : DispatchDedup::kLlOverlapExperimentStage;
        record_[49] = DispatchDedup::kLlSmallBPrefixCores;
        record_[50] = rank; record_[51] = AscendC::GetBlockIdx();
        record_[52] = world; record_[53] = AscendC::GetBlockNum(); record_[54] = bs;
        record_[55] = kSchema; record_[56] = 1U; record_[58] = valid_;
        record_[59] = kMagic; record_[60] = 1U; record_[61] = 1U;
        record_[62] = 0U; // No dedicated metadata cores in this algorithm.
        record_[63] = ((valid_ >> 8U) & 1U) | (((valid_ >> 18U) & 1U) << 1U) |
                      (((valid_ >> 6U) & 1U) << 2U);
        // Same reference publication: last byte avoids flushing the next core.
        dcci_cachelines(reinterpret_cast<__gm__ uint8_t*>(record_), kPacketLogWord*sizeof(uint64_t)-1U);
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
#endif
    }
private:
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    __gm__ uint64_t* record_ = nullptr;
    uint64_t valid_ = 0U, send_[32]{};
    uint64_t waitBegin_=0U, ready_=0U, pollTicks_=0U, consumeTicks_=0U, batches_=0U;
    bool peer_=false, waiting_=false;
#endif
};
} // namespace LlReferenceProfile
