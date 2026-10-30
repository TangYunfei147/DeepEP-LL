#include "../common/ll_reference_profile_device.h"
// Copyright (c) 2026, Lu Lu
#pragma once

#include "kernel_operator.h"
#include "../cb59/count_layout.h"
#include "../common/ll_core_config.h"
#ifndef DEEPEP_LL_STREAM_DEVICE_CONTEXT
#define DEEPEP_LL_STREAM_DEVICE_CONTEXT 1
#define DEEPEP_LL_STREAM_COUNTS_OWNS_CONTEXT 1
#endif
#include "../common/ll_stream_receive.h"
#ifdef DEEPEP_LL_STREAM_COUNTS_OWNS_CONTEXT
#undef DEEPEP_LL_STREAM_DEVICE_CONTEXT
#undef DEEPEP_LL_STREAM_COUNTS_OWNS_CONTEXT
#endif

namespace DispatchDedup::Cb59Stream {
using namespace AscendC;

// This protocol owns one immutable contiguous count-record slice per R core.
// It publishes only to later owners. Each consumer clears its own matrix column,
// so no shared flag is reset while another consumer may still need it. Both old
// and streaming calls leave the same matrices clear for ping-pong reuse.
class StreamCounts {
public:
/* LL_REFERENCE_BEGIN */
__aicore__ inline void SetProfile(LlReferenceProfile::DeviceProfile* profile) { profile_ = profile; }

/* LL_REFERENCE_END */

    __aicore__ inline void Init(const __gm__ CommArgs* args, GM_ADDR expertTokenNumsOut,
                                GM_ADDR sendCountsOut, uint32_t bs, uint32_t k, uint32_t experts,
                                uint32_t expertTokenNumsType, uint32_t magic, TPipe* pipe)
    {
        pipe_ = pipe;

        ranks_ = uint32_t(args->rankSize);
        experts_ = experts;
        receivers_ = GetBlockNum() / 2U;
        receiver_ = GetBlockIdx() - receivers_;
        slice_ = LlStreamRecordSlice(experts, receiver_, receivers_);
        active_ = experts < receivers_ ? experts : receivers_;
        expertTokenNumsType_ = expertTokenNumsType;
        const auto layout = Cb59Replay::DispatchServerDedupCountLayout(experts, GetBlockNum());
        sumWordOffset_ = layout.cumsumOffset / sizeof(uint32_t);
        state_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(args->peerMems[args->rank] +
            DispatchServerDedupWindowOffset(magic) + layout.cntOffset));
        sendCounts_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(sendCountsOut));
        expertCounts_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(expertTokenNumsOut));
        (void)bs;
        (void)k;
    }

    template<class Pool>
    __aicore__ inline void Prepare(Pool& pool)
    {
        if (slice_.Count() == 0U) return;
        const uint32_t n = slice_.Count();
        const uint32_t followers = active_ - receiver_ - 1U;
        const uint32_t expertEnds = slice_.end / ranks_ - slice_.begin / ranks_;
        Allocate(pool, countSnapshotBuf_, n * 32U);
        Allocate(pool, countClearBuf_, n * 32U);
        Allocate(pool, localPrefixBuf_, uint32_t(LlStreamAlign(uint64_t(n) * 4U, 32U)));
        Allocate(pool, publicPrefixBuf_, uint32_t(LlStreamAlign(uint64_t(n) * 4U, 32U)));
        Allocate(pool, sumPublishBuf_, uint32_t(LlStreamAlign(uint64_t(followers) * 32U, 256U)));
        Allocate(pool, sumSnapshotBuf_, receiver_ * 32U);
        Allocate(pool, sumClearBuf_, receiver_ * 32U);
        Allocate(pool, expertCountsBuf_, uint32_t(LlStreamAlign(uint64_t(expertEnds) * 8U, 32U)));

        ReadCounts();


/* LL_REFERENCE_BEGIN */
if (profile_) profile_->Mark<12>();
/* LL_REFERENCE_END */
        auto snapshot = countSnapshotBuf_.Get<uint32_t>();
        auto prefix = localPrefixBuf_.Get<uint32_t>();
        for (uint32_t i = 0U; i < n; ++i) {
            count_ += snapshot.GetValue(i * 8U + 1U);
            prefix.SetValue(i, count_);
        }

        PublishSums();
        ClearCounts();
        ReadPredecessors();
        ClearPredecessors();
/* LL_REFERENCE_BEGIN */
if (profile_) profile_->Mark<13>();
/* LL_REFERENCE_END */

        PublishOutputs();
        // No Reset or global drain here. Every MTE3 source remains allocated and
        // immutable while the receiver operates in the remainder of this pool.
    }

    __aicore__ inline LocalTensor<uint32_t> LocalPrefix() { return localPrefixBuf_.Get<uint32_t>(); }
    __aicore__ inline uint32_t RecordBegin() const { return slice_.begin; }
    __aicore__ inline uint32_t RecordCount() const { return slice_.Count(); }
    __aicore__ inline uint32_t OutputBegin() const { return outputBegin_; }
    __aicore__ inline uint32_t Count() const { return count_; }
    __aicore__ inline uint32_t UbBytes() const { return ubBytes_; }


/* LL_REFERENCE_BEGIN */
private:
    LlReferenceProfile::DeviceProfile* profile_ = nullptr;

public:
/* LL_REFERENCE_END */
private:
    template<class Pool>
    __aicore__ inline void Allocate(Pool& pool, TBuf<>& buffer, uint32_t bytes) {
        if (bytes) { pool.InitBuffer(buffer, bytes); ubBytes_ += bytes; }
    }

    template<HardEvent event>
    __aicore__ inline void Sync() {
        const auto id = pipe_->FetchEventID(event);
        SetFlag<event>(id);
        WaitFlag<event>(id);
    }

    __aicore__ inline void ReadCounts() {
/* LL_REFERENCE_BEGIN */
if (profile_) profile_->Mark<10>();
/* LL_REFERENCE_END */

/* LL_REFERENCE_BEGIN */
if (profile_) profile_->Mark<24>();
/* LL_REFERENCE_END */

        auto snapshot = countSnapshotBuf_.Get<uint32_t>();
        const DataCopyParams copy{uint16_t(slice_.Count()), 1U, 15U, 0U};
        while (true) {
            DataCopy(snapshot, state_[uint64_t(slice_.begin) * 128U], copy);
            Sync<HardEvent::MTE2_S>();
            bool ready = true;
            for (uint32_t i = 0U; i < slice_.Count(); ++i)
                ready = ready && snapshot.GetValue(i * 8U) == 0x3f800000U;
            if (ready)
/* LL_REFERENCE_BEGIN */
{
/* LL_REFERENCE_END */

/* LL_REFERENCE_BEGIN */
if (profile_) profile_->Mark<25>();

/* LL_REFERENCE_END */

/* LL_REFERENCE_BEGIN */
{
/* LL_REFERENCE_END */

/* LL_REFERENCE_BEGIN */
if (profile_) profile_->Mark<11>();

/* LL_REFERENCE_END */
return;
/* LL_REFERENCE_BEGIN */
}
/* LL_REFERENCE_END */

/* LL_REFERENCE_BEGIN */
}
/* LL_REFERENCE_END */

            // Retire all Scalar snapshot reads before MTE2 overwrites it.
            Sync<HardEvent::S_MTE2>();
        }

/* LL_REFERENCE_BEGIN */
if (profile_) profile_->Mark<25>();

/* LL_REFERENCE_END */

/* LL_REFERENCE_BEGIN */
if (profile_) profile_->Mark<11>();

/* LL_REFERENCE_END */
}

    __aicore__ inline void PublishSums() {
        const uint32_t followers = active_ - receiver_ - 1U;
        if (!followers) return;
        auto publish = sumPublishBuf_.Get<uint32_t>();
        const uint32_t first = LlStreamTailExpertFirstLocal(slice_, ranks_);
        const uint32_t tail = count_ - (first ? localPrefixBuf_.Get<uint32_t>().GetValue(first - 1U) : 0U);
        const uint8_t repeats = uint8_t((followers + 7U) / 8U);
        uint64_t sumMask[2] = {0x0101010101010101ULL, 0U};
        uint64_t readyMask[2] = {0x0202020202020202ULL, 0U};
        uint64_t tailMask[2] = {0x0404040404040404ULL, 0U};
        Duplicate<uint32_t>(publish, count_, sumMask, repeats, 1U, 8U);
        Duplicate<uint32_t>(publish, 0x3f800000U, readyMask, repeats, 1U, 8U);
        Duplicate<uint32_t>(publish, tail, tailMask, repeats, 1U, 8U);
        Sync<HardEvent::V_MTE3>();
        // Retain the established 32B sum+ready publication record. Each later
        // consumer has its own copy; the source is never overwritten in-flight.
        DataCopy(state_[sumWordOffset_ + (uint64_t(receiver_) * receivers_ + receiver_ + 1U) * 8U],
                 publish, DataCopyParams{uint16_t(followers), 1U, 0U, 0U});
    }

    __aicore__ inline void ClearCounts() {
        auto clear = countClearBuf_.Get<uint32_t>();
        auto snapshot = countSnapshotBuf_.Get<uint32_t>();
        Duplicate<uint32_t>(clear, 0U, slice_.Count() * 8U);
        Sync<HardEvent::V_S>();
        // Clear ready while retaining the original count field, as cb59 does.
        for (uint32_t i = 0U; i < slice_.Count(); ++i)
            clear.SetValue(i * 8U + 1U, snapshot.GetValue(i * 8U + 1U));
        Sync<HardEvent::S_MTE3>();
        DataCopy(state_[uint64_t(slice_.begin) * 128U], clear,
                 DataCopyParams{uint16_t(slice_.Count()), 1U, 0U, 15U});
    }

    __aicore__ inline void ReadPredecessors() {
/* LL_REFERENCE_BEGIN */
if (profile_) profile_->Mark<26>();
/* LL_REFERENCE_END */

        if (!receiver_)
/* LL_REFERENCE_BEGIN */
{
/* LL_REFERENCE_END */

/* LL_REFERENCE_BEGIN */
if (profile_) profile_->Mark<27>();

/* LL_REFERENCE_END */
return;
/* LL_REFERENCE_BEGIN */
}
/* LL_REFERENCE_END */

        auto sums = sumSnapshotBuf_.Get<uint32_t>();
        const DataCopyParams copy{uint16_t(receiver_), 1U, uint16_t(receivers_ - 1U), 0U};
        while (true) {
            DataCopy(sums, state_[sumWordOffset_ + uint64_t(receiver_) * 8U], copy);
            Sync<HardEvent::MTE2_S>();
            bool ready = true;
            for (uint32_t p = 0U; p < receiver_; ++p)
                ready = ready && sums.GetValue(p * 8U + 1U) == 0x3f800000U;
            if (ready) break;
            Sync<HardEvent::S_MTE2>();
        }
        for (uint32_t p = 0U; p < receiver_; ++p) outputBegin_ += sums.GetValue(p * 8U);

/* LL_REFERENCE_BEGIN */
if (profile_) profile_->Mark<27>();

/* LL_REFERENCE_END */
}

    __aicore__ inline void ClearPredecessors() {
        if (!receiver_) return;
        auto clear = sumClearBuf_.Get<uint32_t>();
        Duplicate<uint32_t>(clear, 0U, receiver_ * 8U);
        Sync<HardEvent::V_MTE3>();
        DataCopy(state_[sumWordOffset_ + uint64_t(receiver_) * 8U], clear,
                 DataCopyParams{uint16_t(receiver_), 1U, 0U, uint16_t(receivers_ - 1U)});
    }

    __aicore__ inline uint32_t ExpertCount(uint32_t expert) {
        const uint32_t firstRecord = expert * ranks_;
        const uint32_t endLocal = (expert + 1U) * ranks_ - slice_.begin;
        auto prefix = localPrefixBuf_.Get<uint32_t>();
        uint32_t total = prefix.GetValue(endLocal - 1U);
        if (firstRecord > slice_.begin) total -= prefix.GetValue(firstRecord - slice_.begin - 1U);
        const uint32_t firstProducer = LlStreamRecordOwner(firstRecord, experts_, receivers_);
        if (firstProducer < receiver_) {
            auto sums = sumSnapshotBuf_.Get<uint32_t>();
            for (uint32_t p = firstProducer; p < receiver_; ++p) {
                const auto part = LlStreamPriorExpertPart(expert, ranks_, LlStreamRecordSlice(experts_, p, receivers_));
                if (part == LlStreamExpertPart::Whole) total += sums.GetValue(p * 8U);
                else if (part == LlStreamExpertPart::Tail) total += sums.GetValue(p * 8U + 2U);
                else if (part == LlStreamExpertPart::Invalid) AscendC::Trap();
            }
        }
        return total;
    }

    __aicore__ inline void PublishOutputs() {
/* LL_REFERENCE_BEGIN */
if (profile_) profile_->Mark<14>();
/* LL_REFERENCE_END */

        auto prefix = localPrefixBuf_.Get<uint32_t>();
        auto output = publicPrefixBuf_.Get<uint32_t>();
        for (uint32_t i = 0U; i < slice_.Count(); ++i) output.SetValue(i, outputBegin_ + prefix.GetValue(i));
        Sync<HardEvent::S_MTE3>();
        DataCopyPad(sendCounts_[slice_.begin], output,
                    DataCopyExtParams{1U, slice_.Count() * uint32_t(sizeof(uint32_t)), 0U, 0U, 0U});
        const uint32_t firstExpert = slice_.begin / ranks_;
        const uint32_t expertEnds = slice_.end / ranks_ - firstExpert;
        if (!expertEnds)
/* LL_REFERENCE_BEGIN */
{
/* LL_REFERENCE_END */

/* LL_REFERENCE_BEGIN */
if (profile_) profile_->Mark<15>();

/* LL_REFERENCE_END */
return;
/* LL_REFERENCE_BEGIN */
}
/* LL_REFERENCE_END */

        auto experts = expertCountsBuf_.Get<int64_t>();
        for (uint32_t i = 0U; i < expertEnds; ++i) {
            const uint32_t expert = firstExpert + i;
            const uint32_t endLocal = (expert + 1U) * ranks_ - slice_.begin;
            experts.SetValue(i, expertTokenNumsType_ == 0U ?
                uint64_t(outputBegin_) + prefix.GetValue(endLocal - 1U) : uint64_t(ExpertCount(expert)));
        }
        Sync<HardEvent::S_MTE3>();
        DataCopyPad(expertCounts_[firstExpert], experts,
                    DataCopyExtParams{1U, expertEnds * uint32_t(sizeof(int64_t)), 0U, 0U, 0U});

/* LL_REFERENCE_BEGIN */
if (profile_) profile_->Mark<15>();

/* LL_REFERENCE_END */
}

    TPipe* pipe_ = nullptr;

    LlStreamSlice slice_{};
    uint32_t ranks_ = 0U, experts_ = 0U, receiver_ = 0U, receivers_ = 0U, active_ = 0U;
    uint32_t count_ = 0U, outputBegin_ = 0U, ubBytes_ = 0U, expertTokenNumsType_ = 0U;
    uint64_t sumWordOffset_ = 0U;
    GlobalTensor<uint32_t> state_, sendCounts_;
    GlobalTensor<int64_t> expertCounts_;
    TBuf<> countSnapshotBuf_, countClearBuf_, localPrefixBuf_, publicPrefixBuf_;
    TBuf<> sumPublishBuf_, sumSnapshotBuf_, sumClearBuf_, expertCountsBuf_;
};
}  // namespace DispatchDedup::Cb59Stream
