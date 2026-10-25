// Copyright (c) 2026, Lu Lu
#pragma once
#include <cstdint>
#if defined(DEEPEP_LL_PAIR_DEVICE_CONTEXT) && !defined(DEEPEP_LL_STREAM_DEVICE_CONTEXT)
#define DEEPEP_LL_STREAM_DEVICE_CONTEXT 1
#define DEEPEP_LL_PAIR_DEFINED_STREAM_CONTEXT 1
#endif
#include "ll_stream_receive.h"
#ifdef DEEPEP_LL_PAIR_DEFINED_STREAM_CONTEXT
#undef DEEPEP_LL_STREAM_DEVICE_CONTEXT
#undef DEEPEP_LL_PAIR_DEFINED_STREAM_CONTEXT
#endif

#if defined(DEEPEP_LL_PAIR_DEVICE_CONTEXT) || (defined(__CCE__) && defined(__CCE_IS_AICORE__))
#define LL_PAIR_INLINE __aicore__ inline
#else
#define LL_PAIR_INLINE inline
#endif

namespace DispatchDedup {
struct LlPairReceivePlan {
    uint32_t begin = 0U, end = 0U;
    bool valid = false, needsPartnerPrefix = false;
    LL_PAIR_INLINE uint32_t Count() const { return end - begin; }
};

// Adjacent record slices are adjacent dense ranges even when either count is
// zero. A boundary inside a record changes the consumer, never that record's q.
LL_PAIR_INLINE LlPairReceivePlan LlMakePairReceivePlan(uint32_t receiver,
    uint32_t ownBegin, uint32_t ownCount, uint32_t partnerBegin, uint32_t partnerCount,
    uint32_t receivers = kLlRecvCoreNum) {
    if (!receivers || receivers % 2U || receiver >= receivers) return {};
    const uint64_t ownEnd = uint64_t(ownBegin) + ownCount;
    const uint64_t partnerEnd = uint64_t(partnerBegin) + partnerCount;
    if (ownEnd > UINT32_MAX || partnerEnd > UINT32_MAX) return {};
    const bool first = receiver % 2U == 0U;
    const uint32_t lowBegin = first ? ownBegin : partnerBegin;
    const uint32_t lowCount = first ? ownCount : partnerCount;
    const uint32_t highBegin = first ? partnerBegin : ownBegin;
    const uint32_t highCount = first ? partnerCount : ownCount;
    if (uint64_t(lowBegin) + lowCount != highBegin) return {};
    const uint64_t total = uint64_t(lowCount) + highCount;
    const uint64_t end = uint64_t(highBegin) + highCount;
    const uint64_t middle = uint64_t(lowBegin) + total / 2U;
    LlPairReceivePlan plan{uint32_t(first ? lowBegin : middle),
                           uint32_t(first ? middle : end), true, false};
    const uint32_t overlapBegin = plan.begin > partnerBegin ? plan.begin : partnerBegin;
    const uint32_t overlapEnd = plan.end < partnerEnd ? plan.end : uint32_t(partnerEnd);
    plan.needsPartnerPrefix = overlapBegin < overlapEnd;
    return plan;
}

LL_PAIR_INLINE uint32_t LlPairAlignedPrefixRecord(uint32_t record) { return record & ~7U; }
LL_PAIR_INLINE uint32_t LlPairPrefixBias(uint32_t record) { return record & 7U; }

// One TBuf contains immutable outgoing header, incoming snapshot (subsequently
// immutable zero/clear source), then the optionally loaded partner prefix. The
// worst-case seven leading words allow a 32B-aligned GM source for every slice.
LL_PAIR_INLINE uint64_t LlPairExtraBufferBytes(uint32_t experts, uint32_t receiver,
                                               uint32_t receivers = kLlRecvCoreNum) {
    if (!receivers || receivers % 2U || receiver >= receivers) return 0U;
    const auto own = LlStreamRecordSlice(experts, receiver, receivers);
    const auto partner = LlStreamRecordSlice(experts, receiver ^ 1U, receivers);
    if (!own.Count() && !partner.Count()) return 0U;
    return 64U + (partner.Count() ? LlStreamAlign((uint64_t(partner.Count()) + 7U) * 4U, 32U) : 0U);
}

LL_PAIR_INLINE uint64_t LlPairMaxExtraBufferBytes(uint32_t experts,
                                                 uint32_t receivers = kLlRecvCoreNum) {
    uint64_t result = 0U;
    for (uint32_t receiver = 0U; receiver < receivers; ++receiver) {
        const uint64_t bytes = LlPairExtraBufferBytes(experts, receiver, receivers);
        if (bytes > result) result = bytes;
    }
    return result;
}

// First bound the complete pair by both its record capacity and the public
// global output capacity; only then divide its actual contiguous range in two.
LL_PAIR_INLINE uint64_t LlPairOwnedRowsUpperBound(uint32_t experts, uint32_t ranks,
    uint32_t bs, uint32_t k, uint32_t receivers = kLlRecvCoreNum) {
    if (!ranks || !receivers || receivers % 2U || experts < ranks || experts % ranks) return 0U;
    const uint64_t localExperts = experts / ranks;
    const uint64_t capacity = uint64_t(ranks) * bs * (k < localExperts ? k : localExperts);
    const uint32_t extra = experts % receivers;
    const uint64_t pairRecords = uint64_t(experts / receivers) * 2U + (extra < 2U ? extra : 2U);
    const uint64_t recordCapacity = pairRecords * bs * k;
    const uint64_t pairCapacity = recordCapacity < capacity ? recordCapacity : capacity;
    return (pairCapacity + 1U) / 2U;
}

LL_PAIR_INLINE uint32_t LlPairMailboxIdentity(uint32_t experts, uint32_t producer, uint32_t magic) {
    return (experts & 0xffffU) | ((magic & 1U) << 16U) | ((producer & 0xffU) << 24U);
}
} // namespace DispatchDedup
#undef LL_PAIR_INLINE
