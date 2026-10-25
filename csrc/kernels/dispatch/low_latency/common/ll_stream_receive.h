// Copyright (c) 2026, Lu Lu
#pragma once
#include <cstdint>
#include "ll_core_config.h"

#if defined(DEEPEP_LL_STREAM_DEVICE_CONTEXT) || (defined(__CCE__) && defined(__CCE_IS_AICORE__))
#define LL_STREAM_INLINE __aicore__ inline
#else
#define LL_STREAM_INLINE inline
#endif

namespace DispatchDedup {
struct LlStreamSlice {
    uint32_t begin = 0U, end = 0U;
    LL_STREAM_INLINE uint32_t Count() const { return end - begin; }
};

LL_STREAM_INLINE uint64_t LlStreamAlign(uint64_t n, uint64_t alignment) {
    return (n + alignment - 1U) / alignment * alignment;
}

LL_STREAM_INLINE LlStreamSlice LlStreamRecordSlice(uint32_t experts, uint32_t receiver,
                                                   uint32_t receivers = kLlRecvCoreNum) {
    if (receivers == 0U || receiver >= receivers) return {};
    const uint32_t base = experts / receivers, extra = experts % receivers;
    const uint32_t begin = receiver * base + (receiver < extra ? receiver : extra);
    return {begin, begin + base + uint32_t(receiver < extra)};
}

LL_STREAM_INLINE uint32_t LlStreamRecordOwner(uint32_t record, uint32_t experts,
                                              uint32_t receivers = kLlRecvCoreNum) {
    if (receivers == 0U || record >= experts) return receivers;
    const uint32_t base = experts / receivers, extra = experts % receivers;
    const uint32_t first = (base + 1U) * extra;
    return record < first ? record / (base + 1U) : extra + (record - first) / base;
}

LL_STREAM_INLINE bool LlUseStreamReceive(uint32_t ranks, uint32_t bs, uint32_t maxTokens) {
    return ranks == 16U && bs > 0U && bs <= 32U && maxTokens > 0U && maxTokens <= 32U;
}

// Route duplicates are not assumed absent. Each count record may contain B*K
// rows; the public output capacity gives an independent, tighter global bound.
LL_STREAM_INLINE uint64_t LlStreamOwnedRowsUpperBound(uint32_t experts, uint32_t ranks,
                                                      uint32_t bs, uint32_t k,
                                                      uint32_t receivers = kLlRecvCoreNum) {
    if (!ranks || !receivers || experts < ranks || experts % ranks) return 0U;
    const uint64_t localExperts = experts / ranks;
    const uint64_t capacity = uint64_t(ranks) * bs * (k < localExperts ? k : localExperts);
    const uint64_t records = (uint64_t(experts) + receivers - 1U) / receivers;
    const uint64_t owned = records * bs * k;
    return owned < capacity ? owned : capacity;
}

// Eight independent sources survive through the original counts.Finish drain:
// count snapshot/clear, local/public prefix, sum publication/snapshot/clear,
// and expert-count output. Only the vector sum publication needs 256B padding.
LL_STREAM_INLINE uint64_t LlStreamCountBufferBytes(uint32_t experts, uint32_t ranks,
                                                   uint32_t receiver,
                                                   uint32_t receivers = kLlRecvCoreNum) {
    if (!ranks) return 0U;
    const auto slice = LlStreamRecordSlice(experts, receiver, receivers);
    if (!slice.Count()) return 0U;
    const uint32_t active = experts < receivers ? experts : receivers;
    const uint32_t followers = active - receiver - 1U;
    const uint32_t expertEnds = slice.end / ranks - slice.begin / ranks;
    return 2U * uint64_t(slice.Count()) * 32U + 2U * LlStreamAlign(uint64_t(slice.Count()) * 4U, 32U) +
           LlStreamAlign(uint64_t(followers) * 32U, 256U) + 2U * uint64_t(receiver) * 32U +
           LlStreamAlign(uint64_t(expertEnds) * 8U, 32U);
}

LL_STREAM_INLINE uint64_t LlStreamMaxCountBufferBytes(uint32_t experts, uint32_t ranks,
                                                      uint32_t receivers = kLlRecvCoreNum) {
    uint64_t result = 0U;
    for (uint32_t receiver = 0U; receiver < receivers; ++receiver) {
        const uint64_t bytes = LlStreamCountBufferBytes(experts, ranks, receiver, receivers);
        if (bytes > result) result = bytes;
    }
    return result;
}

// Does not include the source-count records, persistent protocol buffers, or
// packet lanes. Caller subtracts all three before choosing its lane count.
LL_STREAM_INLINE uint64_t LlStreamReceiverBufferBytes(uint64_t rows, uint64_t blocks) {
    return 2U * LlStreamAlign(rows * 4U, 32U) + 16U * blocks * 32U + 256U +
           16U * LlStreamAlign((blocks + 7U) / 8U, 32U) * 4U + 256U + 16U * 36U;
}

LL_STREAM_INLINE uint32_t LlStreamTailExpertFirstLocal(LlStreamSlice slice, uint32_t ranks) {
    if (!slice.Count() || !ranks) return 0U;
    const uint32_t begin = (slice.end - 1U) / ranks * ranks;
    return begin > slice.begin ? begin - slice.begin : 0U;
}

enum class LlStreamExpertPart : uint32_t { None, Whole, Tail, Invalid };

// For an expert-ending owner, every preceding overlapping producer contributes
// either its whole sum or its last-expert partial sum. It cannot contribute a
// head-only fragment because that producer ends before the owner's first row.
LL_STREAM_INLINE LlStreamExpertPart LlStreamPriorExpertPart(uint32_t expert, uint32_t ranks,
                                                            LlStreamSlice producer) {
    const uint32_t begin = expert * ranks, end = begin + ranks;
    if (producer.end <= begin || producer.begin >= end) return LlStreamExpertPart::None;
    if (producer.end > end) return LlStreamExpertPart::Invalid;
    return producer.begin < begin ? LlStreamExpertPart::Tail : LlStreamExpertPart::Whole;
}
}  // namespace DispatchDedup
#undef LL_STREAM_INLINE
