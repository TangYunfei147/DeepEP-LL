#pragma once

#include <cstdint>
#include "ep_memory_server_dedup_protocol.h"

namespace DispatchDedup {
#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
#define LL_VECTOR_TAIL_INLINE __aicore__ inline
#else
#define LL_VECTOR_TAIL_INLINE inline
#endif

// Only asynchronous, deduplicated remote senders use this candidate. Local
// senders and every other input mode retain their original scalar dependency.
LL_VECTOR_TAIL_INLINE bool LlUsePrequantVectorTail(bool prequantized, bool dedup,
    bool asynchronous, uint32_t core, uint32_t sendCores)
{
    return prequantized && dedup && asynchronous && core < sendCores;
}

struct LlTailWordStore {
    uint32_t blockByteOffset;
    uint64_t mask;
};

// Duplicate's destination remains 32-byte aligned even for a field at +4/+8.
// The mask updates exactly one uint32 word without changing adjacent fields.
LL_VECTOR_TAIL_INLINE LlTailWordStore LlPlanTailWordStore(uint32_t byteOffset)
{
    return {byteOffset & ~31U, uint64_t(1) << ((byteOffset & 31U) / 4U)};
}

// Emit the original deduplicated wire tail. Metadata access is read-only and
// weight bits are copied as uint32, including signed zero and NaN payloads.
// Writer::Word must enqueue a V write, never a scalar store to packet storage.
template<class Writer, class Metadata>
LL_VECTOR_TAIL_INLINE void LlEmitPrequantVectorTail(Writer& writer, Metadata& metadata,
    uint32_t headerOffset, uint32_t indexOffset, uint32_t rank, uint32_t token,
    uint32_t firstK, uint32_t mask, uint32_t start, uint32_t topK)
{
    writer.Word(headerOffset, rank);
    writer.Word(headerOffset + 4U, token);
    writer.Word(headerOffset + 8U, firstK);
    writer.Word(headerOffset + 12U,
        uint32_t(mask == 0U ? EpServerDedupTokenType::Direct : EpServerDedupTokenType::FirstHop));
    const uint32_t localToken = token - start;
    writer.Word(indexOffset + 4U, metadata.WeightBits(localToken * topK + firstK));
    uint32_t entries = 0U;
    for (uint32_t k = firstK + 1U; mask != 0U && k < topK; ++k) {
        if ((mask & (uint32_t(1) << k)) == 0U) continue;
        const uint32_t routeIndex = token * topK + k;
        const uint32_t expert = metadata.Expert(routeIndex);
        const uint32_t q = metadata.Q(expert, routeIndex);
        ++entries;
        writer.Word(indexOffset + entries * 8U, expert | (q << 16U));
        writer.Word(indexOffset + entries * 8U + 4U, metadata.WeightBits(localToken * topK + k));
    }
    writer.Word(indexOffset, entries);
}

#undef LL_VECTOR_TAIL_INLINE
} // namespace DispatchDedup
