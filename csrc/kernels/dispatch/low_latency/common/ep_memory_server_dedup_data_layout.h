// Copyright (c) 2026, Lu Lu
// Modified by candy_cloud_fly 2026

#ifndef DISPATCH_DEDUP_EP_MEMORY_SERVER_DEDUP_DATA_LAYOUT_H
#define DISPATCH_DEDUP_EP_MEMORY_SERVER_DEDUP_DATA_LAYOUT_H

#include "ep_memory_server_dedup_count_layout.h"
#include "ep_memory_server_dedup_input.h"
#include "ll_receive_queue.h"

namespace DispatchDedup {
#if DISPATCH_DEDUP_ASCENDC_AICORE_COMPILE
#define DISPATCH_DEDUP_EP_DATA_INLINE FORCE_INLINE_AICORE
#else
#define DISPATCH_DEDUP_EP_DATA_INLINE inline
#endif

struct EpServerDedupDataLayout {
    uint64_t quantCount = 0;
    uint64_t scaleBytes = 0;
    uint64_t headerPayloadOffset = 0, readIndexPayloadOffset = 0;
    uint64_t headerWireOffset = 0, relayReadIndexWireOffset = 0;
    uint64_t payloadBytes = 0;
    uint64_t blockCount = 0;
    uint64_t packedPayloadBytes = 0;
    uint64_t tokenStride = 0;
    uint64_t returnStride = 0;
    uint64_t sourceOffset = 0, sourceBytes = 0;
    uint64_t receiveQueueOffset = 0, receiveClaimBytes = 0, receivePositionsOffset = 0, receiveQueueBytes = 0;
    uint64_t dispatchOffset = 0, dispatchBytes = 0;
    uint64_t relayOffset = 0, relayBytes = 0, requiredBytes = 0;
};

DISPATCH_DEDUP_EP_DATA_INLINE uint64_t DispatchDedupAlign(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1U) / alignment * alignment;
}

// Host validates positive uint32 shapes and the count UB bound before calling;
// E<=1024, BS*K<2^14 and H<2^32 bound the entire domain below 2^59 bytes.
// Public buffer products are also checked against size_t by the Host builder.
DISPATCH_DEDUP_EP_DATA_INLINE EpServerDedupDataLayout DispatchServerDedupDataLayout(uint32_t bs, uint32_t h,
                                                                                    uint32_t topK, uint32_t experts,
                                                                                    EpServerDedupInputMode mode, uint32_t rankSize)
{
    const bool wireFp8 = mode != EpServerDedupInputMode::Plain16;
    EpServerDedupDataLayout layout{};
    layout.quantCount = DispatchDedupAlign(h, 256U);
    layout.scaleBytes = mode == EpServerDedupInputMode::PrequantizedFp8Packs
                            ? ((uint64_t(h) + 127U) / 128U) * 4U
                            : (wireFp8 ? DispatchDedupAlign((uint64_t(h) + 31U) / 32U, 2U) : 0);
    layout.headerPayloadOffset =
        DispatchDedupAlign(wireFp8 ? layout.quantCount + layout.scaleBytes : uint64_t(h) * 2U, 32U);
    layout.readIndexPayloadOffset = layout.headerPayloadOffset + 32U;
    const uint64_t readIndexBytes = uint64_t(topK) * 8U;
    if (layout.readIndexPayloadOffset % 480U + readIndexBytes > 480U)
        layout.readIndexPayloadOffset = DispatchDedupAlign(layout.readIndexPayloadOffset, 480U);
    layout.headerWireOffset = layout.headerPayloadOffset / 480U * 512U + layout.headerPayloadOffset % 480U;
    layout.relayReadIndexWireOffset =
        layout.readIndexPayloadOffset / 480U * 512U + layout.readIndexPayloadOffset % 480U;
    layout.payloadBytes = DispatchDedupAlign(layout.readIndexPayloadOffset + readIndexBytes, 32U);
    layout.blockCount = (layout.payloadBytes + 479U) / 480U;
    layout.packedPayloadBytes = layout.blockCount * 480U;
    layout.tokenStride = layout.blockCount * 512U;
    layout.returnStride = (uint64_t(h) * 2U + 479U) / 480U * 512U;
    layout.sourceOffset = kEpServerDedupControlBytes;
    layout.sourceBytes = uint64_t(bs) * topK * layout.returnStride;
    // The source-return reservation is unused until the following Combine.
    // Reuse it for the local ready queue; enlarge only when required by shape.
    const auto receiveQueue=LlReceiveQueueMemory(LlReceiveCapacity(bs,topK,experts,rankSize));
    layout.receiveQueueOffset=layout.sourceOffset;
    layout.receiveClaimBytes=receiveQueue.claimBytes;
    layout.receivePositionsOffset=layout.receiveQueueOffset+receiveQueue.positionOffset;
    layout.receiveQueueBytes=receiveQueue.bytes;
    const uint64_t sourceReservation=layout.sourceBytes>receiveQueue.bytes ? layout.sourceBytes : receiveQueue.bytes;
    layout.dispatchOffset = DispatchDedupAlign(layout.sourceOffset + sourceReservation, 512U);
    layout.dispatchBytes = uint64_t(experts) * bs * layout.tokenStride;
    layout.relayOffset = DispatchDedupAlign(layout.dispatchOffset + layout.dispatchBytes, 512U);
    // Historical Combine reservation, unused by Dispatch. The new pull Combine
    // must define its stage/result allocation before consuming the five tables.
    layout.relayBytes = uint64_t(8U) * bs * topK * layout.returnStride;
    layout.requiredBytes = layout.relayOffset + layout.relayBytes;
    return layout;
}

// Plain/internal-quantization planning; the actual EP is explicit.
DISPATCH_DEDUP_EP_DATA_INLINE EpServerDedupDataLayout DispatchServerDedupDataLayout(uint32_t bs, uint32_t h,
                                                                                    uint32_t topK, uint32_t experts,
                                                                                    bool quantized, uint32_t rankSize)
{
    return DispatchServerDedupDataLayout(
        bs, h, topK, experts, quantized ? EpServerDedupInputMode::Quantize16 : EpServerDedupInputMode::Plain16, rankSize);
}

DISPATCH_DEDUP_EP_DATA_INLINE uint64_t DispatchServerDedupDispatchOffset(const EpServerDedupDataLayout& layout,
                                                                         uint32_t source, uint32_t destination,
                                                                         uint32_t localExpert, uint32_t q,
                                                                         uint32_t ranks, uint32_t localExperts,
                                                                         uint32_t bs)
{
    return layout.dispatchOffset +
           (((uint64_t(source) + destination) % ranks * localExperts + localExpert) * bs + q) * layout.tokenStride;
}

#undef DISPATCH_DEDUP_EP_DATA_INLINE
}  // namespace DispatchDedup
#endif
