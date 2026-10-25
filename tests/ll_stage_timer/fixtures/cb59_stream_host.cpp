#include <algorithm>
#include <cassert>
#include <cstdint>
#include "csrc/kernels/dispatch/low_latency/common/ep_memory_server_dedup_layout.h"
#include "csrc/kernels/dispatch/low_latency/common/ll_stream_receive.h"

using namespace DispatchDedup;

int main()
{
    static_assert(kLlCb59Stream && kLlCb59Replay, "stream keeps the replay family layout");
    static_assert(kLlCoreProtocolVersion == 22028U, "stream wheel identity");
    unsigned admitted = 0, streamRejectedOnly = 0, fallbackChecked = 0;
    for (uint32_t ranks : {16U, 32U})
    for (uint32_t b : {1U, 2U, 8U, 32U, 64U})
    for (uint32_t experts : {32U, 384U, 1024U})
    for (uint32_t k : {1U, 8U, 32U})
    for (uint32_t h : {256U, 3584U, 7168U, 16000U, 32768U})
    for (auto mode : {EpServerDedupInputMode::Plain16, EpServerDedupInputMode::Quantize16,
                      EpServerDedupInputMode::PrequantizedFp8Packs}) {
        EpMemoryServerDedupLayout layout{};
        if (BuildDispatchServerDedupLayout(ranks, b, h, k, experts, &layout, mode, 56U) != 0) continue;
        const bool oldFits = Cb59Replay::PacketUbFits(layout, b, k, experts, ranks, 56U);
        // M>32 always retains the old path, including for a small actual B.
        assert(DispatchDedupPacketUbFits(layout, b, k, experts, ranks, 56U, true) == oldFits);
        if (ranks != 16U || b > 32U) {
            assert(DispatchDedupPacketUbFits(layout, b, k, experts, ranks, 56U, false) == oldFits);
            ++fallbackChecked;
            continue;
        }
        const bool streamFits = Cb59Replay::StreamUbFits(layout, b, k, experts, ranks, 56U);
        assert(DispatchDedupPacketUbFits(layout, b, k, experts, ranks, 56U, false) == streamFits);
        if (oldFits && !streamFits) {
            // A small selected stream case must reject its budget failure,
            // even when the original replay would have enough memory.
            ++streamRejectedOnly;
        }
        if (!streamFits) continue;
        ++admitted;
        const uint64_t reserved = LlStreamAlign((uint64_t(experts) + 55U) / 56U, 8U) * 32U;
        for (uint32_t receiver = 0; receiver < 28U; ++receiver) {
            const auto slice = LlStreamRecordSlice(experts, receiver);
            const uint64_t actualRowsBound = std::min<uint64_t>(layout.capacityRows, uint64_t(slice.Count()) * b * k);
            const uint64_t live = reserved + LlStreamCountBufferBytes(experts, ranks, receiver) +
                                  LlStreamReceiverBufferBytes(actualRowsBound, layout.data.blockCount);
            assert(live + layout.data.tokenStride + 64U <= Cb59Replay::kEpServerDedupCountUbBytes);
        }
    }
    assert(admitted && streamRejectedOnly && fallbackChecked);
    // Every currently requested B and resource input mode is admitted.
    for (uint32_t b : {1U, 2U, 4U, 8U, 16U, 24U, 32U})
    for (auto mode : {EpServerDedupInputMode::Plain16, EpServerDedupInputMode::Quantize16,
                      EpServerDedupInputMode::PrequantizedFp8Packs}) {
        EpMemoryServerDedupLayout layout{};
        assert(BuildDispatchServerDedupLayout(16U, b, 3584U, 8U, 384U, &layout, mode, 56U) == 0);
        assert(DispatchDedupPacketUbFits(layout, b, 8U, 384U, 16U, 56U, false));
    }
}
