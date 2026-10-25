#include <cassert>
#include <cstdint>
#include <iostream>
#include "csrc/kernels/dispatch/low_latency/common/ep_memory_server_dedup_layout.h"
#include "csrc/kernels/dispatch/low_latency/cb59/host_layout.h"
#include "csrc/kernels/combine/low_latency/combine_layout.h"
#include "cb59_reference_layout.h"

using namespace DispatchDedup;

int main()
{
    static_assert(Cb59Replay::kEpServerDedupCountUbBytes == 194560U, "old UB arena");
    static_assert(Cb59Replay::UB_SINGLE_DMA_SIZE_MAX == 194560, "old common UB limit");
    static_assert(Cb59Replay::UB_SINGLE_PING_PONG_ADD_SIZE_MAX == 97280, "old half arena");
    static_assert(Cb59Replay::PING_PONG_SIZE == 2, "old input ring");
    static_assert(kEpServerDedupControlBytes == COMBINE_DEDUP_EP_SERVER_DEDUP_SHARED_DATA_OFFSET,
                  "dispatch and combine retain the public payload base");
    uint64_t combineDigest = 0;
    uint32_t accepted = 0, rejected = 0;
    for (uint32_t experts : {16U, 32U, 128U, 384U, 1024U}) {
        const auto actual = Cb59Replay::DispatchServerDedupCountLayout(experts, 56);
        const auto archived = Cb59Reference::DispatchServerDedupCountLayout(experts, 56);
        assert(actual.cntOffset == archived.cntOffset);
        assert(actual.cumsumOffset == archived.cumsumOffset);
        assert(actual.prefixOffset == archived.prefixOffset);
        assert(actual.prefixStride == archived.prefixStride);
        assert(actual.cumsumFlagOffset == archived.cumsumFlagOffset);
        assert(actual.counterOffset == archived.counterOffset);
        assert(actual.stateBytes == archived.stateBytes);
        assert(actual.cntOffset == 72U * 512U);
        assert(actual.cumsumOffset + 28U * 28U * 32U == actual.prefixOffset);
        assert(actual.prefixOffset + 28U * actual.prefixStride == actual.cumsumFlagOffset);
        assert(actual.cntOffset + actual.cumsumFlagOffset + 28U * 28U * 32U <= actual.counterOffset);
        assert(actual.stateBytes <= COMBINE_DEDUP_EP_SERVER_DEDUP_COMBINE_CONTROL_OFFSET);
        for (uint32_t receiver = 0; receiver < 28; ++receiver) {
            const uint64_t end = actual.cntOffset + actual.prefixOffset +
                                 receiver * actual.prefixStride + experts * 4U;
            assert(end <= actual.cntOffset + actual.cumsumFlagOffset);
        }
    }
    const auto known = Cb59Replay::DispatchServerDedupCountLayout(384U, 56U);
    assert(known.stateBytes == 327680U && known.prefixOffset == 221696U);
    for (uint32_t ranks : {16U, 32U, 64U, 128U, 256U})
    for (uint32_t experts : {128U, 384U, 1024U}) {
        if (experts % ranks) continue;
        for (uint32_t b : {1U, 2U, 4U, 8U, 16U, 24U, 32U, 64U, 511U, 512U, 1024U})
        for (uint32_t k : {1U, 8U, 32U}) {
            const auto countCores = b < 512U ? 56U : 28U;
            const bool countFits = Cb59Reference::DispatchServerDedupCountSendUbBytes(
                uint64_t(b) * k, experts, countCores) <= 194560U;
            assert(Cb59Replay::CountSendUbFits(uint64_t(b) * k, b, experts, 56U) == countFits);
            for (uint32_t h : {256U, 3584U, 7168U, 16000U, 32768U})
            for (auto mode : {EpServerDedupInputMode::Plain16, EpServerDedupInputMode::Quantize16,
                              EpServerDedupInputMode::PrequantizedFp8Packs}) {
                EpMemoryServerDedupLayout input{};
                input.inputMode = mode;
                input.capacityRows = uint64_t(ranks) * b * std::min(k, experts / ranks);
                input.data = DispatchServerDedupDataLayout(b, h, k, experts, mode);
                const bool archived = Cb59Reference::DispatchDedupPacketUbFits(input, b, k, experts, ranks, 56);
                const bool replay = Cb59Replay::PacketUbFits(input, b, k, experts, ranks, 56);
                assert(replay == archived);
                if (replay) ++accepted; else ++rejected;
#if DEEPEP_LL_DISPATCH_CB59
                // Both API modes use the original role split and UB layout.
                assert(DispatchDedupPacketUbFits(input, b, k, experts, ranks, 56, false) == archived);
                assert(DispatchDedupPacketUbFits(input, b, k, experts, ranks, 56, true) == archived);
#endif
                EpMemoryServerDedupLayout built;
                const auto result = BuildDispatchServerDedupLayout(ranks, b, h, k, experts, &built, mode, 56);
                if (result == 0) {
                    assert(built.data.sourceOffset == 2U * 1024U * 1024U);
                    assert(built.data.dispatchOffset == input.data.dispatchOffset);
                    assert(built.data.requiredBytes <= kEpServerDedupWindowUsableBytes);
                    assert(built.tokenTypeBytes == input.capacityRows * 4U);
                    assert(built.destinationIndexBytes == input.capacityRows * 12U);
                    assert(built.relayReadIndexBytes == input.capacityRows * k * 8U);
                    assert(built.sendCountsBytes == experts * 4U);
#if DEEPEP_LL_DISPATCH_CB59
                    assert(countFits);
                    assert(built.countStateBytes == Cb59Reference::DispatchServerDedupCountLayout(experts, 56).stateBytes);
#else
                    assert(built.countStateBytes == DispatchServerDedupCountLayout(experts, 56).stateBytes);
#endif
                }
                CombineDedup::Layout combine;
                if (CombineDedup::BuildLayout(ranks, b, h, k, experts, combine, 56)) {
                    assert(combine.capacityRows == input.capacityRows);
                    assert(combine.requiredWindowBytes <= kEpServerDedupWindowUsableBytes);
                    combineDigest = combineDigest * 31U + combine.requiredWindowBytes;
                }
            }
        }
    }
    for (uint32_t b : {1U, 2U, 4U, 8U, 16U, 24U, 32U}) {
        EpMemoryServerDedupLayout d;
        assert(BuildDispatchServerDedupLayout(16, b, 3584, 8, 384, &d,
            EpServerDedupInputMode::Quantize16, 56) == 0);
        assert(DispatchDedupPacketUbFits(d, b, 8, 384, 16, 56, false));
        CombineDedup::Layout c;
        assert(CombineDedup::BuildLayout(16, b, 3584, 8, 384, c, 56));
    }
    assert(accepted && rejected);
    assert(!Cb59Replay::CountSendUbFits(8, 1, 384, 55));
    assert(!Cb59Replay::CountSendUbFits(8, 1, 384, 64));
    std::cout << combineDigest << '\n';
}
