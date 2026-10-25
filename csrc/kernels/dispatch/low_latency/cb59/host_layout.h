// Copyright (c) 2026, Lu Lu
#pragma once

#include <algorithm>
#include "count_layout.h"
#include "../common/ep_memory_server_dedup_input.h"
#include "../common/ep_memory_server_dedup_protocol.h"
#include "../common/ll_stream_receive.h"

namespace DispatchDedup {
namespace Cb59Replay {
// cb59 publishes counts from every AIV below B512, then overlays the same
// scratch arena with sender or receiver work. Source records remain reserved.
inline bool CountSendUbFits(uint64_t routes, uint32_t bs, uint32_t experts, uint32_t coreNum)
{
    if (!LlCoreCountValid(coreNum)) return false;
    const uint32_t countCores = bs < kEpServerDedupAllCoreCountBsThreshold ? coreNum : coreNum / 2U;
    return DispatchServerDedupCountSendUbBytes(routes, experts, countCores) <= kEpServerDedupCountUbBytes;
}

// Kept generic to avoid including the public Host layout back into this header.
// These are cb59's mode-specific peaks, including one full packet lane. Shared
// public allocation checks still validate pointer sizes and GM window bounds.
template<class Layout>
inline bool PacketUbFits(const Layout& layout, int64_t b, int64_t k, int64_t num_experts,
                         int64_t world_size, uint32_t coreNum)
{
    if (!LlCoreCountValid(coreNum)) return false;
    const auto align = [](uint64_t n, uint64_t a) { return (n + a - 1) / a * a; };
    const uint64_t cores = b < kEpServerDedupAllCoreCountBsThreshold ? coreNum : coreNum / 2U;
    const uint64_t reserved =
        b < kEpServerDedupAllCoreCountBsThreshold ? align((num_experts + cores - 1) / cores, 8) * 32 : 0;
    const uint64_t tokens = (b + coreNum / 2U - 1U) / (coreNum / 2U);
    const uint64_t groups = std::min<uint64_t>(world_size / kLlRanksPerGroup - 1, k / 2);
    const bool needQuantize = layout.inputMode == EpServerDedupInputMode::Quantize16;
    const uint64_t inputBytes = needQuantize ? layout.data.quantCount * 2U : layout.data.packedPayloadBytes;
    const uint64_t qBytes = align(b * k * 4, 32);
    const uint64_t scaleCount = layout.data.quantCount / 32U;
    const uint64_t quantWork = align(align(scaleCount, 32) * 4U + scaleCount * 2U, 32);
    const uint64_t scratchBytes = needQuantize ? std::max(qBytes, quantWork) : qBytes;
    const uint64_t sender = reserved + 2 * inputBytes + (needQuantize ? layout.data.packedPayloadBytes : 0U) +
                            align(b * k * 4, 256) + 2 * scratchBytes + qBytes + 2 * align(tokens * k * 4, 32) +
                            align(tokens * 4, 32) + align(tokens * k * 16, 32) + 64 + align(groups * 4, 32) +
                            layout.data.tokenStride;
    const uint64_t serverStateExtra = std::max<uint64_t>(64U, align(world_size / kLlRanksPerGroup * 4U, 32U)) - 64U;
    const uint64_t receiver = align(num_experts * 4, 32) +
                             2 * align(((layout.capacityRows + coreNum / 2U - 1U) / (coreNum / 2U)) * 4, 32) +
                             16 * layout.data.blockCount * 32 + 256 +
                             16 * align((layout.data.blockCount + 7) / 8, 32) * 4 + 256 + 16 * 36 +
                             layout.data.tokenStride + 64;
    return layout.data.blockCount > 0 && layout.data.blockCount <= UINT8_MAX &&
           sender + serverStateExtra <= kEpServerDedupCountUbBytes && receiver <= kEpServerDedupCountUbBytes;
}

// The streaming candidate keeps count-send records, its protocol buffers and
// receiver scratch live together. Admit only shapes with room for at least one
// unchanged packet lane; a rejected small shape never falls back at runtime.
template<class Layout>
inline bool StreamUbFits(const Layout& layout, int64_t b, int64_t k, int64_t num_experts,
                         int64_t world_size, uint32_t coreNum)
{
    if (world_size != 16 || b <= 0 || b > 32 || k <= 0 || k > kEpServerDedupMaxTopK ||
        num_experts < world_size || num_experts > kEpServerDedupMaxExperts || num_experts % world_size != 0 ||
        !PacketUbFits(layout, b, k, num_experts, world_size, coreNum)) return false;
    const uint32_t receivers = coreNum / 2U;
    const uint64_t reserved = LlStreamAlign((uint64_t(num_experts) + coreNum - 1U) / coreNum, 8U) * 32U;
    const uint64_t rows = LlStreamOwnedRowsUpperBound(uint32_t(num_experts), uint32_t(world_size),
                                                    uint32_t(b), uint32_t(k), receivers);
    const uint64_t fixed = reserved + LlStreamMaxCountBufferBytes(uint32_t(num_experts), uint32_t(world_size), receivers) +
                           LlStreamReceiverBufferBytes(rows, layout.data.blockCount);
    const uint64_t lane = layout.data.tokenStride + 64U;
    return fixed <= kEpServerDedupCountUbBytes && lane <= kEpServerDedupCountUbBytes - fixed;
}
}  // namespace Cb59Replay
}  // namespace DispatchDedup
