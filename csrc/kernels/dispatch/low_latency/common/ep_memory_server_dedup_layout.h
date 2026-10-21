// Copyright (c) 2026, Lu Lu
// Modified by candy_cloud_fly 2026

#ifndef DISPATCH_DEDUP_EP_MEMORY_SERVER_DEDUP_LAYOUT_H
#define DISPATCH_DEDUP_EP_MEMORY_SERVER_DEDUP_LAYOUT_H

#include <cstddef>
#include <cstdint>
#include "ep_memory_server_dedup_data_layout.h"
#include "ll_core_config.h"
#include "ll_group_compact.h"
#include "../../../combine/low_latency/ep_memory_server_dedup_buffer_layout.h"

namespace DispatchDedup {
static_assert(kLlSharedPlanControlLimit == COMBINE_DEDUP_EP_SERVER_DEDUP_COMBINE_CONTROL_OFFSET,
              "Shared dispatch plan must stop before combine control");
static_assert(kLlSharedPlanReadyOffset + 32U <= kEpServerDedupMagicStride,
              "Shared ready flag must fit the persistent per-core slot");
struct EpMemoryServerDedupLayout {
    uint64_t globalBs = 0;
    uint64_t localExpertNum = 0;
    uint64_t capacityRows = 0;
    EpServerDedupInputMode inputMode = EpServerDedupInputMode::Plain16;
    uint64_t xBytes = 0;
    uint64_t inputScalesBytes = 0;
    uint64_t expertIdsBytes = 0;
    uint64_t weightBytes = 0;
    uint64_t expandXBytes = 0;
    uint64_t dynamicScalesBytes = 0;
    uint64_t expertTokenNumsBytes = 0;
    uint64_t sendCountsBytes = 0;
    uint64_t countStateBytes = 0;
    uint64_t tokenTypeBytes = 0;
    uint64_t destinationIndexBytes = 0;
    uint64_t relayReadIndexRowStrideBytes = 0;
    uint64_t relayReadIndexBytes = 0;
    uint64_t sourceMaskBytes = 0;
    EpServerDedupDataLayout data{};
};

// Caller-side allocation records only. The public bare-pointer API cannot infer these sizes.
struct EpServerDedupAllocation {
    const void* data = nullptr;
    uint64_t bytes = 0;
    uint64_t requiredBytes = 0;
    uint32_t alignment = 1;
    int32_t deviceId = -1;
};

// Host-only layout validation; the device entry uses the queried runtime core split.
int BuildDispatchServerDedupLayout(int64_t rankSize, int64_t bs, int64_t h, int64_t topK, int64_t moeExpertNum,
                                   EpMemoryServerDedupLayout* out, bool quantized = false, uint32_t coreNum = kLlDefaultPlanCoreNum);
int BuildDispatchServerDedupLayout(int64_t rankSize, int64_t bs, int64_t h, int64_t topK, int64_t moeExpertNum,
                                   EpMemoryServerDedupLayout* out, EpServerDedupInputMode mode, uint32_t coreNum = kLlDefaultPlanCoreNum);
int ValidateDispatchServerDedupAllocations(const EpServerDedupAllocation* allocations, std::size_t count,
                                           int32_t deviceId);

}  // namespace DispatchDedup

#include <algorithm>
#include <limits>

#include "comm_args.h"
#include "ep_memory_server_dedup_protocol.h"
#include "ep_memory_server_dedup_count_layout.h"
#include "dispatch_dedup_types.h"

namespace DispatchDedup {
namespace {
bool MultiplyBytes(uint64_t lhs, uint64_t rhs, uint64_t* out)
{
    const uint64_t limit = std::numeric_limits<std::size_t>::max();
    if (lhs != 0 && rhs > limit / lhs) {
        return false;
    }
    *out = lhs * rhs;
    return true;
}

}  // namespace

inline int BuildDispatchServerDedupLayout(int64_t rankSize, int64_t bs, int64_t h, int64_t topK, int64_t moeExpertNum,
                                          EpMemoryServerDedupLayout* out, EpServerDedupInputMode mode, uint32_t coreNum)
{
    if (out == nullptr || !LlCoreCountValid(coreNum)) {
        return DispatchDedup::DISPATCH_DEDUP_ERROR_PARA_CHECK_FAIL;
    }
    *out = EpMemoryServerDedupLayout{};
    if ((mode != EpServerDedupInputMode::Plain16 && mode != EpServerDedupInputMode::Quantize16 &&
         mode != EpServerDedupInputMode::PrequantizedFp8Packs) ||
        rankSize < 16 || rankSize > DispatchDedup::DISPATCH_DEDUP_MAX_RANK_SIZE ||
        rankSize % kLlRanksPerGroup != 0 || bs <= 0 || h <= 0 || h > UINT32_MAX || topK <= 0 ||
        topK > kEpServerDedupMaxTopK || moeExpertNum <= 0 || moeExpertNum > kEpServerDedupMaxExperts ||
        topK > moeExpertNum || moeExpertNum % rankSize != 0 || bs > UINT32_MAX / topK) {
        return DispatchDedup::DISPATCH_DEDUP_ERROR_PARA_CHECK_FAIL;
    }

    if (DispatchGroupLocalCountSendUbBytes(uint64_t(bs)*topK,uint32_t(moeExpertNum),uint32_t(rankSize))>
        kEpServerDedupCountUbBytes) return DispatchDedup::DISPATCH_DEDUP_ERROR_PARA_CHECK_FAIL;

    const bool wireFp8 = mode != EpServerDedupInputMode::Plain16;
    const bool prequantized = mode == EpServerDedupInputMode::PrequantizedFp8Packs;
    EpMemoryServerDedupLayout next{};
    next.inputMode = mode;
    next.globalBs = static_cast<uint64_t>(rankSize * bs);
    next.localExpertNum = static_cast<uint64_t>(moeExpertNum / rankSize);
    next.countStateBytes = DispatchServerDedupCountLayout(uint32_t(moeExpertNum), coreNum).stateBytes;
    next.data = DispatchServerDedupDataLayout(uint32_t(bs), uint32_t(h), uint32_t(topK), uint32_t(moeExpertNum), mode);
    if (uint64_t(bs)*topK>=65536U ||
        next.countStateBytes+LlMakeSharedGroupPlan(uint32_t(bs),uint32_t(topK),
            uint32_t(rankSize)/kLlRanksPerGroup,kLlSendCoreNum).gmBytes>kLlSharedPlanControlLimit ||
        next.data.requiredBytes > kEpServerDedupWindowUsableBytes) {
        return DispatchDedup::DISPATCH_DEDUP_ERROR_PARA_CHECK_FAIL;
    }
    next.relayReadIndexRowStrideBytes = static_cast<uint64_t>(topK) * 8U;
    uint64_t rowBytes = 0;
    uint64_t routeCount = 0;
    if (!MultiplyBytes(next.globalBs, std::min<uint64_t>(topK, next.localExpertNum), &next.capacityRows) ||
        !MultiplyBytes(h, prequantized ? 1U : 2U, &rowBytes) || !MultiplyBytes(bs, rowBytes, &next.xBytes) ||
        !MultiplyBytes(bs, prequantized ? next.data.scaleBytes : 0U, &next.inputScalesBytes) ||
        !MultiplyBytes(bs, topK, &routeCount) || !MultiplyBytes(routeCount, sizeof(int32_t), &next.expertIdsBytes) ||
        !MultiplyBytes(routeCount, sizeof(float), &next.weightBytes) ||
        !MultiplyBytes(next.capacityRows, wireFp8 ? uint64_t(h) : rowBytes, &next.expandXBytes) ||
        !MultiplyBytes(next.capacityRows, next.data.scaleBytes, &next.dynamicScalesBytes) ||
        !MultiplyBytes(next.localExpertNum, sizeof(int64_t), &next.expertTokenNumsBytes) ||
        !MultiplyBytes(moeExpertNum, sizeof(int32_t), &next.sendCountsBytes) ||
        !MultiplyBytes(next.capacityRows, sizeof(uint32_t), &next.tokenTypeBytes) ||
        !MultiplyBytes(next.capacityRows, 3U * sizeof(uint32_t), &next.destinationIndexBytes) ||
        !MultiplyBytes(bs, sizeof(uint32_t), &next.sourceMaskBytes) ||
        // Sparse first-hop indices use the full Dispatch output-row capacity.
        !MultiplyBytes(next.capacityRows, next.relayReadIndexRowStrideBytes, &next.relayReadIndexBytes)) {
        return DispatchDedup::DISPATCH_DEDUP_ERROR_PARA_CHECK_FAIL;
    }
    *out = next;
    return DispatchDedup::DISPATCH_DEDUP_SUCCESS;
}

inline int BuildDispatchServerDedupLayout(int64_t rankSize, int64_t bs, int64_t h, int64_t topK, int64_t moeExpertNum,
                                          EpMemoryServerDedupLayout* out, bool quantized, uint32_t coreNum)
{
    return BuildDispatchServerDedupLayout(
        rankSize, bs, h, topK, moeExpertNum, out,
        quantized ? EpServerDedupInputMode::Quantize16 : EpServerDedupInputMode::Plain16, coreNum);
}

inline int ValidateDispatchServerDedupAllocations(const EpServerDedupAllocation* allocations, std::size_t count,
                                                  int32_t deviceId)
{
    if (allocations == nullptr || count == 0 || deviceId < 0) {
        return DispatchDedup::DISPATCH_DEDUP_ERROR_PARA_CHECK_FAIL;
    }
    for (std::size_t i = 0; i < count; ++i) {
        const auto& allocation = allocations[i];
        const uintptr_t begin = reinterpret_cast<uintptr_t>(allocation.data);
        if (allocation.data == nullptr || allocation.requiredBytes == 0 ||
            allocation.bytes < allocation.requiredBytes || allocation.deviceId != deviceId ||
            allocation.alignment == 0 || (allocation.alignment & (allocation.alignment - 1U)) != 0 ||
            begin % allocation.alignment != 0 || allocation.bytes > std::numeric_limits<uintptr_t>::max() - begin) {
            return DispatchDedup::DISPATCH_DEDUP_ERROR_PARA_CHECK_FAIL;
        }
        const uintptr_t end = begin + allocation.bytes;
        for (std::size_t j = 0; j < i; ++j) {
            const uintptr_t otherBegin = reinterpret_cast<uintptr_t>(allocations[j].data);
            const uintptr_t otherEnd = otherBegin + allocations[j].bytes;
            if (begin < otherEnd && otherBegin < end) {
                return DispatchDedup::DISPATCH_DEDUP_ERROR_PARA_CHECK_FAIL;
            }
        }
    }
    return DispatchDedup::DISPATCH_DEDUP_SUCCESS;
}

}  // namespace DispatchDedup

namespace DispatchDedup {
// Matches each input mode, including quantization scratch or packed scale staging.
inline bool DispatchDedupPacketUbFits(const EpMemoryServerDedupLayout& layout, int64_t b, int64_t k,
                                      int64_t num_experts, int64_t world_size, uint32_t coreNum = kLlDefaultPlanCoreNum)
{
    if (!LlCoreCountValid(coreNum)) return false;
    const auto align = [](uint64_t n, uint64_t a) { return (n + a - 1) / a * a; };
    const uint64_t reserved=align(LlGroupCountMaxRecords(uint32_t(num_experts),uint32_t(world_size),false),8U)*32U;
    const uint64_t localReserved=align(LlGroupCountMaxRecords(uint32_t(num_experts),uint32_t(world_size),true),8U)*32U+
                                 2U*align(b*k*4U,256U);
    const uint64_t tokens = (b + kLlSendCoreNum - 1U) / kLlSendCoreNum;
    const uint64_t groups = std::min<uint64_t>(world_size / kLlRanksPerGroup - 1, k / 2);
    const bool needQuantize = layout.inputMode == EpServerDedupInputMode::Quantize16;
    const uint64_t inputBytes = needQuantize ? layout.data.quantCount * 2U : layout.data.packedPayloadBytes;
    // Prequantized scales occupy the two full payload input buffers (no alias to quantBuf).
    const uint64_t qBytes = align(b * k * 4, 32);
    const uint64_t scaleCount = layout.data.quantCount / 32U;
    const uint64_t quantWork = align(align(scaleCount, 32) * 4U + scaleCount * 2U, 32);
    const uint64_t scratchBytes = needQuantize ? std::max(qBytes, quantWork) : qBytes;
    const uint64_t sender = reserved + 2 * inputBytes + (needQuantize ? layout.data.packedPayloadBytes : 0U) +
                            align(b * k * 4, 256) + 2 * scratchBytes + qBytes + qBytes + align(tokens * k * 4, 32) +
                            align(tokens * 4, 32) +
                            LlMakeSharedGroupPlan(uint32_t(b),uint32_t(k),uint32_t(world_size)/kLlRanksPerGroup,kLlSendCoreNum).ubBytes +
                            64 + align(groups * 4, 32) +
                            layout.data.tokenStride;
    const uint64_t serverStateExtra = std::max<uint64_t>(64U, align(world_size / kLlRanksPerGroup * 4U, 32U)) - 64U;
    const uint64_t localSender=localReserved+2U*inputBytes+align(b*k*4U,32U)+
                               (needQuantize ? quantWork+layout.data.packedPayloadBytes : 0U)+layout.data.tokenStride;
    const uint64_t receiver = align(num_experts * 4, 32) + 2 * align(((layout.capacityRows + kLlDispatchRecvCoreNum - 1U) / kLlDispatchRecvCoreNum) * 4, 32) +
                              16 * layout.data.blockCount * 32 + 256 +
                              16 * align((layout.data.blockCount + 7) / 8, 32) * 4 + 256 + 16 * 36 +
                              layout.data.tokenStride + 64;
    return layout.data.blockCount > 0 && layout.data.blockCount <= UINT8_MAX &&
           sender + serverStateExtra <= kEpServerDedupCountUbBytes &&
           localSender <= kEpServerDedupCountUbBytes && receiver <= kEpServerDedupCountUbBytes;
}
}  // namespace DispatchDedup

#endif  // DISPATCH_DEDUP_EP_MEMORY_SERVER_DEDUP_LAYOUT_H
