// Unmodified function bodies from cb59bff99ecd39d27ef57788c5e6fd0009e4e636.
// CPU-only independent archive; no production header includes this fixture.
#pragma once
namespace DispatchDedup { namespace Cb59Reference {
constexpr uint32_t kEpServerDedupCountUbBytes=190U*1024U;
inline EpServerDedupCountLayout DispatchServerDedupCountLayout(uint32_t experts, uint32_t coreNum = kLlDefaultPlanCoreNum)
{
    EpServerDedupCountLayout layout{};
    const uint32_t recvCores = coreNum / 2U;
    const uint64_t matrixBytes = uint64_t(recvCores) * recvCores * kEpServerDedupCountRecordBytes;
    layout.cntOffset = kEpServerDedupControlPrefixBytes;
    layout.cumsumOffset = uint64_t(experts) * kEpServerDedupCountGmStride;
    layout.prefixOffset = layout.cumsumOffset + matrixBytes;
    layout.prefixStride = (uint64_t(experts) * sizeof(int32_t) + 31U) / 32U * 32U;
    layout.cumsumFlagOffset = layout.prefixOffset + recvCores * layout.prefixStride;
    const uint64_t bytes = layout.cntOffset + layout.cumsumFlagOffset + matrixBytes;
    layout.counterOffset = (bytes + 1023U) / 1024U * 1024U;
    layout.stateBytes = layout.counterOffset + 1024U;
    return layout;
}

inline uint64_t DispatchServerDedupCountSendUbBytes(uint64_t routes, uint32_t experts,
                                                                            uint32_t countCores)
{
    const uint64_t routeBytes = (routes * sizeof(int32_t) + 255U) / 256U * 256U;
    const uint32_t records = (experts + countCores - 1U) / countCores;
    const uint32_t alignedRecords = (records + 7U) / 8U * 8U;
    return 4U * routeBytes + uint64_t(alignedRecords) * (32U + sizeof(int32_t));
}

inline bool DispatchDedupPacketUbFits(const EpMemoryServerDedupLayout& layout, int64_t b, int64_t k,
                                      int64_t num_experts, int64_t world_size, uint32_t coreNum = kLlDefaultPlanCoreNum)
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
    // Prequantized scales occupy the two full payload input buffers (no alias to quantBuf).
    const uint64_t qBytes = align(b * k * 4, 32);
    const uint64_t scaleCount = layout.data.quantCount / 32U;
    const uint64_t quantWork = align(align(scaleCount, 32) * 4U + scaleCount * 2U, 32);
    const uint64_t scratchBytes = needQuantize ? std::max(qBytes, quantWork) : qBytes;
    const uint64_t sender = reserved + 2 * inputBytes + (needQuantize ? layout.data.packedPayloadBytes : 0U) +
                            align(b * k * 4, 256) + 2 * scratchBytes + qBytes + 2 * align(tokens * k * 4, 32) +
                            align(tokens * 4, 32) + align(tokens * k * 16, 32) + 64 + align(groups * 4, 32) +
                            layout.data.tokenStride;
    const uint64_t serverStateExtra = std::max<uint64_t>(64U, align(world_size / kLlRanksPerGroup * 4U, 32U)) - 64U;
    const uint64_t receiver = align(num_experts * 4, 32) + 2 * align(((layout.capacityRows + coreNum / 2U - 1U) / (coreNum / 2U)) * 4, 32) +
                              16 * layout.data.blockCount * 32 + 256 +
                              16 * align((layout.data.blockCount + 7) / 8, 32) * 4 + 256 + 16 * 36 +
                              layout.data.tokenStride + 64;
    return layout.data.blockCount > 0 && layout.data.blockCount <= UINT8_MAX &&
           sender + serverStateExtra <= kEpServerDedupCountUbBytes && receiver <= kEpServerDedupCountUbBytes;
}
} }
