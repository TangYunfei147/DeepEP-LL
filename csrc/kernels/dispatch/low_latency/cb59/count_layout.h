// Copyright (c) 2026, Lu Lu
#pragma once

#include "../common/ep_memory_server_dedup_count_layout.h"

// Private control layout of cb59bff99ecd39d27ef57788c5e6fd0009e4e636.
// The public packet format and the 2 MiB payload base remain shared. Never use
// the current 56-consumer prefix layout or its larger UB arena in this replay.
namespace DispatchDedup {
namespace Cb59Replay {
constexpr uint32_t kEpServerDedupCountUbBytes = 190U * 1024U;
constexpr int64_t UB_SINGLE_DMA_SIZE_MAX = kEpServerDedupCountUbBytes;
constexpr int64_t UB_SINGLE_PING_PONG_ADD_SIZE_MAX = UB_SINGLE_DMA_SIZE_MAX / 2;
constexpr int64_t PING_PONG_SIZE = 2;

#if DISPATCH_DEDUP_ASCENDC_AICORE_COMPILE
#define DEEPEP_CB59_LAYOUT_INLINE FORCE_INLINE_AICORE
#else
#define DEEPEP_CB59_LAYOUT_INLINE inline
#endif

DEEPEP_CB59_LAYOUT_INLINE EpServerDedupCountLayout DispatchServerDedupCountLayout(
    uint32_t experts, uint32_t coreNum = kLlDefaultPlanCoreNum)
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

#undef DEEPEP_CB59_LAYOUT_INLINE
}  // namespace Cb59Replay
}  // namespace DispatchDedup
