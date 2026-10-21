// Copyright (c) 2026, Lu Lu
// Modified by ryan_li 2026

#ifndef DISPATCH_DEDUP_EP_MEMORY_SERVER_DEDUP_COUNT_LAYOUT_H
#define DISPATCH_DEDUP_EP_MEMORY_SERVER_DEDUP_COUNT_LAYOUT_H

#include "comm_args.h"
#include "ll_core_config.h"
#include "ll_prefix_partition.h"
#include "ll_group_local_send.h"
#include "dispatch_dedup_types.h"

// Legacy count-probe threshold; production group-local source ownership uses LlGroupCountPlan.
#ifndef DISPATCH_SERVER_DEDUP_ALL_CORE_COUNT_BS_THRESHOLD
#define DISPATCH_SERVER_DEDUP_ALL_CORE_COUNT_BS_THRESHOLD 512U
#endif
#include "ep_memory_server_dedup_mode.h"

namespace DispatchDedup {
// Retained for older extracted count probes; the production dispatch no longer selects owners by BS.
constexpr uint32_t kEpServerDedupAllCoreCountBsThreshold = DISPATCH_SERVER_DEDUP_ALL_CORE_COUNT_BS_THRESHOLD;
constexpr uint32_t kEpServerDedupMaxExperts = 1024U;
constexpr uint32_t kEpServerDedupCountRecordBytes = 32U;
constexpr uint32_t kEpServerDedupCountGmStride = 512U;
constexpr uint32_t kEpServerDedupCountUbBytes = 230U * 1024U;
// Persistent per-AIV state lives only in window 0's prefix, never in the
// selected ping-pong bank. Keep identical count offsets in both windows.
constexpr uint64_t kEpServerDedupMagicStride = 512U;
constexpr uint64_t kEpServerDedupControlPrefixBytes = kLlMaxCoreNum * kEpServerDedupMagicStride;
static_assert(kEpServerDedupControlPrefixBytes == 36U * 1024U, "capacity for 72 isolated magic slots");
static_assert(kEpServerDedupCountGmStride % kEpServerDedupCountRecordBytes == 0U &&
                  kEpServerDedupControlPrefixBytes % kEpServerDedupCountGmStride == 0U,
              "GM count slots must be aligned whole records");
constexpr uint64_t kEpServerDedupPeerBytes = 8ULL * 1024U * 1024U * 1024U;
constexpr uint64_t kEpServerDedupWindowBytes = kEpServerDedupPeerBytes / 2U;
constexpr uint64_t kEpServerDedupControlBytes = 2ULL * 1024U * 1024U;
// Runtime owns the domain's final flag region. Leave the same tail unused in
// window 0 so both windows have identical capacity and shape-independent stride.
constexpr uint64_t kEpServerDedupWindowUsableBytes =
    kEpServerDedupWindowBytes - DispatchDedup::DISPATCH_DEDUP_FLAG_BUFF_BYTES;

#if DISPATCH_DEDUP_ASCENDC_AICORE_COMPILE
#define DISPATCH_DEDUP_EP_DEDUP_INLINE FORCE_INLINE_AICORE
#else
#define DISPATCH_DEDUP_EP_DEDUP_INLINE inline
#endif

// Matrix offsets are relative to cntOffset; cntOffset, counterOffset and
// stateBytes are relative to the selected window. Leading padding is retained.
struct EpServerDedupCountLayout {
    uint64_t cntOffset;
    uint64_t cumsumOffset;
    uint64_t cumsumFlagOffset;
    uint64_t prefixOffset;
    uint64_t prefixStride;
    uint64_t counterOffset;  // Retired counter: preserve its reserved layout, never read/write it.
    uint64_t stateBytes;
};

// Keep each rank's core split fixed for the communicator lifetime: prefix storage is not
// cleared and can overlap another split's control flags. All ranks must use the
// same communicator call order and serialize its kernels
// on one stream. Count publication (including zeros) protects two-call reuse.
DISPATCH_DEDUP_EP_DEDUP_INLINE uint64_t DispatchServerDedupWindowOffset(int64_t magic)
{
    return (uint64_t(magic) % 2U) * kEpServerDedupWindowBytes;
}

DISPATCH_DEDUP_EP_DEDUP_INLINE EpServerDedupCountLayout DispatchServerDedupCountLayout(uint32_t experts, uint32_t coreNum = kLlDefaultPlanCoreNum)
{
    (void)coreNum; // Retained for callers; the receive count is fixed by the build.
    EpServerDedupCountLayout layout{};
    const uint32_t recvCores = kLlRecvCoreNum;
    const uint64_t matrixBytes = uint64_t(recvCores) * recvCores * kEpServerDedupCountRecordBytes;
    layout.cntOffset = kEpServerDedupControlPrefixBytes;
    layout.cumsumOffset = uint64_t(experts) * kEpServerDedupCountGmStride;
    layout.prefixOffset = layout.cumsumOffset + matrixBytes;
    layout.prefixStride = (uint64_t(experts) * sizeof(int32_t) + 31U) / 32U * 32U;
    layout.cumsumFlagOffset = layout.prefixOffset + layout.prefixStride;
    const uint64_t readyBytes = uint64_t(kLlDispatchRecvCoreNum) * recvCores * kEpServerDedupCountRecordBytes;
    const uint64_t bytes = layout.cntOffset + layout.cumsumFlagOffset + readyBytes;
    layout.counterOffset = (bytes + 1023U) / 1024U * 1024U;
    layout.stateBytes = layout.counterOffset + 1024U;
    return layout;
}

// Peak sendCnt allocation: four route-sized UB buffers plus count records/histogram.
// Sender/receive scratch overlays those four buffers and the histogram after
// filtering; the records remain reserved until the asynchronous writes retire.
DISPATCH_DEDUP_EP_DEDUP_INLINE uint64_t DispatchServerDedupCountSendUbBytes(uint64_t routes, uint32_t experts,
                                                                            uint32_t countCores)
{
    const uint64_t routeBytes = (routes * sizeof(int32_t) + 255U) / 256U * 256U;
    const uint32_t records = (experts + countCores - 1U) / countCores;
    const uint32_t alignedRecords = (records + 7U) / 8U * 8U;
    return 4U * routeBytes + uint64_t(alignedRecords) * (32U + sizeof(int32_t));
}

// Peak while counting and retaining local descriptors, including two remote
// ranges around the local group. Actual count record lengths may be smaller.
DISPATCH_DEDUP_EP_DEDUP_INLINE uint64_t DispatchGroupLocalCountSendUbBytes(uint64_t routes,uint32_t experts,uint32_t ranks)
{
    const uint64_t routeBytes=(routes*4U+255U)/256U*256U;
    const uint64_t remote=(LlGroupCountMaxRecords(experts,ranks,false)+7U)/8U*8U;
    const uint64_t local=(LlGroupCountMaxRecords(experts,ranks,true)+7U)/8U*8U;
    const uint64_t remoteBytes=remote ? 4U*routeBytes+remote*36U : 0U;
    const uint64_t localBytes=local ? 6U*routeBytes+local*36U : 0U;
    return remoteBytes>localBytes ? remoteBytes : localBytes;
}

#undef DISPATCH_DEDUP_EP_DEDUP_INLINE
}  // namespace DispatchDedup
#endif
