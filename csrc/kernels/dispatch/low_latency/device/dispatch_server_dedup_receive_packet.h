// Copyright (c) 2026, Lu Lu
// Modified by candy_cloud_fly 2026

#ifndef DISPATCH_SERVER_DEDUP_RECEIVE_PACKET_H
#define DISPATCH_SERVER_DEDUP_RECEIVE_PACKET_H

#include "kernel_operator.h"
#include "../common/ep_memory_server_dedup_data_layout.h"
#include "../common/ep_memory_server_dedup_protocol.h"

namespace DispatchDedup {
using namespace AscendC;

// Immediate events use different HardEvents from the fixed data lane IDs.
template <HardEvent event>
__aicore__ inline void ServerDedupReceiveSync()
{
    SetFlag<event>(0U);
    WaitFlag<event>(0U);
}

// Init supplies physical offsets. The entire index segment fits in one payload
// block, so every block is contiguous and no per-entry Flag mapping is needed.
class ServerDedupPacketTail {
public:
    __aicore__ inline ServerDedupPacketTail(LocalTensor<uint8_t> wire, uint32_t headerOffset, uint32_t indexOffset)
        : header_(wire[headerOffset].ReinterpretCast<uint32_t>()), index_(wire[indexOffset].ReinterpretCast<uint32_t>())
    {
    }
    __aicore__ inline LocalTensor<uint32_t> Header()
    {
        return header_;
    }
    __aicore__ inline LocalTensor<uint32_t> ReadIndex()
    {
        return index_;
    }
    __aicore__ inline LocalTensor<uint32_t> Entry(uint32_t index)
    {
        return index_[(index + 1U) * 2U];
    }

private:
    LocalTensor<uint32_t> header_, index_;
};

__aicore__ inline void ReadServerDedupFlags(GM_ADDR slot, uint32_t blocks, LocalTensor<float> flags, bool uncached=false)
{
    GlobalTensor<float> input;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(slot + 480U));
    if (uncached) input.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    DataCopy(flags, input, DataCopyParams{uint16_t(blocks), 1U, 15U, 0U});
}

// Snapshot the local role independently of the header that forwarding changes.
__aicore__ inline uint32_t BuildServerDedupMetadata(LocalTensor<uint32_t> type, LocalTensor<uint32_t> destination,
                                                ServerDedupPacketTail tail, uint32_t slot)
{
    auto header = tail.Header();
    const uint32_t role = header.GetValue(3U);
    type.SetValue(0U, role);
    if (role == uint32_t(EpServerDedupTokenType::Terminal)) {
        destination.SetValue(0U, slot);
        destination.SetValue(1U, UINT32_MAX);
        destination.SetValue(2U, UINT32_MAX);
    } else {
        for (uint32_t i = 0U; i < 3U; ++i) destination.SetValue(i, header.GetValue(i));
    }
    return role;
}

__aicore__ inline void PrepareServerDedupForward(ServerDedupPacketTail tail)
{
    tail.Header().SetValue(3U, uint32_t(EpServerDedupTokenType::Terminal));
}
}  // namespace DispatchDedup
#endif
