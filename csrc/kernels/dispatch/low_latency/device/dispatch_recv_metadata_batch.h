#pragma once

#include "kernel_operator.h"
#include "../common/ep_memory_server_dedup_protocol.h"
#include "../common/ll_recv_metadata_layout.h"

namespace DispatchDedup {
using namespace AscendC;

// The stable task index j names row begin+j even when pending entries are
// removed or FIRST_HOP packets are consumed first. These two output arrays have
// no dispatch/peer consumer: combine reads them after dispatch completion.
// RelayReadIndex remains on the existing per-packet path. This buffer survives
// until the caller's existing final PIPE_ALL, after Flush has submitted its DMA.
class ServerDedupRecvMetadataBatch {
public:
    __aicore__ inline void Prepare(const LlRecvMetadataBatchLayout& layout, TPipe* pipe)
    {
        layout_ = layout;
        if (!layout_.enabled) return;
        pipe->InitBuffer(storage_, layout_.bytes);
        type_ = storage_.Get<uint32_t>();
        destination_ = storage_.Get<uint8_t>()[layout_.destinationOffset].ReinterpretCast<uint32_t>();
    }

    __aicore__ inline bool Enabled() const { return layout_.enabled; }

    // Must run before PrepareServerDedupForward rewrites header[3]. The caller
    // supplies the already-snapshotted original role; header[0..2] is immutable.
    __aicore__ inline void Snapshot(uint32_t j, uint32_t role, uint32_t slot,
                                    LocalTensor<uint32_t> header)
    {
        type_.SetValue(j, role);
        if (role == uint32_t(EpServerDedupTokenType::Terminal)) {
            destination_.SetValue(j * 3U, slot);
            destination_.SetValue(j * 3U + 1U, UINT32_MAX);
            destination_.SetValue(j * 3U + 2U, UINT32_MAX);
        } else {
            destination_.SetValue(j * 3U, header.GetValue(0U));
            destination_.SetValue(j * 3U + 1U, header.GetValue(1U));
            destination_.SetValue(j * 3U + 2U, header.GetValue(2U));
        }
    }

    // No cross-core barrier, no pipeline drain and no early UB reuse. Exact
    // byte lengths avoid overwriting an adjacent core's rows at either end.
    // Caller must issue this only once, after all owned tasks are consumed.
    __aicore__ inline void Flush(GlobalTensor<uint32_t> tokenType,
                                 GlobalTensor<uint32_t> destinationIndex, uint32_t begin)
    {
        if (!layout_.enabled) return;
        SetFlag<HardEvent::S_MTE3>(7U);
        WaitFlag<HardEvent::S_MTE3>(7U);
        DataCopyPad(tokenType[begin], type_, {1U, layout_.rows * 4U, 0U, 0U, 0U});
        DataCopyPad(destinationIndex[uint64_t(begin) * 3U], destination_,
                    {1U, layout_.rows * 12U, 0U, 0U, 0U});
    }

private:
    LlRecvMetadataBatchLayout layout_{};
    TBuf<> storage_;
    LocalTensor<uint32_t> type_, destination_;
};
}  // namespace DispatchDedup
