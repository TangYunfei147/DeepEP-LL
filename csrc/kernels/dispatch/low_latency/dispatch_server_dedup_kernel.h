// Copyright (c) 2026, Lu Lu
// Modified by ryan_li 2026

#ifndef DISPATCH_SERVER_DEDUP_KERNEL_H
#define DISPATCH_SERVER_DEDUP_KERNEL_H

#include "common/dispatch_ll_debug.h"
#include "device/dispatch_server_dedup_magic.h"
#include "device/dispatch_server_dedup_counts.h"
#include "device/dispatch_server_dedup_send.h"
#include "device/dispatch_server_dedup_recv.h"

namespace DispatchDedup {
template <class XType, bool WireFp8, bool E4M3, bool Prequantized = false>
class DispatchMemoryServerDedupKernel {
public:
    __aicore__ inline void Init(GM_ADDR commArgs, GM_ADDR x, GM_ADDR expertIds, GM_ADDR expertScales,
                                GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, GM_ADDR expertTokenNumsOut,
                                GM_ADDR sendCountsOut, GM_ADDR tokenTypeOut, GM_ADDR destinationIndexOut,
                                GM_ADDR relayReadIndexOut, GM_ADDR sourceMaskOut, GM_ADDR expertScalesOut, uint32_t bs,
                                uint32_t h, uint32_t topK, uint32_t experts, uint32_t expertTokenNumsType, TPipe* pipe,
                                GM_ADDR inputScales = nullptr, bool enableDedup = true)
    {
        args_ = reinterpret_cast<const __gm__ DispatchDedup::CommArgs*>(commArgs);
        core_ = GetBlockIdx();
        sendCores_ = GetBlockNum() / 2U;
        pipe_ = pipe;
        LL_DLOG(LL_DISPATCH_DEBUG_STAGE, args_->rank, core_, "MAGIC_BEGIN", bs, h, topK, experts);
        const uint32_t magic = RotateDispatchServerDedupMagic(args_, core_, pipe);
        LL_DLOG(LL_DISPATCH_DEBUG_STAGE, args_->rank, core_, "MAGIC_DONE", magic, args_->rankSize, 0, 0);
        counts_.Init(commArgs, expertIds, expertTokenNumsOut, sendCountsOut, bs, topK, experts, expertTokenNumsType,
                     magic, pipe);
        LL_DLOG(LL_DISPATCH_DEBUG_STAGE, args_->rank, core_, "COUNTS_INIT_DONE", magic, 0, 0, 0);
        if (core_ < sendCores_)
            sender_.Init(args_, x, expertIds, expertScales, sourceMaskOut, expertScalesOut, bs, h, topK, experts, pipe,
                         magic, inputScales, enableDedup);
        else
            receiver_.Init(args_, expandXOut, dynamicScalesOut, tokenTypeOut, destinationIndexOut, relayReadIndexOut,
                           bs, h, topK, experts, pipe, magic, enableDedup);
        LL_DLOG(LL_DISPATCH_DEBUG_STAGE, args_->rank, core_, "INIT_DONE", magic, enableDedup, 0, 0);
    }

    // 完整dispatch：所有核先发送计数，再按固定核分工并行执行Sender与Receiver。
    // Sender可与接收核的计数前缀准备重叠，不能把提取阶段耗时直接相加当作完整时延。
    __aicore__ inline void Process()
    {
        LL_DLOG(LL_DISPATCH_DEBUG_STAGE, args_->rank, core_, "COUNT_SEND_BEGIN", 0, 0, 0, 0);
        counts_.SendCnt();
        LL_DLOG(LL_DISPATCH_DEBUG_STAGE, args_->rank, core_, "COUNT_SEND_DONE", 0, 0, 0, 0);

        if (core_ < sendCores_) {
            LL_DLOG(LL_DISPATCH_DEBUG_STAGE, args_->rank, core_, "PAYLOAD_SEND_BEGIN", 0, 0, 0, 0);
            sender_.Process(counts_.ScratchPool(), counts_.ReservedBytes());
            LL_DLOG(LL_DISPATCH_DEBUG_STAGE, args_->rank, core_, "PAYLOAD_SEND_DONE", 0, 0, 0, 0);

        } else {
            LL_DLOG(LL_DISPATCH_DEBUG_STAGE, args_->rank, core_, "COUNT_RECV_BEGIN", 0, 0, 0, 0);
            counts_.RecvCnt();
            LL_DLOG(LL_DISPATCH_DEBUG_STAGE, args_->rank, core_, "COUNT_RECV_DONE", 0, 0, 0, 0);
            LL_DLOG(LL_DISPATCH_DEBUG_STAGE, args_->rank, core_, "PREFIX_SUM_BEGIN", 0, 0, 0, 0);
            counts_.CalcCumsum();
            LL_DLOG(LL_DISPATCH_DEBUG_STAGE, args_->rank, core_, "PREFIX_SUM_DONE", 0, 0, 0, 0);
            // 接收目标地址依赖计数前缀；写回完成并排空后，Receiver才可重置pipe并消费GM。
            LL_DLOG(LL_DISPATCH_DEBUG_STAGE, args_->rank, core_, "PREFIX_READY_BEGIN", 0, 0, 0, 0);
            counts_.CompleteCumsum();
            LL_DLOG(LL_DISPATCH_DEBUG_STAGE, args_->rank, core_, "PREFIX_READY_DONE", 0, 0, 0, 0);
            // CompleteCumsum may still be writing expert counts from its UB.
            // Retire that stage before receiver resets the pipe and reloads GM.
            LL_DLOG(LL_DISPATCH_DEBUG_STAGE, args_->rank, core_, "PREFIX_DRAIN_BEGIN", 0, 0, 0, 0);
            PipeBarrier<PIPE_ALL>();
            LL_DLOG(LL_DISPATCH_DEBUG_STAGE, args_->rank, core_, "PREFIX_DRAIN_DONE", 0, 0, 0, 0);
            LL_DLOG(LL_DISPATCH_DEBUG_STAGE, args_->rank, core_, "PAYLOAD_RECV_BEGIN", 0, 0, 0, 0);
            receiver_.Process();
            LL_DLOG(LL_DISPATCH_DEBUG_STAGE, args_->rank, core_, "PAYLOAD_RECV_DONE", 0, 0, 0, 0);
        }

        LL_DLOG(LL_DISPATCH_DEBUG_STAGE, args_->rank, core_, "FINISH_BEGIN", 0, 0, 0, 0);
        counts_.Finish();
        LL_DLOG(LL_DISPATCH_DEBUG_STAGE, args_->rank, core_, "FINISH_DONE", 0, 0, 0, 0);

        // No shared work follows local completion. Same-stream kernel completion
        // orders the next launch after all cores finish their writes and cleanup.
    }

private:
    ServerDedupCounts counts_;
    ServerDedupSender<XType, WireFp8, E4M3, Prequantized> sender_;
    ServerDedupReceiver<XType, WireFp8, E4M3, Prequantized> receiver_;
    const __gm__ DispatchDedup::CommArgs* args_ = nullptr;
    TPipe* pipe_ = nullptr;
    uint32_t core_ = 0, sendCores_ = 0;
};
}  // namespace DispatchDedup
#endif
