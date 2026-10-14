// Copyright (c) 2026, Lu Lu
// Modified by ryan_li 2026

#ifndef DISPATCH_SERVER_DEDUP_KERNEL_H
#define DISPATCH_SERVER_DEDUP_KERNEL_H

#include "common/ll_stage_timer_device.h"
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
                                GM_ADDR inputScales = nullptr, bool enableDedup = true,
                                LlStageTimer::DeviceTimer* timer = nullptr)
    {
        timer_ = timer;
        args_ = reinterpret_cast<const __gm__ DispatchDedup::CommArgs*>(commArgs);
        core_ = GetBlockIdx();
        sendCores_ = GetBlockNum() / 2U;
        pipe_ = pipe;
        if (timer_) timer_->Start(LlStageTimer::Magic);
        const uint32_t magic = RotateDispatchServerDedupMagic(args_, core_, pipe);
        if (timer_) timer_->End(LlStageTimer::Magic);
        if (timer_) timer_->Start(LlStageTimer::CountInit);
        counts_.Init(commArgs, expertIds, expertTokenNumsOut, sendCountsOut, bs, topK, experts, expertTokenNumsType,
                     magic, pipe);
        if (timer_) timer_->End(LlStageTimer::CountInit);
        if (core_ < sendCores_)
            sender_.Init(args_, x, expertIds, expertScales, sourceMaskOut, expertScalesOut, bs, h, topK, experts, pipe,
                         magic, inputScales, enableDedup);
        else
            receiver_.Init(args_, expandXOut, dynamicScalesOut, tokenTypeOut, destinationIndexOut, relayReadIndexOut,
                           bs, h, topK, experts, pipe, magic, enableDedup);
    }

    // 完整dispatch：所有核先发送计数，再按固定核分工并行执行Sender与Receiver。
    // Sender可与接收核的计数前缀准备重叠，不能把提取阶段耗时直接相加当作完整时延。
    __aicore__ inline void Process()
    {
        if (timer_) timer_->Start(LlStageTimer::CountSend);
        counts_.SendCnt();
        if (timer_) timer_->End(LlStageTimer::CountSend);

        if (core_ < sendCores_) {
            if (timer_) timer_->Start(LlStageTimer::Sender);
            sender_.Process(counts_.ScratchPool(), counts_.ReservedBytes(), timer_);
            if (timer_) timer_->End(LlStageTimer::Sender);

        } else {
            if (timer_) timer_->Start(LlStageTimer::CountReceive);
            counts_.RecvCnt();
            if (timer_) timer_->End(LlStageTimer::CountReceive);
            if (timer_) timer_->Start(LlStageTimer::CountPrefix);
            counts_.CalcCumsum();
            if (timer_) timer_->End(LlStageTimer::CountPrefix);
            // 接收目标地址依赖计数前缀；写回完成并排空后，Receiver才可重置pipe并消费GM。
            if (timer_) timer_->Start(LlStageTimer::CountPublish);
            counts_.CompleteCumsum();
            // CompleteCumsum may still be writing expert counts from its UB.
            // Retire that stage before receiver resets the pipe and reloads GM.
            PipeBarrier<PIPE_ALL>();
            if (timer_) timer_->End(LlStageTimer::CountPublish);
            if (timer_) timer_->Start(LlStageTimer::Receiver);
            receiver_.Process(timer_);
            if (timer_) timer_->End(LlStageTimer::Receiver);
        }

        if (timer_) timer_->Start(LlStageTimer::CountFinish);
        counts_.Finish();
        if (timer_) timer_->End(LlStageTimer::CountFinish);

        // No shared work follows local completion. Same-stream kernel completion
        // orders the next launch after all cores finish their writes and cleanup.
    }

private:
    LlStageTimer::DeviceTimer* timer_ = nullptr;
    ServerDedupCounts counts_;
    ServerDedupSender<XType, WireFp8, E4M3, Prequantized> sender_;
    ServerDedupReceiver<XType, WireFp8, E4M3, Prequantized> receiver_;
    const __gm__ DispatchDedup::CommArgs* args_ = nullptr;
    TPipe* pipe_ = nullptr;
    uint32_t core_ = 0, sendCores_ = 0;
};
}  // namespace DispatchDedup
#endif
