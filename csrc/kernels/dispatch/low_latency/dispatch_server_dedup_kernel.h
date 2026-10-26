// Copyright (c) 2026, Lu Lu
// Modified by ryan_li 2026

#ifndef DISPATCH_SERVER_DEDUP_KERNEL_H
#define DISPATCH_SERVER_DEDUP_KERNEL_H

#include "common/ll_stage_timer_device.h"
#include "common/ll_expert_timer_device.h"
#include "common/ll_reference_profile_device.h"
#include "device/dispatch_server_dedup_magic.h"
#include "device/dispatch_server_dedup_counts.h"
#include "device/dispatch_server_dedup_send.h"
#include "device/dispatch_server_dedup_recv.h"

namespace DispatchDedup {
template <class XType, bool WireFp8, bool E4M3, bool Prequantized = false, uint32_t Group = 16U>
class DispatchMemoryServerDedupKernel {
public:
    __aicore__ inline void Init(GM_ADDR commArgs, GM_ADDR x, GM_ADDR expertIds, GM_ADDR expertScales,
                                GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, GM_ADDR expertTokenNumsOut,
                                GM_ADDR sendCountsOut, GM_ADDR tokenTypeOut, GM_ADDR destinationIndexOut,
                                GM_ADDR relayReadIndexOut, GM_ADDR sourceMaskOut, GM_ADDR expertScalesOut, uint32_t bs,
                                uint32_t h, uint32_t topK, uint32_t experts, uint32_t expertTokenNumsType, TPipe* pipe,
                                GM_ADDR inputScales = nullptr, bool enableDedup = true,
                                LlStageTimer::DeviceTimer* timer = nullptr,
                                LlExpertTimer::DeviceTimer* expertTimer = nullptr,
                                LlReferenceProfile::DeviceProfile* profile = nullptr)
    {
        timer_ = timer;
        profile_ = profile;
        args_ = reinterpret_cast<const __gm__ DispatchDedup::CommArgs*>(commArgs);
        core_ = GetBlockIdx();
        sendCores_ = GetBlockNum() / 2U;
        pipe_ = pipe;
        if (timer_) timer_->Start(LlStageTimer::Magic);
        if (profile_) profile_->Mark<2>();
        const uint32_t magic = RotateDispatchServerDedupMagic(args_, core_, pipe);
        if (profile_) profile_->Mark<3>();
        if (timer_) timer_->End(LlStageTimer::Magic);
        if (timer_) timer_->Start(LlStageTimer::CountInit);
        counts_.Init(commArgs, expertIds, expertTokenNumsOut, sendCountsOut, bs, topK, experts, expertTokenNumsType,
                     magic, pipe, profile_);
        if (profile_) profile_->Mark<4>();
        if (timer_) timer_->End(LlStageTimer::CountInit);
        if (core_ < sendCores_)
            sender_.Init(args_, x, expertIds, expertScales, sourceMaskOut, expertScalesOut, bs, h, topK, experts, pipe,
                         magic, inputScales, enableDedup);
        else
            receiver_.Init(args_, expandXOut, dynamicScalesOut, tokenTypeOut, destinationIndexOut, relayReadIndexOut,
                           bs, h, topK, experts, pipe, magic, enableDedup, expertTimer);
        if (profile_) profile_->Mark<5>();
    }

    // 完整dispatch：所有核先发送计数，再按固定核分工并行执行Sender与Receiver。
    // Sender可与接收核的计数前缀准备重叠，不能把提取阶段耗时直接相加当作完整时延。
    __aicore__ inline void Process()
    {
        if (timer_) timer_->Start(LlStageTimer::CountSend);
        if (profile_) profile_->Mark<6>();
        counts_.SendCnt();
        if (profile_) profile_->Mark<7>();
        if (timer_) timer_->End(LlStageTimer::CountSend);

        if (core_ < sendCores_) {
            if (timer_) timer_->Start(LlStageTimer::Sender);
            if (profile_) profile_->Mark<8>();
            sender_.Process(counts_.ScratchPool(), counts_.ReservedBytes(), timer_, profile_);
            if (profile_) profile_->Mark<9>();
            if (timer_) timer_->End(LlStageTimer::Sender);

        } else {
            if (timer_) timer_->Start(LlStageTimer::CountReceive);
            if (profile_) profile_->Mark<10>();
            counts_.RecvCnt();
            if (profile_) profile_->Mark<11>();
            if (timer_) timer_->End(LlStageTimer::CountReceive);
            if (timer_) timer_->Start(LlStageTimer::CountPrefix);
            if (profile_) profile_->Mark<12>();
            counts_.CalcCumsum();
            if (profile_) profile_->Mark<13>();
            if (timer_) timer_->End(LlStageTimer::CountPrefix);
            // 接收目标地址依赖计数前缀；写回完成并排空后，Receiver才可重置pipe并消费GM。
            if (timer_) timer_->Start(LlStageTimer::CountPublish);
            if (profile_) profile_->Mark<14>();
            counts_.CompleteCumsum();
            if (profile_) profile_->Mark<15>();
            // CompleteCumsum may still be writing expert counts from its UB.
            // Retire that stage before receiver resets the pipe and reloads GM.
            if (profile_) profile_->Mark<16>();
            PipeBarrier<PIPE_ALL>();
            if (profile_) profile_->Mark<17>();
            if (timer_) timer_->End(LlStageTimer::CountPublish);
            if (timer_) timer_->Start(LlStageTimer::Receiver);
            if (profile_) profile_->Mark<18>();
            receiver_.Process(timer_, profile_);
            if (profile_) profile_->Mark<19>();
            if (timer_) timer_->End(LlStageTimer::Receiver);
        }

        if (timer_) timer_->Start(LlStageTimer::CountFinish);
        if (profile_) profile_->Mark<20>();
        counts_.Finish();
        if (profile_) profile_->Mark<21>();
        if (timer_) timer_->End(LlStageTimer::CountFinish);

        // No shared work follows local completion. Same-stream kernel completion
        // orders the next launch after all cores finish their writes and cleanup.
    }

private:
    LlStageTimer::DeviceTimer* timer_ = nullptr;
    LlReferenceProfile::DeviceProfile* profile_ = nullptr;
    ServerDedupCounts counts_;
    ServerDedupSender<XType, WireFp8, E4M3, Prequantized, Group> sender_;
    ServerDedupReceiver<XType, WireFp8, E4M3, Prequantized> receiver_;
    const __gm__ DispatchDedup::CommArgs* args_ = nullptr;
    TPipe* pipe_ = nullptr;
    uint32_t core_ = 0, sendCores_ = 0;
};
}  // namespace DispatchDedup
#endif
