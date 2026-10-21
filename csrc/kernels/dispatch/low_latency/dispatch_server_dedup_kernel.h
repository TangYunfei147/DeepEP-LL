// Copyright (c) 2026, Lu Lu
// Modified by ryan_li 2026

#ifndef DISPATCH_SERVER_DEDUP_KERNEL_H
#define DISPATCH_SERVER_DEDUP_KERNEL_H

#include "common/ll_stage_timer_device.h"
#include "common/ll_expert_timer_device.h"
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
                                LlStageTimer::DeviceTimer* timer = nullptr,
                                LlExpertTimer::DeviceTimer* expertTimer = nullptr)
    {
        timer_ = timer;
        args_ = reinterpret_cast<const __gm__ DispatchDedup::CommArgs*>(commArgs);
        core_ = GetBlockIdx();
        sendCores_ = kLlSendCoreNum;
        pipe_ = pipe;
        constexpr auto mode=Prequantized ? EpServerDedupInputMode::PrequantizedFp8Packs :
            (WireFp8 ? EpServerDedupInputMode::Quantize16 : EpServerDedupInputMode::Plain16);
        overlap_=LlMakeOverlapLayout(DispatchServerDedupDataLayout(bs,h,topK,experts,mode),
                                    bs,topK,experts,args_->rankSize,args_->rank,core_,mode);
        counts_.ConfigureOverlap(overlap_);
        sender_.ConfigureOverlap(overlap_);
        receiver_.ConfigureOverlap(overlap_);
        const uint32_t magic = RotateDispatchServerDedupMagic(args_, core_, pipe);
        counts_.Init(commArgs, expertIds, expertTokenNumsOut, sendCountsOut, bs, topK, experts, expertTokenNumsType,
                     magic, pipe, timer);
        sender_.Init(args_, x, expertIds, expertScales, sourceMaskOut, expertScalesOut, bs, h, topK, experts, pipe,
                     magic, inputScales, enableDedup);
        receiver_.Init(args_, expandXOut, dynamicScalesOut, tokenTypeOut, destinationIndexOut, relayReadIndexOut,
                       bs, h, topK, experts, pipe, magic, enableDedup, expertTimer);
    }

    // 完整dispatch：S核跨组初发；R核本组初发后按原流程准备前缀；全部核接收。
    // Sender可与接收核的计数前缀准备重叠，不能把提取阶段耗时直接相加当作完整时延。
    __aicore__ inline void Process()
    {
        if (overlap_.enabled) {
            LL_SEG_COUNT(OVERLAP_PATH,1);
            LL_SEG_COUNT(SEND_LANES,overlap_.sendLanes);
            LL_SEG_COUNT(RECV_LANES,overlap_.recvLanes);
        } else { LL_SEG_COUNT(SERIAL_FALLBACK,1); }
        counts_.SendCnt();

        if (core_ < sendCores_) {
            sender_.Process(counts_.ScratchPool(), counts_.ReservedBytes(), timer_);

        } else {
            const uint32_t localTasks=counts_.LocalTaskCount();
            if (localTasks) sender_.ProcessGroupLocal(counts_.LocalSendScratchPool(),counts_.ReservedBytes(),
                                                       counts_.LocalRouteIndices(),counts_.LocalExpertQ(),localTasks,timer_);
            LL_SEG_SCOPE(CAL_CUMSUM);
            counts_.RecvCnt();
            counts_.CalcCumsum();
        }

        // Every core joins receive after completing its own send or prefix work.
        // Prefix producers remain the original R cores. Each of the 56 consumers
        // has its own publication row and disjoint output/cleanup ownership.
        counts_.CompleteCumsum();
        { LL_SEG_SCOPE(TRANSITION_DRAIN);
        PipeBarrier<PIPE_ALL>();
        }
        if (overlap_.enabled) receiver_.SetPrefix(counts_.Prefix());
        { LL_SEG_SCOPE(LOCAL_WINDOW_COPY);
        receiver_.Process(timer_);
        }

        { LL_SEG_SCOPE(FINAL_DRAIN);
        counts_.Finish();
        }

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
    LlOverlapLayout overlap_{};
    uint32_t core_ = 0, sendCores_ = 0;
};
}  // namespace DispatchDedup
#endif
