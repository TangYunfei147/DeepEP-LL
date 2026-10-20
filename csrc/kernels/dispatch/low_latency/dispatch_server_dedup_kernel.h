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


        pipe_ = pipe;
        const uint32_t magic = RotateDispatchServerDedupMagic(args_, GetBlockIdx(), pipe);
        counts_.Init(commArgs, expertIds, expertTokenNumsOut, sendCountsOut, bs, topK, experts, expertTokenNumsType,
                     magic, pipe, timer);
        sender_.Init(args_, x, expertIds, expertScales, sourceMaskOut, expertScalesOut, bs, h, topK, experts, pipe,
                         magic, inputScales, enableDedup);
        receiver_.Init(args_, expandXOut, dynamicScalesOut, tokenTypeOut, destinationIndexOut, relayReadIndexOut,
                       bs, h, topK, experts, pipe, magic, enableDedup, expertTimer);
    }

    // 全部56核发送，发送阶段同步后全部56核准备前缀并接收。
    // No initial sender/receiver split or group-local sender handoff.
    __aicore__ inline void Process()
    {
        counts_.SendCnt();

        sender_.Process(counts_.ScratchPool(), counts_.ReservedBytes(), timer_);
        // True phase boundary: all 56 local AIVs retire sends before any enters
        // count receive or packet receive. Empty send shards still participate.
        { LL_SEG_SCOPE(SEND_DRAIN);
          PipeBarrier<PIPE_ALL>();
          SyncAll<true>();
        }
        counts_.PrepareReceive();
        { LL_SEG_SCOPE(CAL_CUMSUM);
          counts_.RecvCnt();
          counts_.CalcCumsum();
        }

        // Every core joins receive after completing its own send or prefix work.
        // All 56 cores produce the prefix. Each of the 56 consumers
        // has its own publication row and disjoint output/cleanup ownership.
        counts_.CompleteCumsum();
        PipeBarrier<PIPE_ALL>();
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

};
}  // namespace DispatchDedup
#endif
