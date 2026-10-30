// Copyright (c) 2026, Lu Lu
// Modified by ryan_li 2026

#ifndef DISPATCH_SERVER_DEDUP_KERNEL_H
#define DISPATCH_SERVER_DEDUP_KERNEL_H
#include "common/ll_reference_profile_device.h"

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
                                LlReferenceProfile::DeviceProfile* profile = nullptr)
    {
        profile_ = profile;
        enableDedup_=enableDedup;
        args_ = reinterpret_cast<const __gm__ DispatchDedup::CommArgs*>(commArgs);
        core_ = GetBlockIdx();
        sendCores_ = kLlSendCoreNum;
        pipe_ = pipe;
        constexpr auto mode=Prequantized ? EpServerDedupInputMode::PrequantizedFp8Packs :
            (WireFp8 ? EpServerDedupInputMode::Quantize16 : EpServerDedupInputMode::Plain16);
        overlap_=LlMakeOverlapLayout(DispatchServerDedupDataLayout(bs,h,topK,experts,mode),
                                    bs,topK,experts,args_->rankSize,args_->rank,core_,mode,enableDedup);
        counts_.ConfigureOverlap(overlap_);
        sender_.ConfigureOverlap(overlap_);
        receiver_.ConfigureOverlap(overlap_);

        if(profile_) profile_->Mark<2>();
        const uint32_t magic = RotateDispatchServerDedupMagic(args_, core_, pipe);
        if(profile_) profile_->Mark<3>();

        counts_.Init(commArgs, expertIds, expertTokenNumsOut, sendCountsOut, bs, topK, experts, expertTokenNumsType,
                     magic, pipe, profile, enableDedup);
        if(profile_) profile_->Mark<4>();

        sender_.Init(args_, x, expertIds, expertScales, sourceMaskOut, expertScalesOut, bs, h, topK, experts, pipe,
                     magic, inputScales, enableDedup);
        receiver_.Init(args_, expandXOut, dynamicScalesOut, tokenTypeOut, destinationIndexOut, relayReadIndexOut,
                       bs, h, topK, experts, pipe, magic, enableDedup);
        if(profile_) profile_->Mark<5>();
    }

#ifdef DEEPEP_LL_REFERENCE_PROFILE
    __aicore__ inline void ExportPacketTrace() { receiver_.ExportPacketTrace(profile_); }
#endif

    // Bisheng does not inherit __aicore__ on a plain lambda's operator().
    // Keep the callback and the body explicitly device-callable; no std::function
    // or indirect function pointer is needed for the templated sender callback.
    struct LocalProgress {
        DispatchMemoryServerDedupKernel* kernel;
        bool* prepared;
        __aicore__ inline void operator()() { kernel->AdvanceLocal(*prepared); }
    };
    __aicore__ inline void AdvanceLocal(bool& prepared)
    {
        if (!prepared) {
            if (!counts_.TryRecvCnt()) return;
            counts_.CalcCumsum();
            counts_.CompleteCumsum();
            receiver_.SetPrefix(counts_.Prefix());
            receiver_.Prepare(profile_);
            prepared=true;
        }
        receiver_.TryBatch();
    }

    // S remote submission overlaps receive; R cooperatively advances local TX/counts/RX.
    // Sender可与接收核的计数前缀准备重叠，不能把提取阶段耗时直接相加当作完整时延。
    __aicore__ inline void Process()
    {
        counts_.SendCnt();

        if (!enableDedup_) {
            if (core_<sendCores_) {
                sender_.Process(counts_.ScratchPool(),counts_.ReservedBytes(),profile_);
                sender_.Retire();
                counts_.Finish();
                return; // S owns all source routes and never waits for receive prefix.
            }
            counts_.RecvCnt();
            counts_.CalcCumsum();
            counts_.CompleteCumsum();
            if (overlap_.enabled) receiver_.SetPrefix(counts_.Prefix());
            receiver_.Process(profile_);
            counts_.Finish();
            return;
        }
        if (overlap_.Cooperative() && core_>=sendCores_) {
            counts_.PrepareRecvCnt();
            bool prepared=false;
            LocalProgress cooperate{this,&prepared};
            const uint32_t tasks=counts_.LocalTaskCount();
            if (tasks) sender_.ProcessGroupLocal(counts_.LocalSendScratchPool(),counts_.ReservedBytes(),
                counts_.LocalRouteIndices(),counts_.LocalExpertQ(),tasks,profile_,cooperate);
            while (!prepared) cooperate();
            receiver_.Run();
        } else {
            if (core_<sendCores_) {
                sender_.Process(counts_.ScratchPool(),counts_.ReservedBytes(),profile_);
            } else {
                // Stage3 publishes prefix before local TX. Stage1/2 retain
                // the original local-send -> count/prefix order.
                if (overlap_.Async()) { counts_.RecvCnt(); counts_.CalcCumsum(); }
                const uint32_t tasks=counts_.LocalTaskCount();
                if (tasks) sender_.ProcessGroupLocal(counts_.LocalSendScratchPool(),counts_.ReservedBytes(),
                    counts_.LocalRouteIndices(),counts_.LocalExpertQ(),tasks,profile_);
                if (!overlap_.Async()) { counts_.RecvCnt(); counts_.CalcCumsum(); }
            }
            counts_.CompleteCumsum();
            if (!overlap_.Async()) {
                if(profile_) profile_->Mark<16>();
                PipeBarrier<PIPE_ALL>();
                if(profile_) profile_->Mark<17>();
            }
            if (overlap_.enabled) receiver_.SetPrefix(counts_.Prefix());
            receiver_.Process(profile_);
        }
        {
        sender_.Retire();
        counts_.Finish(); }

        // No shared work follows local completion. Same-stream kernel completion
        // orders the next launch after all cores finish their writes and cleanup.
    }

private:
    LlReferenceProfile::DeviceProfile* profile_ = nullptr;
    ServerDedupCounts counts_;
    ServerDedupSender<XType, WireFp8, E4M3, Prequantized> sender_;
    ServerDedupReceiver<XType, WireFp8, E4M3, Prequantized> receiver_;
    const __gm__ DispatchDedup::CommArgs* args_ = nullptr;
    TPipe* pipe_ = nullptr;
    LlOverlapLayout overlap_{};
    uint32_t core_ = 0, sendCores_ = 0;
    bool enableDedup_=true;
};
}  // namespace DispatchDedup
#endif
