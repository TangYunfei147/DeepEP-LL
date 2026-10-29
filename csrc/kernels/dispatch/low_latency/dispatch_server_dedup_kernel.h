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
template <class XType, bool WireFp8, bool E4M3, bool Prequantized = false, uint32_t Group = 16U>
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
        args_ = reinterpret_cast<const __gm__ DispatchDedup::CommArgs*>(commArgs);
        core_ = GetBlockIdx();
        pipe_ = pipe;

        if (profile_) profile_->Mark<2>();  // w2: magic_begin
        const uint32_t magic = RotateDispatchServerDedupMagic(args_, core_, pipe);
        if (profile_) profile_->Mark<3>();  // w3: magic_end

        commArgs_=commArgs;ids_=expertIds;expertNums_=expertTokenNumsOut;sendCounts_=sendCountsOut;
        bs_=bs;h_=h;k_=topK;experts_=experts;numsType_=expertTokenNumsType;magic_=magic;
        originalCtrl_=GetCtrlSpr<60,60>();SetCtrlSpr<60,60>(0);
        if(core_>=kLlMetadataCores && core_<ReceiveBegin(GetBlockNum())) {
            pipe_->InitBufPool(senderScratch_,kEpServerDedupCountUbBytes);
            sender_.Init(args_,x,expertIds,expertScales,sourceMaskOut,expertScalesOut,bs,h,topK,experts,pipe,magic,inputScales,enableDedup);
        } else if(core_>=ReceiveBegin(GetBlockNum())) {
            receiver_.Init(args_,expandXOut,dynamicScalesOut,tokenTypeOut,destinationIndexOut,relayReadIndexOut,
                bs,h,topK,experts,pipe,magic,enableDedup);
        }
        if (profile_ && core_>=kLlMetadataCores) profile_->Mark<5>();  // w5: role_init_end
    }

    // 元数据核、S与R独立运行；S不等count，R在prefix之前即可固定分工转发。
    // 各核及嵌套阶段并行，不能相加当作完整时延。
    __aicore__ inline void Process()
    {

        if(core_<kLlMetadataCores) {
            ServerDedupCounts counts;
            constexpr auto mode=Prequantized?EpServerDedupInputMode::PrequantizedFp8Packs:
                (WireFp8?EpServerDedupInputMode::Quantize16:EpServerDedupInputMode::Plain16);
            const auto data=DispatchServerDedupDataLayout(bs_,h_,k_,experts_,mode);
            counts.Init(commArgs_,ids_,expertNums_,sendCounts_,
                bs_,k_,experts_,numsType_,magic_,pipe_,profile_,
                args_->peerMems[args_->rank]+DispatchServerDedupWindowOffset(magic_)+data.relayOffset);
            if(profile_){profile_->Mark<4>();  // w4: counts_init_end
            profile_->Mark<5>();  // w5: role_init_end
            profile_->Mark<6>();}  // w6: send_count_begin
            counts.SendCnt();
            if(profile_){profile_->Mark<7>();  // w7: send_count_issue_end
            profile_->Mark<10>();}  // w10: recv_count_begin
            counts.RecvCnt();
            if(profile_){profile_->Mark<11>();  // w11: recv_count_end
            profile_->Mark<12>();}  // w12: cumsum_begin
            counts.CalcCumsum();
            if(profile_){profile_->Mark<13>();  // w13: cumsum_issue_end
            profile_->Mark<14>();}  // w14: complete_cumsum_begin
            counts.CompleteCumsum();
            if(profile_){profile_->Mark<15>();  // w15: complete_cumsum_issue_end
            profile_->Mark<16>();}  // w16: count_drain_begin
            counts.Finish();counts.RetireOwnScratch();
            if(profile_){profile_->Mark<17>();  // w17: count_drain_end_PIPE_ALL
            profile_->Value<37>((experts_+kLlMetadataCores-1U-core_)/kLlMetadataCores);}  // w37: receive_task_count
        } else if(core_<ReceiveBegin(GetBlockNum())) {
            if(profile_)profile_->Mark<8>();  // w8: sender_begin
            sender_.Process(senderScratch_,profile_);
            if(profile_)profile_->Mark<9>();  // w9: sender_end_after_MTE3_S
        } else {
            if(profile_)profile_->Mark<18>();  // w18: receiver_begin
            receiver_.Process(profile_);
            if(profile_)profile_->Mark<19>();  // w19: receiver_end
        }
        if(profile_)profile_->Mark<20>();  // w20: final_drain_begin
        PipeBarrier<PIPE_ALL>();SetCtrlSpr<60,60>(originalCtrl_);
        if(profile_)profile_->Mark<21>();  // w21: final_drain_end_PIPE_ALL
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if(core_>=ReceiveBegin(GetBlockNum()))receiver_.ExportPacketTrace(profile_);
#endif

        // No shared work follows local completion. Same-stream kernel completion
        // orders the next launch after all cores finish their writes and cleanup.
    }

private:
    LlReferenceProfile::DeviceProfile* profile_ = nullptr;
    TBufPool<TPosition::VECCALC,16> senderScratch_;
    GM_ADDR commArgs_=nullptr;
    GM_ADDR ids_=nullptr;
    GM_ADDR expertNums_=nullptr;
    GM_ADDR sendCounts_=nullptr;
    uint32_t bs_=0,h_=0,k_=0,experts_=0,numsType_=0,magic_=0;
    int64_t originalCtrl_=0;
    ServerDedupSender<XType, WireFp8, E4M3, Prequantized, Group> sender_;
    ServerDedupReceiver<XType, WireFp8, E4M3, Prequantized> receiver_;
    const __gm__ DispatchDedup::CommArgs* args_ = nullptr;
    TPipe* pipe_ = nullptr;
    uint32_t core_ = 0;
};
}  // namespace DispatchDedup
#endif
