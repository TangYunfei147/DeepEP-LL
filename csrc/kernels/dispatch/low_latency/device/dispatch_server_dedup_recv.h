// Copyright (c) 2026, Lu Lu
// Modified by ryan_li 2026

#ifndef DISPATCH_SERVER_DEDUP_RECV_H
#define DISPATCH_SERVER_DEDUP_RECV_H
#include "../common/ll_stage_timer_device.h"
#include "../common/ll_expert_timer_device.h"

#include "kernel_operator.h"
#include "adv_api/reduce/sum.h"
#include "../common/ep_memory_server_dedup_protocol.h"
#include "../../../common/ep_packet_receive_layout.h"
#include "dispatch_server_dedup_receive_tasks.h"
#include "dispatch_server_dedup_receive_packet.h"
#include "dispatch_server_dedup_receive_queue.h"

namespace DispatchDedup {
using namespace AscendC;

// 中文审计：完整dispatch的接收、包消费和输出元数据路径；第2层提取核不调用本类。
// 审计接收任务/包格式时继续查看dispatch_server_dedup_receive_tasks.h和receive_packet.h。
template <class XType, bool WireFp8, bool E4M3, bool Prequantized = false>
class ServerDedupReceiver {
    static_assert(!Prequantized || WireFp8, "prequantized input requires FP8 wire");
    static constexpr EpServerDedupInputMode InputMode =
        Prequantized ? EpServerDedupInputMode::PrequantizedFp8Packs
                     : (WireFp8 ? EpServerDedupInputMode::Quantize16 : EpServerDedupInputMode::Plain16);

public:
    // 摘要：绑定本 rank 接收输出，确定接收核编号和本轮双窗口偏移。
    __aicore__ inline void Init(const __gm__ DispatchDedup::CommArgs* args, GM_ADDR expandX, GM_ADDR dynamicScales,
                                GM_ADDR tokenType, GM_ADDR destinationIndex, GM_ADDR relayReadIndex, uint32_t bs,
                                uint32_t h, uint32_t k, uint32_t experts, TPipe* pipe, int64_t magic = 0,
                                bool enableDedup = true, LlExpertTimer::DeviceTimer* expertTimer = nullptr)
    {
        args_ = args;
        expertTimer_ = expertTimer;
        coreNum_ = GetBlockNum();
        pipe_ = pipe;
        enableDedup_ = enableDedup;
        bs_ = bs;
        k_ = k;
        experts_ = experts;
        localExperts_ = experts / uint32_t(args_->rankSize);
        hiddenBytes_ = WireFp8 ? h : h * uint32_t(sizeof(XType));
        data_ = DispatchServerDedupDataLayout(bs, h, k, experts, InputMode, uint32_t(args_->rankSize));
        windowOffset_ = DispatchServerDedupWindowOffset(magic);
        data_.dispatchOffset += windowOffset_;
        expandX_.SetGlobalBuffer(expandX);
        dynamicScales_.SetGlobalBuffer(dynamicScales);
        tokenType_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(tokenType));
        destinationIndex_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(destinationIndex));
        relayReadIndex_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(relayReadIndex));
    }

    __aicore__ inline GM_ADDR QueueAddress() const {
        return args_->peerMems[args_->rank]+windowOffset_+data_.receiveQueueOffset;
    }
    __aicore__ inline uint32_t QueueClearBytes() const {
        return uint32_t(kLlReceiveHeaderBytes+data_.receiveClaimBytes);
    }
    __aicore__ inline GM_ADDR PositionAddress() const {
        return args_->peerMems[args_->rank]+windowOffset_+data_.receivePositionsOffset;
    }
    // 摘要：执行接收主流程：加载前缀、构建任务、批量探测、预取并消费就绪包，最后清理本核槽位标记。
    __aicore__ inline void Process(LlStageTimer::DeviceTimer* timer,uint32_t total)
    {
        timer_=timer;
        LL_SEG_SCOPE(FORMAT_OUTPUT);
        pipe_->Reset();
        if (!total) return;
        NativeReceiveQueue queue;
        queue.Init(reinterpret_cast<__gm__ uint32_t*>(QueueAddress()),total,coreNum_);
        if (!queue.Enter()) return; // Late/extra workers avoid all receive setup.
        const uint32_t batches=(total+kProbeBatch-1U)/kProbeBatch;
        const uint32_t seenBytes=(batches+31U)/32U*4U;
        pipe_->InitBuffer(seenBuf_,uint32_t(DispatchDedupAlign(seenBytes,32U)));
        auto seen=seenBuf_.Get<uint32_t>();
        Duplicate<uint32_t>(seen,0U,uint32_t(DispatchDedupAlign(seenBytes,32U))/4U);
        tasks_.Init(pipe_);
        GlobalTensor<uint32_t> positions;
        positions.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(PositionAddress()));
        const uint32_t blocks=uint32_t(data_.blockCount);
        const uint32_t flagsBytes=kProbeBatch*blocks*32U+256U;
        const uint32_t workBytes=kProbeBatch*uint32_t(DispatchDedupAlign((blocks+7U)/8U,32U))*4U+256U;
        pipe_->InitBuffer(flagsBuf_,flagsBytes);
        pipe_->InitBuffer(sumBuf_,kProbeBatch*32U);
        pipe_->InitBuffer(workBuf_,workBytes);
        pipe_->InitBuffer(zeroFlagsBuf_,blocks*32U);
        Duplicate(flagsBuf_.Get<float>(),0.0f,flagsBytes/4U);
        Duplicate(zeroFlagsBuf_.Get<float>(),0.0f,blocks*8U);
        ServerDedupReceiveSync<HardEvent::V_S>();
        ServerDedupReceiveSync<HardEvent::V_MTE2>();
        ServerDedupReceiveSync<HardEvent::V_MTE3>();
        const uint32_t fixed=uint32_t(DispatchDedupAlign(seenBytes,32U))+kProbeBatch*4U+
                             flagsBytes+workBytes+kProbeBatch*32U+blocks*32U;
        laneStride_=uint32_t(data_.tokenStride)+64U;
        const uint32_t available=(kEpServerDedupCountUbBytes-fixed)/laneStride_;
        lanes_=available<8U ? available : 8U;
        if (lanes_>total) lanes_=total;
        pipe_->InitBuffer(lanesBuf_,lanes_*laneStride_);
        for (uint32_t lane=0;lane<lanes_;++lane) SetFlag<HardEvent::MTE3_MTE2>(lane);
        uint32_t issued=0U,consumed=0U,batch=0U;
        while (queue.Next(batch)) {
            const uint32_t bit=uint32_t(1)<<(batch&31U),word=batch>>5U;
            if (seen.GetValue(word)&bit) continue;
            const uint32_t begin=batch*kProbeBatch;
            const uint32_t count=total-begin<kProbeBatch ? total-begin : kProbeBatch;
            // Protect old Scalar descriptor reads before this fixed tile is reused.
            ServerDedupReceiveSync<HardEvent::S_MTE2>();
            tasks_.Load(positions,begin,count);
            ServerDedupReceiveSync<HardEvent::MTE2_S>();
            ProbeResult probe=Probe(count);
            if (!probe.readyMask) continue; // No atomics for an empty probe.
            const uint32_t valid=LlReceiveValidMask(count);
            uint32_t observed=0U;
            const uint32_t owned=queue.Claim(batch,probe.readyMask,valid,observed);
            if (observed==valid) seen.SetValue(word,seen.GetValue(word)|bit);
            if (!owned) continue;
            uint32_t nextLoad=count;
            for (uint32_t i=count;i>0U;--i) {
                const uint32_t index=i-1U;
                if (!(owned&(uint32_t(1)<<index))) continue;
                while (issued-consumed<lanes_ && nextLoad) {
                    --nextLoad;
                    if (!(owned&(uint32_t(1)<<nextLoad))) continue;
                    IssueLoad(nextLoad,issued%lanes_,(probe.keepFlagsMask&(uint32_t(1)<<nextLoad))!=0U);
                    ++issued;
                }
                ConsumeLoaded(index,consumed%lanes_,(probe.keepFlagsMask&(uint32_t(1)<<index))!=0U);
                ++consumed;
            }
            // A constant zero buffer permits cleanup without a per-batch drain.
            // MTE3 orders each clear after this owner's output and relay writes.
            ClearFlags(owned);
        }
        for (uint32_t lane=0;lane<lanes_;++lane) WaitFlag<HardEvent::MTE3_MTE2>(lane);
        PipeBarrier<PIPE_ALL>();
    }

private:
    static constexpr uint32_t kProbeBatch = kLlReceiveBatch;
    struct ProbeResult {
        uint32_t readyMask = 0U;
        uint32_t keepFlagsMask = 0U;
    };
    // 摘要：将稳定任务编号 j 映射为本轮通信窗口中的实际 packet 槽地址。
    __aicore__ inline __gm__ uint8_t* Slot(uint32_t j)
    {
        return args_->peerMems[args_->rank] + data_.dispatchOffset + uint64_t(tasks_.Position(j)) * data_.tokenStride;
    }

    // 摘要：保存最多 16 项 pending 任务快照，检查每个包全部块的标记，返回就绪位图。
    __aicore__ inline ProbeResult Probe(uint32_t count)
    {
        LL_SEG_SCOPE(RECV_POLL);
        auto flags = flagsBuf_.Get<float>();
        auto sums = sumBuf_.Get<float>();
        auto work = workBuf_.Get<uint8_t>();
        const uint32_t blocks = uint32_t(data_.blockCount);
        for (uint32_t i = 0; i < count; ++i) {
            ReadServerDedupFlags(Slot(i), blocks, flags[i * blocks * 8U]);
        }
        ServerDedupReceiveSync<HardEvent::MTE2_V>();
        Sum<float>(sums, flags, work, SumParams{count, blocks * 8U, blocks * 8U});
        ServerDedupReceiveSync<HardEvent::V_S>();
        ProbeResult result;
        for (uint32_t i = 0; i < count; ++i) {
            const auto layout = PacketReceiveLayout::Decode(sums.GetValue(i), blocks);
            if (layout != PacketReceiveLayout::State::Pending) result.readyMask |= uint32_t(1) << i;
            if (layout == PacketReceiveLayout::State::KeepFlags) result.keepFlagsMask |= uint32_t(1) << i;
        }
        return result;
    }

    // 摘要：等待 lane 的旧输出完成后，将 packet 从 GM 预取到 UB，分别发布给 Scalar 和 MTE3 的依赖事件。
    __aicore__ inline void IssueLoad(uint32_t j, uint32_t lane, bool keepFlags)
    {
        LL_SEG_BEGIN(segBegin1, RECV_BUFFER_WAIT);
        WaitFlag<HardEvent::MTE3_MTE2>(lane);
        LL_SEG_END(segBegin1, RECV_BUFFER_WAIT);
        LL_SEG_BEGIN(segBegin2, RECV_INPUT);
        auto wire = lanesBuf_.Get<uint8_t>()[lane * laneStride_];
        GlobalTensor<uint8_t> source;
        source.SetGlobalBuffer(Slot(j));
        if (keepFlags) {
            DataCopy(wire, source, uint32_t(data_.tokenStride));
        } else {
            // Compact every 480B payload block, skipping the 32B GM Flag.
            // GM source stride is bytes; UB destination stride is 32B blocks.
            DataCopyPad(wire, source, {uint16_t(data_.blockCount), 480U, 32U, 0U, 0U}, {false, 0U, 0U, 0});
        }
        SetFlag<HardEvent::MTE2_MTE3>(lane);
        // Publish immediately after this load, before queuing later loads.
        SetFlag<HardEvent::MTE2_S>(lane);
        LL_SEG_END(segBegin2, RECV_INPUT);
    }

    // 摘要：消费已预取包，写出 token/scale 和路由元数据；FIRST_HOP 还向其余专家转发，再归还 lane。
    __aicore__ inline void ConsumeLoaded(uint32_t j, uint32_t lane, bool keepFlags)
    {
        LL_SEG_SCOPE(RECV_COPY);
        LL_SEG_COUNT(RECV_READY_TOKENS, 1U);
        LL_SEG_BEGIN(segBegin3, RECV_INPUT);
        WaitFlag<HardEvent::MTE2_S>(lane);
        LL_SEG_END(segBegin3, RECV_INPUT);
        LL_SEG_SCOPE(RECV_OUTPUT);
        uint32_t expert = 0;
        if (expertTimer_ && expertTimer_->Enabled()) {
            // Position encodes rotated source, local expert, q. Source rotation
            // does not change the local-expert remainder.
            expert = uint32_t(args_->rank)*localExperts_ + (tasks_.Position(j)/bs_)%localExperts_;
            expertTimer_->CountPacket(expert);
        }
        auto wire = lanesBuf_.Get<uint8_t>()[lane * laneStride_];
        const uint32_t row = tasks_.Begin() + j;
        const uint32_t headerOffset = uint32_t(keepFlags ? data_.headerWireOffset : data_.headerPayloadOffset);
        if (!enableDedup_ && !keepFlags) {
            // Compact offsets are logical payload offsets. Retain this lane
            // until all payload, scale and metadata output reads complete.
            LL_SEG_BEGIN(segBegin4, RECV_BUFFER_WAIT);
            WaitFlag<HardEvent::MTE2_MTE3>(lane);
            LL_SEG_END(segBegin4, RECV_BUFFER_WAIT);
            if (expertTimer_) expertTimer_->Start(expert, LlExpertTimer::OutputWrite);
            DataCopyPad(expandX_[uint64_t(row) * hiddenBytes_], wire, {1U, hiddenBytes_, 0U, 0U, 0U});
            if constexpr (WireFp8)
                DataCopyPad(dynamicScales_[uint64_t(row) * data_.scaleBytes], wire[uint32_t(data_.quantCount)],
                            {1U, uint32_t(data_.scaleBytes), 0U, 0U, 0U});
            if (expertTimer_) expertTimer_->End(expert, LlExpertTimer::OutputWrite);
            if (expertTimer_) expertTimer_->Start(expert, LlExpertTimer::Metadata);
            auto header = wire[headerOffset].template ReinterpretCast<uint32_t>();
            if (expertTimer_) expertTimer_->End(expert, LlExpertTimer::Metadata);
            if (expertTimer_) expertTimer_->Start(expert, LlExpertTimer::OutputWrite);
            DataCopyPad(destinationIndex_[uint64_t(row) * 3U], header, {1U, 12U, 0U, 0U, 0U});
            SetFlag<HardEvent::MTE3_MTE2>(lane);
            if (expertTimer_) expertTimer_->End(expert, LlExpertTimer::OutputWrite);
            return;
        }
        if (expertTimer_) expertTimer_->Start(expert, LlExpertTimer::Metadata);
        auto type = wire[uint32_t(data_.tokenStride)].template ReinterpretCast<uint32_t>();
        auto destination = wire[uint32_t(data_.tokenStride) + 32U].template ReinterpretCast<uint32_t>();
        ServerDedupPacketTail tail(wire, headerOffset,
                                   uint32_t(keepFlags ? data_.relayReadIndexWireOffset : data_.readIndexPayloadOffset));
        const bool firstHop = enableDedup_ && tail.Header().GetValue(3U) == uint32_t(EpServerDedupTokenType::FirstHop);
        if (enableDedup_)
            BuildServerDedupMetadata(type, destination, tail, tasks_.Position(j));
        else
            for (uint32_t i = 0U; i < 3U; ++i) destination.SetValue(i, tail.Header().GetValue(i));
        if (expertTimer_) expertTimer_->End(expert, LlExpertTimer::Metadata);
        // FIRST_HOP preparation is shared work: retain only the core stage timer.
        if (firstHop) {
            PrepareServerDedupForward(tail);
            // Forwarded packets only need compact payload. The hint remains
            // independent of the application-level token type in the header.
            wire.template ReinterpretCast<float>().SetValue(PacketReceiveLayout::kMarkerOffset / 4U,
                                                            PacketReceiveLayout::kCompact);
        }
        LL_SEG_BEGIN(segBegin7, RECV_BUFFER_WAIT);
        ServerDedupReceiveSync<HardEvent::S_MTE3>();
        WaitFlag<HardEvent::MTE2_MTE3>(lane);
        LL_SEG_END(segBegin7, RECV_BUFFER_WAIT);
        if (expertTimer_) expertTimer_->Start(expert, LlExpertTimer::OutputWrite);
        if (keepFlags) {
            WriteServerDedupPayloadRange(expandX_[uint64_t(row) * hiddenBytes_], wire, 0U, hiddenBytes_);
            if constexpr (WireFp8)
                WriteServerDedupPayloadRange(dynamicScales_[uint64_t(row) * data_.scaleBytes], wire,
                                             uint32_t(data_.quantCount), uint32_t(data_.scaleBytes));
        } else {
            DataCopyPad(expandX_[uint64_t(row) * hiddenBytes_], wire, {1U, hiddenBytes_, 0U, 0U, 0U});
            if constexpr (WireFp8)
                DataCopyPad(dynamicScales_[uint64_t(row) * data_.scaleBytes], wire[uint32_t(data_.quantCount)],
                            {1U, uint32_t(data_.scaleBytes), 0U, 0U, 0U});
        }
        if (enableDedup_) DataCopyPad(tokenType_[row], type, {1U, 4U, 0U, 0U, 0U});
        DataCopyPad(destinationIndex_[uint64_t(row) * 3U], destination, {1U, 12U, 0U, 0U, 0U});
        if (expertTimer_) expertTimer_->End(expert, LlExpertTimer::OutputWrite);
        if (firstHop) {
            DataCopyPad(relayReadIndex_[uint64_t(row) * k_ * 2U], tail.ReadIndex(), {1U, k_ * 8U, 0U, 0U, 0U});
            const uint32_t sourceRank = tail.Header().GetValue(0U);
            const uint32_t ranks = uint32_t(args_->rankSize);
            const uint32_t members = tail.ReadIndex().GetValue(0U);
            for (uint32_t i = 0U; i < members; ++i) {
                const uint32_t index = tail.Entry(i).GetValue(0U);
                const uint32_t expertId = index & 0xffffU;
                const uint32_t q = index >> 16U;
                const uint32_t destinationRank = expertId / localExperts_;
                if (expertTimer_) expertTimer_->Start(expertId, LlExpertTimer::Forward);
                GlobalTensor<uint8_t> output;
                output.SetGlobalBuffer(args_->peerMems[destinationRank] +
                                       DispatchServerDedupDispatchOffset(data_, sourceRank, destinationRank,
                                                                         expertId % localExperts_, q, ranks,
                                                                         localExperts_, bs_));
                DataCopy(output, wire, uint32_t(data_.tokenStride));
                if (expertTimer_) expertTimer_->End(expertId, LlExpertTimer::Forward);
            }
        }
        SetFlag<HardEvent::MTE3_MTE2>(lane);
    }

    // 摘要：仅清零本核拥有的所有接收槽中的 32B 块标记，保留 payload；完成等待由调用方负责。
    __aicore__ inline void ClearFlags(uint32_t owned)
    {
        LL_SEG_SCOPE(CLEAR_LOCAL_FLAGS);
        auto zeros = zeroFlagsBuf_.Get<float>();
        for (uint32_t j = 0; j < tasks_.Count(); ++j) {
            if (!(owned&(uint32_t(1)<<j))) continue;
            GlobalTensor<float> output;
            output.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(Slot(j) + 480U));
            DataCopyPad(output, zeros, {uint16_t(data_.blockCount), 32U, 0U, 480U, 0U});
        }
    }

    const __gm__ DispatchDedup::CommArgs* args_ = nullptr;
    LlStageTimer::DeviceTimer* timer_ = nullptr;
    LlExpertTimer::DeviceTimer* expertTimer_ = nullptr;
    TPipe* pipe_ = nullptr;
    bool enableDedup_ = false;
    uint64_t windowOffset_ = 0;
    uint32_t coreNum_ = 0;
    uint32_t bs_ = 0U, k_ = 0U, experts_ = 0U, localExperts_ = 0U;
    uint32_t hiddenBytes_ = 0U, laneStride_ = 0U, lanes_ = 0U;
    EpServerDedupDataLayout data_{};
    ServerDedupReceiveTasks tasks_;
    TBuf<> flagsBuf_, sumBuf_, workBuf_, zeroFlagsBuf_, seenBuf_, lanesBuf_;
    GlobalTensor<uint8_t> expandX_, dynamicScales_;
    GlobalTensor<uint32_t> tokenType_, destinationIndex_, relayReadIndex_;
};
}  // namespace DispatchDedup
#endif
