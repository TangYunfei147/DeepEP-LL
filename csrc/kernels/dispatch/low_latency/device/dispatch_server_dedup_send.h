// Copyright (c) 2026, Lu Lu
// Modified by ryan_li 2026

// 中文审计：生产Sender与第2层元数据实验的共同源码。
// Process负责UB/路由准备和发送调度；CalTokenSendExpertCnt计算q；SendPacket负责实际包构造与发送。
// prepare_snapshot.py仅在实验副本中截断SendPacket并加入观测输出，生产路径不受该提取影响。
#ifndef DISPATCH_SERVER_DEDUP_SEND_H
#define DISPATCH_SERVER_DEDUP_SEND_H
#include "../common/ll_overlap_layout.h"
#include "../common/ll_stage_timer_device.h"

#include "kernel_operator.h"
#include "dispatch_dedup_mxfp8_quant.h"
#include "../common/ep_memory_server_dedup_data_layout.h"
#include "../common/ep_memory_server_dedup_protocol.h"
#include "../common/ll_send_order.h"
#include "../common/ll_group_compact.h"
#include "../common/ll_input_prefetch.h"
#include "../../../common/ep_packet_receive_layout.h"

namespace DispatchDedup {
using namespace AscendC;

// R02-01/R05-04 source path: first hops precede the cached DIRECT tasks on
// each core, sharing one packet sequence and ring of in-flight send buffers.
template <class XType, bool WireFp8, bool E4M3, bool Prequantized = false>
class ServerDedupSender {
    static_assert(!Prequantized || WireFp8, "prequantized input requires FP8 wire");
    static constexpr bool NeedQuantize = WireFp8 && !Prequantized;
    static constexpr EpServerDedupInputMode InputMode =
        Prequantized ? EpServerDedupInputMode::PrequantizedFp8Packs
                     : (NeedQuantize ? EpServerDedupInputMode::Quantize16 : EpServerDedupInputMode::Plain16);

public:
    // 摘要：绑定输入输出和本核 token 分片，并按 magic 奇偶选择本轮通信窗口。
    __aicore__ inline void Init(const __gm__ DispatchDedup::CommArgs* args, GM_ADDR x, GM_ADDR ids, GM_ADDR weights,
                                GM_ADDR sourceMask, GM_ADDR sourceWeights, uint32_t bs, uint32_t h, uint32_t k,
                                uint32_t experts, TPipe* pipe, int64_t magic = 0, GM_ADDR inputScales = nullptr,
                                bool enableDedup = true, uint32_t coreNum = 0U)
    {
        args_ = args;
        (void)coreNum; // The launch split is fixed by the build configuration.
        sendCores_ = kLlSendCoreNum;
        pipe_ = pipe;
        enableDedup_ = enableDedup;
        bs_ = bs;
        h_ = h;
        k_ = k;
        localExperts_ = experts / args->rankSize;
        const uint32_t core = GetBlockIdx();
        const uint32_t base = bs / sendCores_, remainder = bs % sendCores_;
        count_ = base + (core < remainder ? 1U : 0U);
        start_ = core * base + (core < remainder ? core : remainder);
        data_ = DispatchServerDedupDataLayout(bs, h, k, experts, InputMode);
        data_.dispatchOffset += DispatchServerDedupWindowOffset(magic);
        inputScales_.SetGlobalBuffer(inputScales);
        x_.SetGlobalBuffer(reinterpret_cast<__gm__ XType*>(x));
        ids_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ids));
        weights_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weights));
        maskOut_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(sourceMask));
        weightOut_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sourceWeights));
    }

    // Consume outstanding sender events only after receiving. Sources remain
    // allocated until the caller's final PIPE_ALL. Empty senders have zero lanes.
    __aicore__ inline void Retire() {
        if (overlap_.Async()) {
            for (uint32_t lane=0;lane<lanes_;++lane) WaitFlag<HardEvent::MTE3_V>(lane);
        }
    }
    __aicore__ inline void ConfigureOverlap(LlOverlapLayout plan) { overlap_=plan; }
    // 摘要：执行本核发送主流程：准备 UB、优先发送去重首跳、再发送 DIRECT，最后写回源端元数据。
    __aicore__ inline void Process(TBufPool<TPosition::VECCALC, 16>& scratch, uint32_t reservedBytes,
                                   LlStageTimer::DeviceTimer* timer = nullptr)
    {
        // Packet ownership is independent of the metadata token partition.
        // Even a core with count_==0 may own remote packets. Count scratch
        // excludes reservedBytes still referenced by in-flight count writes.
        timer_ = timer;
        LL_SEG_BEGIN(segBegin1, PREPARE_INPUT);
        scratch_ = &scratch;
        TQue<QuePosition::VECIN, 1> input;
        TBufPool<TPosition::VECIN, 1> cleanPool;
        TBuf<> cleanInput;
        // 输入队列使用两份缓冲：MTE2 搬入 token 后交给量化/打包阶段。
        // 量化会读取对齐到 256 元素的尾部，所以需先把两份输入缓冲的 padding 清零。
        const uint32_t inputBytes =
            NeedQuantize ? uint32_t(data_.quantCount * sizeof(XType)) : uint32_t(data_.packedPayloadBytes);
        scratch_->InitBuffer(input, 2U, inputBytes);
        usedBytes_ = reservedBytes + 2U * inputBytes;
        if constexpr (NeedQuantize || Prequantized) {
            // Alias the two input buffers exactly as Memory does. Quantization
            // reads the compute padding, so initialize it once before copying input.
            pipe_->InitBufPool(cleanPool, 2U * inputBytes, scratch);
            cleanPool.InitBuffer(cleanInput, 2U * inputBytes);
            Duplicate<uint8_t>(cleanInput.Get<uint8_t>(), 0U, 2U * inputBytes);
            Sync<HardEvent::V_MTE2>();
        }

        // 2. 分配路由、q 前缀统计、权重、源掩码以及两类发送任务的临时空间。
        // 路由表加载本 rank 全部 bs×k 项，因为 q 必须按原始路由顺序计算，
        // 不能仅统计当前核的 token 分片；跨分片发送也需要完整输入权重。
        const uint32_t routeBytes = uint32_t(DispatchDedupAlign(uint64_t(bs_) * k_ * 4U, 256U));
        const uint32_t qBytes = uint32_t(DispatchDedupAlign(uint64_t(bs_) * k_ * 4U, 32U));
        const uint32_t scaleCount = uint32_t(data_.quantCount / 32U);
        const uint32_t quantWorkBytes =
            uint32_t(DispatchDedupAlign(DispatchDedupAlign(scaleCount, 32U) * 4U + uint64_t(scaleCount) * 2U, 32U));
        const uint32_t scratchBytes = NeedQuantize && quantWorkBytes > qBytes ? quantWorkBytes : qBytes;
        Allocate(idsBuf_, routeBytes);
        Allocate(qDstBuf_, scratchBytes);
        Allocate(qSubBuf_, scratchBytes);
        Allocate(qWorkBuf_, qBytes);
        if constexpr (NeedQuantize) Allocate(quantBuf_, uint32_t(data_.packedPayloadBytes));
        Allocate(weightInBuf_, uint32_t(DispatchDedupAlign(uint64_t(bs_) * k_ * 4U, 32U)));
        Allocate(weightOutBuf_, uint32_t(DispatchDedupAlign(uint64_t(count_) * k_ * 4U, 32U)));
        Allocate(maskBuf_, uint32_t(DispatchDedupAlign(uint64_t(count_) * 4U, 32U)));
        // Stable compact indices survive q/quantization scratch reuse.
        Allocate(directBuf_, qBytes);
        // One uint32 state per server, including EP256's 16 host groups.
        const uint32_t serverBytes = uint32_t(DispatchDedupAlign(uint64_t(args_->rankSize / kLlRanksPerGroup) * 4U, 32U));
        Allocate(serverBuf_, serverBytes < 64U ? 64U : serverBytes);
        const uint32_t remoteServers = args_->rankSize / kLlRanksPerGroup - 1U;
        const uint32_t maxGroups = remoteServers < k_ / 2U ? remoteServers : k_ / 2U;
        if (maxGroups != 0U) Allocate(firstHopBuf_, uint32_t(DispatchDedupAlign(maxGroups * 4U, 32U)));

        // The design's safe input contract guarantees room for at least one lane.
        // 3. 用剩余 UB 构造最多 8 个发送 lane，每个 lane 容纳一个完整 wire packet。
        // lane 数由当前 shape 决定；输入容量校验需保证至少能容纳一个。
        const uint32_t available = (kEpServerDedupCountUbBytes - usedBytes_) / uint32_t(data_.tokenStride);
        lanes_ = overlap_.enabled ? overlap_.sendLanes : (available < 8U ? available : 8U);
        Allocate(outBuf_, lanes_ * uint32_t(data_.tokenStride));
        routeIds_ = idsBuf_.Get<int32_t>();
        routeWeights_ = weightInBuf_.Get<float>();
        if (count_ != 0U) {
            sourceWeights_ = weightOutBuf_.Get<float>();
            sourceMasks_ = maskBuf_.Get<uint32_t>();

        }
        directTasks_ = directBuf_.Get<uint32_t>();
        serverState_ = serverBuf_.Get<uint32_t>();
        if (maxGroups != 0U) firstHopMasks_ = firstHopBuf_.Get<uint32_t>();
        DataCopyPad(routeIds_, ids_, {1U, bs_ * k_ * 4U, 0U, 0U, 0U}, {false, 0U, 0U, 0});
        DataCopyPad(routeWeights_, weights_, {1U, bs_ * k_ * 4U, 0U, 0U, 0U},
                    {false, 0U, 0U, 0.0f});
        // Stage the owned metadata at UB offset zero: start_*k_ need not be
        // 32-byte aligned. Do not vector-copy an unaligned slice of full weights.
        if (count_ != 0U)
            DataCopyPad(sourceWeights_, weights_[uint64_t(start_)*k_],
                        {1U,count_*k_*4U,0U,0U,0U},{false,0U,0U,0.0f});
        Sync<HardEvent::MTE2_S>();
        // 每个 512B 块尾部的 32B 标记预置为 1；发送时 payload 和标记一起搬运。
        // 初始 MTE3_V 事件让 lane 首次可写，后续复用由上一包的发送完成事件控制。
        uint64_t flagMask[2] = {0xffU, 0U};
        for (uint32_t lane = 0; lane < lanes_; ++lane) {
            // Initialize only the 32B flags. Padding and unused entries carry no semantics.
            Duplicate<float>(outBuf_.Get<float>()[lane * data_.tokenStride / 4U + 120U], 1.0f, flagMask,
                             uint8_t(data_.blockCount), 1U, 16U);
            SetFlag<HardEvent::MTE3_V>(lane);
        }
        PipeBarrier<PIPE_V>();

        LL_SEG_END(segBegin1, PREPARE_INPUT);
        { LL_SEG_SCOPE(SEND_TOKENS);
        BeginInputPrefetch();
        if (enableDedup_)
            SendWithDedup(input);
        else
            SendWithoutDedup(input);
        EndInputPrefetch(input);
        }

        // Keep metadata sources alive through receive. Only the serial fallback
        // drains here; overlap retires all lane events at final kernel completion.
        if (!overlap_.Async()) {
        LL_SEG_BEGIN(sendDrainBegin, SEND_DRAIN);
        for (uint32_t lane = 0; lane < lanes_; ++lane) WaitFlag<HardEvent::MTE3_V>(lane);
        LL_SEG_END(sendDrainBegin, SEND_DRAIN);
        }
        Sync<HardEvent::S_MTE3>();
        if (enableDedup_ && count_ != 0U) DataCopyPad(maskOut_[start_], sourceMasks_, {1U, count_ * 4U, 0U, 0U, 0U});
        if (count_ != 0U)
            DataCopyPad(weightOut_[uint64_t(start_) * k_], sourceWeights_,
                        {1U, count_ * k_ * 4U, 0U, 0U, 0U});
        { LL_SEG_SCOPE(SEND_SUBMIT_END); }
        if (!overlap_.Async()) { LL_SEG_SCOPE(SEND_COMPLETE_WAIT);
        Sync<HardEvent::MTE3_S>();
        }
    }


    // Original R cores; optional bounded callbacks advance counts and receive.
    // Descriptors came from the source-count pass and remain outside scratch.
    struct NoCooperate { __aicore__ inline void operator()() {} };
    template<class Cooperate=NoCooperate>
    __aicore__ inline void ProcessGroupLocal(TBufPool<TPosition::VECCALC,16>& scratch,
                                            uint32_t reservedBytes,LocalTensor<uint32_t> indices,
                                            LocalTensor<uint32_t> expertQ,uint32_t tasks,
                                            LlStageTimer::DeviceTimer* timer=nullptr, Cooperate cooperate=Cooperate{})
    {
        if (!tasks) return;
        timer_=timer;
        LL_SEG_BEGIN(localPrepareBegin, PREPARE_INPUT);
        scratch_=&scratch;
        start_=0U; count_=bs_;
        TQue<QuePosition::VECIN,1> input;
        TBufPool<TPosition::VECIN,1> cleanPool;
        TBuf<> cleanInput;
        const uint32_t inputBytes=NeedQuantize ? uint32_t(data_.quantCount*sizeof(XType))
                                              : uint32_t(data_.packedPayloadBytes);
        scratch_->InitBuffer(input,2U,inputBytes);
        usedBytes_=reservedBytes+2U*inputBytes;
        if constexpr (NeedQuantize || Prequantized) {
            pipe_->InitBufPool(cleanPool,2U*inputBytes,scratch);
            cleanPool.InitBuffer(cleanInput,2U*inputBytes);
            Duplicate<uint8_t>(cleanInput.Get<uint8_t>(),0U,2U*inputBytes);
            Sync<HardEvent::V_MTE2>();
        }
        if constexpr (NeedQuantize) {
            const uint32_t scales=uint32_t(data_.quantCount/32U);
            const uint32_t work=uint32_t(DispatchDedupAlign(DispatchDedupAlign(scales,32U)*4U+uint64_t(scales)*2U,32U));
            Allocate(qDstBuf_,work);
            Allocate(quantBuf_,uint32_t(data_.packedPayloadBytes));
        }
        if (enableDedup_) {
            Allocate(weightInBuf_,uint32_t(DispatchDedupAlign(uint64_t(bs_)*k_*4U,32U)));
            routeWeights_=weightInBuf_.Get<float>();
            DataCopyPad(routeWeights_,weights_,{1U,bs_*k_*4U,0U,0U,0U},{false,0U,0U,0.0f});
            Sync<HardEvent::MTE2_S>();
        }
        const uint32_t available=(kEpServerDedupCountUbBytes-usedBytes_)/uint32_t(data_.tokenStride);
        lanes_=overlap_.enabled ? overlap_.sendLanes : (available<8U ? available : 8U);
        Allocate(outBuf_,lanes_*uint32_t(data_.tokenStride));
        uint64_t flagMask[2]={0xffU,0U};
        for (uint32_t lane=0;lane<lanes_;++lane) {
            Duplicate<float>(outBuf_.Get<float>()[lane*data_.tokenStride/4U+120U],1.0f,flagMask,
                             uint8_t(data_.blockCount),1U,16U);
            SetFlag<HardEvent::MTE3_V>(lane);
        }
        PipeBarrier<PIPE_V>();
        LL_SEG_END(localPrepareBegin, PREPARE_INPUT);
        // Only the cooperative overlap path yields. Serial fallback keeps one
        // continuous lookahead stream. Eight tasks amortize the callback/flush.
        cooperate();
        const uint32_t quantum=overlap_.Cooperative() ? 8U:tasks;
        for (uint32_t begin=0;begin<tasks;) {
            const uint32_t end=tasks-begin<quantum ? tasks:begin+quantum;
            { LL_SEG_SCOPE(SEND_TOKENS);
            BeginInputPrefetch();
            for (uint32_t task=begin;task<end;++task) {
                const uint32_t index=indices.GetValue(task),packed=expertQ.GetValue(task);
                SendPacket(input,index/k_,index%k_,packed&0xffffU,packed>>16U,0U);
            }
            EndInputPrefetch(input);
            }
            begin=end;
            if (begin==tasks) { LL_SEG_SCOPE(SEND_SUBMIT_END); }
            cooperate();
        }
        if (!overlap_.Async()) {
        LL_SEG_BEGIN(localDrainBegin, SEND_DRAIN);
        // Retire all reads of packet scratch before RecvCnt overlays that UB.
        for (uint32_t lane=0;lane<lanes_;++lane) WaitFlag<HardEvent::MTE3_V>(lane);
        if (!overlap_.Async()) { LL_SEG_SCOPE(SEND_COMPLETE_WAIT);
        Sync<HardEvent::MTE3_S>();
        }
        if (!overlap_.enabled) scratch_->Reset();
        // Packet packing may use a different control mode; the unchanged
        // count-receive reductions require integer FP32 subnormals preserved.
        SetCtrlSpr<60,60>(0);
        LL_SEG_END(localDrainBegin, SEND_DRAIN);
        }

    }

    // Original token owners alone publish source masks/weights. Packet owners
    // are selected independently below; the minimum-k first hop is unchanged.
    __aicore__ inline void SendWithDedup(TQue<QuePosition::VECIN, 1>& input)
    {
        LL_SEG_BEGIN(metadataBegin, PREPARE_INPUT);
        for (uint32_t token=0; token<count_; ++token) {
            uint32_t directMask=0, firstHopCount=0, localMask=0;
            Classify(token,directMask,firstHopCount,localMask);
            uint32_t sourceMask=directMask|localMask;
            for (uint32_t group=0; group<firstHopCount; ++group) {
                const uint32_t mask=firstHopMasks_.GetValue(group);
                uint32_t firstK=0;
                while ((mask & (uint32_t(1)<<firstK))==0U) ++firstK;
                sourceMask |= uint32_t(1)<<firstK;
                sourceWeights_.SetValue(token*k_+firstK,1.0f);
            }
            sourceMasks_.SetValue(token,sourceMask);
        }
        LL_SEG_END(metadataBegin, PREPARE_INPUT);
        SendCompactGroups(input,true);
    }

    __aicore__ inline void SendWithoutDedup(TQue<QuePosition::VECIN, 1>& input)
    {
        SendCompactGroups(input,false);
    }

    // Vector selection preserves original route order. Reuse q scratch only
    // during selection; directTasks_ is separate from all q/quantization work.
    __aicore__ inline uint32_t BuildGroupIndices(uint32_t group)
    {
        LL_SEG_SCOPE(PREPARE_INPUT);
        const uint32_t routes=bs_*k_, first=group*localExperts_*kLlRanksPerGroup;
        auto lower=qSubBuf_.Get<uint8_t>();
        auto upper=qWorkBuf_.Get<uint8_t>();
        auto indices=qDstBuf_.Get<int32_t>();
        // q scratch may have just been used by scalar q reads or quantization.
        Sync<HardEvent::S_V>();
        Compares(lower,routeIds_,int32_t(first),CMPMODE::GE,routes);
        Compares(upper,routeIds_,int32_t(first+localExperts_*kLlRanksPerGroup),CMPMODE::LT,routes);
        PipeBarrier<PIPE_V>();
        And(lower.ReinterpretCast<uint16_t>(),lower.ReinterpretCast<uint16_t>(),
            upper.ReinterpretCast<uint16_t>(),(routes+15U)/16U);
        CreateVecIndex(indices,0,routes);
        PipeBarrier<PIPE_V>();
        uint64_t selected=0;
        GatherMask(directTasks_,indices.ReinterpretCast<uint32_t>(),lower.ReinterpretCast<uint32_t>(),
                   true,routes,{1U,1U,0U,0U},selected);
        Sync<HardEvent::V_S>();
        return uint32_t(selected);
    }

    struct CompactAccess {
        LocalTensor<uint32_t> indices;
        __aicore__ inline uint32_t Index(uint32_t i) { return indices.GetValue(i); }
    };
    struct CompactEmit {
        ServerDedupSender& sender;
        TQue<QuePosition::VECIN,1>& input;
        __aicore__ inline void operator()(uint32_t index,uint32_t mask) {
            const uint32_t expert=uint32_t(sender.routeIds_.GetValue(index));
            const uint32_t q=sender.CalTokenSendExpertCnt(expert,index);
            sender.SendPacket(input,index/sender.k_,index%sender.k_,expert,q,mask);
        }
    };
    __aicore__ inline void SendCompactGroups(TQue<QuePosition::VECIN,1>& input,bool dedup)
    {
        const uint32_t groups=args_->rankSize/kLlRanksPerGroup;
        const uint32_t own=args_->rank/kLlRanksPerGroup, core=GetBlockIdx();
        const uint32_t phase=uint32_t(args_->rank)%kLlRanksPerGroup;
        uint32_t owner=phase%sendCores_;
        CompactAccess access{directTasks_};
        CompactEmit emit{*this,input};
        // First hops precede singleton DIRECT packets on every core. There is
        // no inter-core barrier. A COMMON group order is required for ownership;
        // rank phase varies, core phase does not. Carry the cursor across phases.
        for (uint32_t pass=0; pass<(dedup?2U:1U); ++pass) {
            for (uint32_t step=0; step<groups; ++step) {
                const uint32_t group=LlDirectGroupOrder(groups,own,phase,0U,step);
                if (group==own) continue; // R still owns local-group initial sends.
                const uint32_t size=BuildGroupIndices(group);
                LlVisitCompactGroup(access,size,k_,dedup,dedup && pass==0U,
                                    sendCores_,core,owner,emit);
            }
        }
    }

private:
    // 摘要：通过指定流水硬事件建立先后依赖，等待前序操作满足后续流水的访问要求。
    template <HardEvent event>
    __aicore__ inline void Sync()
    {
        const TEventID id = pipe_->FetchEventID(event);
        SetFlag<event>(id);
        WaitFlag<event>(id);
    }
    // 摘要：从计数阶段共享的 scratch 池分配 UB，并累计已占用字节数。
    __aicore__ inline void Allocate(TBuf<>& buffer, uint32_t bytes)
    {
        // Auto keeps the dedup layout even for cores owning routes but no tokens.
        if (bytes == 0U) return;
        scratch_->InitBuffer(buffer, bytes);
        usedBytes_ += bytes;
    }
    // 摘要：按目标 server 对当前 token 的路由分类：同 server 多条远端路由合并为 FIRST_HOP，其余为 DIRECT。
    __aicore__ inline void Classify(uint32_t token, uint32_t& directMask, uint32_t& groups, uint32_t& localMask)
    {
        uint32_t seen = 0, grouped = 0;
        for (uint32_t k = 0; k < k_; ++k) {
            const uint32_t expert = uint32_t(routeIds_.GetValue((start_ + token) * k_ + k));
            const uint32_t server = expert / localExperts_ / kLlRanksPerGroup;
            const uint32_t bit = uint32_t(1) << k, serverBit = uint32_t(1) << server;
            if (server == uint32_t(args_->rank) / kLlRanksPerGroup) {
                localMask |= bit;
            } else if ((seen & serverBit) == 0U) {
                seen |= serverBit;
                serverState_.SetValue(server, k);
                directMask |= bit;
            } else if ((grouped & serverBit) == 0U) {
                const uint32_t firstBit = uint32_t(1) << serverState_.GetValue(server);
                directMask &= ~firstBit;
                firstHopMasks_.SetValue(groups, firstBit | bit);
                serverState_.SetValue(server, groups++);
                grouped |= serverBit;
            } else {
                const uint32_t index = serverState_.GetValue(server);
                firstHopMasks_.SetValue(index, firstHopMasks_.GetValue(index) | bit);
            }
        }
    }
    // 摘要：统计原始展平路由中 index 之前指向该专家的条数，作为槽内序号 q；首 token 按当前约定返回 0。
    __aicore__ inline uint32_t CalTokenSendExpertCnt(uint32_t expert, uint32_t index)
    {
        LL_SEG_SCOPE(CAL_TOKEN);
        if (index < k_) return 0U;
        // Memory's prefix reduction over the complete original route sequence.
        auto dst = qDstBuf_.Get<int32_t>();
        auto sub = qSubBuf_.Get<int32_t>();
        Duplicate<int32_t>(dst, int32_t(expert), index);
        PipeBarrier<PIPE_V>();
        Sub(sub, routeIds_, dst, index);
        PipeBarrier<PIPE_V>();
        auto input = sub.template ReinterpretCast<float>();
        auto output = dst.template ReinterpretCast<float>();
        Abs(output, input, index);
        PipeBarrier<PIPE_V>();
        Mins(sub, dst, 1, index);
        PipeBarrier<PIPE_V>();
        ReduceSum<float>(output, input, qWorkBuf_.Get<float>(), index);
        Sync<HardEvent::V_S>();
        return index - uint32_t(dst.GetValue(0));
    }
    template <class Fp8Type>
    // 摘要：以每 32 元素为一组生成 MXFP8 scale 和量化数据，复用 q 计算的临时 UB。
    __aicore__ inline void QuantMxfp8(LocalTensor<uint8_t> output, LocalTensor<XType> input)
    {
        LL_SEG_SCOPE(TOKEN_QUANT);
        const uint32_t scaleCount = uint32_t(data_.quantCount / 32U);
        auto work = qDstBuf_.Get<float>();
        auto* src = reinterpret_cast<__ubuf__ XType*>(input.GetPhyAddr());
        auto* maxExp = reinterpret_cast<__ubuf__ uint16_t*>(work.GetPhyAddr());
        auto* halfScale = reinterpret_cast<__ubuf__ uint16_t*>(work[DispatchDedupAlign(scaleCount, 32U)].GetPhyAddr());
        auto* dst = reinterpret_cast<__ubuf__ int8_t*>(output.GetPhyAddr());
        auto* scale = reinterpret_cast<__ubuf__ uint16_t*>(output[data_.quantCount].GetPhyAddr());
        DispatchDedupMxfp8Quant::ComputeMaxExp(src, maxExp, uint32_t(data_.quantCount));
        DispatchDedupMxfp8Quant::ComputeScale<Fp8Type>(maxExp, scale, halfScale, scaleCount);
        DispatchDedupMxfp8Quant::ComputeFp8Data<XType, Fp8Type, RoundMode::CAST_TRUNC, RoundMode::CAST_RINT>(
            src, halfScale, dst, uint32_t(data_.quantCount));
    }
    // 摘要：填充源 rank/token、路由类型和首路由信息，并为首跳包编码其余成员的专家、q 与权重。
    __aicore__ inline void FillTail(LocalTensor<uint8_t> payload, uint32_t token, uint32_t firstK, uint32_t mask)
    {
        auto header = payload[data_.headerPayloadOffset].template ReinterpretCast<uint32_t>();
        header.SetValue(0U, uint32_t(args_->rank));
        header.SetValue(1U, token);
        header.SetValue(2U, firstK);
        auto index = payload[data_.readIndexPayloadOffset].template ReinterpretCast<uint32_t>();
        header.SetValue(3U, uint32_t(mask == 0U ? EpServerDedupTokenType::Direct : EpServerDedupTokenType::FirstHop));
        // Direct route ownership differs from the weight-copy token partition.
        // Its reserved relay payload is unused; never index routeWeights_ here.
        if (!enableDedup_) {
            index.SetValue(0U, 0U);
            return;
        }
        auto weights = index.template ReinterpretCast<float>();
        weights.SetValue(1U, routeWeights_.GetValue(token * k_ + firstK));
        uint32_t entries = 0U;
        for (uint32_t k = firstK + 1U; mask != 0U && k < k_; ++k)
            if (mask & (uint32_t(1) << k)) {
                const uint32_t routeIndex = token * k_ + k;
                const uint32_t expert = uint32_t(routeIds_.GetValue(routeIndex));
                const uint32_t q = CalTokenSendExpertCnt(expert, routeIndex);
                ++entries;
                index.SetValue(entries * 2U, expert | (q << 16U));
                weights.SetValue(entries * 2U + 1U, routeWeights_.GetValue(token * k_ + k));
            }
        index.SetValue(0U, entries);
    }
    // The generator retains its original ordering; only execution is delayed
    // by one descriptor so next-input MTE2 can overlap current compute/pack.
    __aicore__ inline void SendPacket(TQue<QuePosition::VECIN,1>& input,uint32_t token,uint32_t firstK,
                                     uint32_t expert,uint32_t q,uint32_t mask)
    {
        PrefetchOps ops{*this,input};
        inputPrefetch_.Push(ops,LlSendPacketTask{token,firstK,expert,q,mask});
    }

    __aicore__ inline void BeginInputPrefetch()
    {
        inputPrefetch_.Reset();
        inputsIssued_=0U;
        if constexpr (!NeedQuantize) {
            inputReady_[0]=overlap_.enabled ? TEventID(6) : pipe_->AllocEventID<HardEvent::MTE2_S>();
            inputReady_[1]=overlap_.enabled ? TEventID(7) : pipe_->AllocEventID<HardEvent::MTE2_S>();
        }
    }
    __aicore__ inline void EndInputPrefetch(TQue<QuePosition::VECIN,1>& input)
    {
        PrefetchOps ops{*this,input};
        inputPrefetch_.Flush(ops); // Includes singleton and final packet.
        if constexpr (!NeedQuantize) {
            if (!overlap_.enabled) {
            pipe_->ReleaseEventID<HardEvent::MTE2_S>(inputReady_[0]);
            pipe_->ReleaseEventID<HardEvent::MTE2_S>(inputReady_[1]);
            }
        }
    }
    __aicore__ inline void IssueInput(TQue<QuePosition::VECIN,1>& input,uint32_t token)
    {
        LL_SEG_BEGIN(segBegin2, TOKEN_INPUT);
        auto x=input.template AllocTensor<XType>();
        // Quantized modes clear both buffers once in PREPARE_INPUT. Per-packet
        // clearing here would add a V dependency ahead of current computation.
        // Payload/scales and every valid tail field are overwritten per use;
        // old bytes outside those fields are padding, never consumed.
        DataCopyPad(x,x_[uint64_t(token)*h_],{1U,h_*uint32_t(sizeof(XType)),0U,0U,0U},
                    {NeedQuantize,0U,0U,XType(0)});
        if constexpr (Prequantized) {
            auto scale=x.template ReinterpretCast<uint8_t>()[uint32_t(data_.quantCount)];
            DataCopyPad(scale,inputScales_[uint64_t(token)*data_.scaleBytes],
                        {1U,uint32_t(data_.scaleBytes),0U,0U,0U},{false,0U,0U,uint8_t(0)});
        }
        if constexpr (!NeedQuantize)
            SetFlag<HardEvent::MTE2_S>(inputReady_[inputsIssued_&1U]);
        input.EnQue(x);
        ++inputsIssued_;
        LL_SEG_END(segBegin2, TOKEN_INPUT);
    }
    struct PrefetchOps {
        ServerDedupSender& sender;
        TQue<QuePosition::VECIN,1>& input;
        __aicore__ inline void Issue(const LlSendPacketTask& task) { sender.IssueInput(input,task.token); }
        __aicore__ inline void Consume(const LlSendPacketTask& task) {
            sender.ConsumePacket(input,sender.pendingInput_,task);
        }
        __aicore__ inline void Take() {
            sender.pendingInput_=input.template DeQue<XType>();
            sender.pendingReady_=(sender.inputsIssued_-1U)&1U;
        }
    };
    __aicore__ inline void ConsumePacket(TQue<QuePosition::VECIN,1>& input,LocalTensor<XType> x,
                                        const LlSendPacketTask& task)
    {
        const uint32_t token=task.token,firstK=task.firstK,expert=task.expert,q=task.q,mask=task.mask;
        const uint32_t destination=expert/localExperts_;
        LocalTensor<uint8_t> payload;
        if constexpr (NeedQuantize) {
            payload = quantBuf_.Get<uint8_t>();
            if constexpr (E4M3)
                QuantMxfp8<fp8_e4m3fn_t>(payload, x);
            else
                QuantMxfp8<fp8_e5m2_t>(payload, x);
            input.template FreeTensor<XType>(x);
            LL_SEG_BEGIN(quantWaitBegin, TOKEN_QUANT_WAIT);
            Sync<HardEvent::V_S>();
            LL_SEG_END(quantWaitBegin, TOKEN_QUANT_WAIT);
        } else {
            LL_SEG_BEGIN(inputWaitBegin, TOKEN_INPUT_WAIT);
            // This marker precedes the NEXT input's DMA. A fresh SetFlag here
            // would instead wait for both inputs and destroy the overlap.
            WaitFlag<HardEvent::MTE2_S>(inputReady_[pendingReady_]);
            LL_SEG_END(inputWaitBegin, TOKEN_INPUT_WAIT);
            payload = x.template ReinterpretCast<uint8_t>();
        }
        LL_SEG_BEGIN(segBegin5, TOKEN_METADATA);
        FillTail(payload, token, firstK, mask);
        LL_SEG_END(segBegin5, TOKEN_METADATA);
        Sync<HardEvent::S_V>();
        const uint32_t lane = packets_ % lanes_;
        LL_SEG_BEGIN(segBegin6, TOKEN_BUFFER_WAIT);
        WaitFlag<HardEvent::MTE3_V>(lane);
        LL_SEG_END(segBegin6, TOKEN_BUFFER_WAIT);
        LL_SEG_BEGIN(packBegin, TOKEN_PACK);
        auto out = outBuf_.Get<uint8_t>()[lane * data_.tokenStride];
        auto outWords = out.template ReinterpretCast<int32_t>();
        auto inWords = payload.template ReinterpretCast<int32_t>();
        Copy(outWords, inWords, uint64_t(64), uint8_t(data_.blockCount), {1, 1, 16, 15});
        Copy(outWords[64], inWords[64], uint64_t(56), uint8_t(data_.blockCount), {1, 1, 16, 15});
        // Refresh this packet's layout marker on every reused lane, before
        // the existing V_MTE3 publication dependency.
        uint64_t markerMask[2] = {1U, 0U};
        Duplicate<float>(out.template ReinterpretCast<float>()[PacketReceiveLayout::kMarkerOffset / 4U],
                         mask != 0U ? PacketReceiveLayout::kKeepFlags : PacketReceiveLayout::kCompact, markerMask, 1U,
                         1U, 8U);
        if constexpr (!NeedQuantize) input.template FreeTensor<XType>(x);
        SetFlag<HardEvent::V_MTE3>(lane);
        WaitFlag<HardEvent::V_MTE3>(lane);
        LL_SEG_END(packBegin, TOKEN_PACK);
        GlobalTensor<uint8_t> destinationGm;
        destinationGm.SetGlobalBuffer(args_->peerMems[destination] +
                                      DispatchServerDedupDispatchOffset(data_, args_->rank, destination,
                                                                        expert % localExperts_, q, args_->rankSize,
                                                                        localExperts_, bs_));
        LL_SEG_BEGIN(segBegin7, TOKEN_SEND_ISSUE);
        DataCopy(destinationGm, out, uint32_t(data_.tokenStride));
        SetFlag<HardEvent::MTE3_V>(lane);
        LL_SEG_END(segBegin7, TOKEN_SEND_ISSUE);
        ++packets_;
    }

    const __gm__ DispatchDedup::CommArgs* args_ = nullptr;
    TPipe* pipe_ = nullptr;
    LlStageTimer::DeviceTimer* timer_ = nullptr;
    bool enableDedup_ = false;
    TBufPool<TPosition::VECCALC, 16>* scratch_ = nullptr;
    uint32_t sendCores_ = 0;
    uint32_t bs_ = 0, h_ = 0, k_ = 0, localExperts_ = 0, start_ = 0, count_ = 0;
    LlOverlapLayout overlap_{};
    uint32_t usedBytes_ = 0, lanes_ = 0, packets_ = 0;
    LlOnePacketInput inputPrefetch_;
    LocalTensor<XType> pendingInput_;
    TEventID inputReady_[2]{};
    uint32_t inputsIssued_=0U,pendingReady_=0U;
    EpServerDedupDataLayout data_{};
    GlobalTensor<XType> x_;
    GlobalTensor<uint8_t> inputScales_;
    GlobalTensor<int32_t> ids_;
    GlobalTensor<float> weights_, weightOut_;
    GlobalTensor<uint32_t> maskOut_;
    TBuf<> idsBuf_, qDstBuf_, qSubBuf_, qWorkBuf_, quantBuf_;
    TBuf<> weightInBuf_, weightOutBuf_, maskBuf_, directBuf_, serverBuf_, firstHopBuf_, outBuf_;
    LocalTensor<int32_t> routeIds_;
    LocalTensor<float> routeWeights_, sourceWeights_;
    LocalTensor<uint32_t> sourceMasks_, directTasks_, serverState_, firstHopMasks_;
};
}  // namespace DispatchDedup
#endif
