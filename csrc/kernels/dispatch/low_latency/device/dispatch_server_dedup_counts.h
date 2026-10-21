// Copyright (c) 2026, Lu Lu
// Modified by ryan_li 2026

#ifndef DISPATCH_SERVER_DEDUP_COUNTS_H
#define DISPATCH_SERVER_DEDUP_COUNTS_H

#include "kernel_operator.h"
#include "dispatch_frequency_histogram.h"
#include "../common/ll_overlap_layout.h"
#include "../common/ll_stage_timer_device.h"
#include "adv_api/reduce/sum.h"
#include "../common/ep_memory_server_dedup_count_layout.h"
#include "../common/ll_group_local_send.h"

namespace DispatchDedup {
using namespace AscendC;

// 中文审计：这里包含跨rank计数交换，不属于four_metadata测得的本地路由阶段。
// 修改计数协议需联查dispatch_server_dedup_kernel.h中的阶段顺序及Receiver使用的前缀布局。
// Adapted from Memory's CalAndSendCntByRankAndExpert / WaitDispatch / cumsum
// chain. Counts cover the complete original routing tensor, including relay routes.
// 本类没有 Process()：上层依次调用 SendCnt，再由接收核执行
// RecvCnt → CalcCumsum；发送核在 SendCnt 后进入 token 发送，之后所有核CompleteCumsum。
// 计数覆盖原始专家路由（含后续 relay 路由），不按去重后的跨 server 包数计数。
class ServerDedupCounts {
public:
    // 摘要：绑定路由与计数输出，选择本轮控制区；保存并调整 CTRL[60]，供整数位模式的 FP32 归约使用。
    __aicore__ inline void Init(GM_ADDR commArgs, GM_ADDR expertIds, GM_ADDR expertTokenNumsOut, GM_ADDR sendCountsOut,
                                uint32_t bs, uint32_t topK, uint32_t experts, uint32_t expertTokenNumsType,
                                int64_t magic, TPipe* pipe, LlStageTimer::DeviceTimer* timer = nullptr)
    {
        timer_ = timer;
        tpipe_ = pipe;
        args_ = reinterpret_cast<const __gm__ DispatchDedup::CommArgs*>(commArgs);
        epRankId_ = args_->rank;
        epWorldSize_ = args_->rankSize;
        aivId_ = GetBlockIdx();
        coreNum_ = GetBlockNum();
        aivUsedAllToAll_ = kLlSendCoreNum;
        aivUsedCumSum_ = kLlRecvCoreNum;
        axisBS_ = bs;
        axisK_ = topK;
        expertIdsCnt_ = bs * topK;
        rscvStatusNum_ = experts;
        activeCumSumCores_ = LlMakePrefixPartition(experts,0U).active;
        moeExpertNumPerRank_ = experts / epWorldSize_;
        expertTokenNumsType_ = expertTokenNumsType;
        countPlan_ = LlMakeGroupCountPlan(experts, epWorldSize_, epRankId_, aivId_);
        layout_ = DispatchServerDedupCountLayout(experts, coreNum_);
        stateOffset_ = DispatchServerDedupWindowOffset(magic) + layout_.cntOffset;
        GM_ADDR localState = args_->peerMems[epRankId_] + stateOffset_;
        windowInstatusFp32Tensor_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(localState));
        selfRankWinInGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(localState));
        recvCntWorkspaceGM_ = localState + layout_.prefixOffset;
        sendCountsOutGM_ = sendCountsOut;
        expertIdsGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(expertIds));
        expertTokenNumsOutGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(expertTokenNumsOut));
        // Memory's count reductions add integer bit patterns as FP32 subnormals.
        originalOverflowMode_ = GetCtrlSpr<60, 60>();
        SetCtrlSpr<60, 60>(0);
    }

    __aicore__ inline void ConfigureOverlap(LlOverlapLayout plan) { overlap_=plan; }
    // 摘要：统计完整原始路由的专家计数并异步发往专家所属 rank，同时保留仍被 MTE3 使用的记录 UB。
    __aicore__ inline void SendCnt();
    // 摘要：将本 rank 的计数记录分给接收核，等待本片计数到齐并计算局部前缀与总数。
    __aicore__ inline void RecvCnt();
    __aicore__ inline void PrepareRecvCnt();
    __aicore__ inline bool TryRecvCnt();
    // 摘要：汇合接收核片段总数，补齐跨核前缀偏移并发布完整前缀表的就绪标记。
    __aicore__ inline void CalcCumsum()
    {
        LL_SEG_SCOPE(CAL_CUMSUM);
        CalRecvAndSetFlag();
    }
    // 摘要：等待所有前缀片段就绪，加载本核前缀副本，并由首个接收核输出各专家计数。
    __aicore__ inline void CompleteCumsum();
    __aicore__ inline void PreparePrefix();
    __aicore__ inline bool TryCompletePrefix();
    // 摘要：返回后续阶段可复用的 UB 池；在途计数发送使用的 statusBuf_ 不在复用范围内。
    __aicore__ inline TBufPool<TPosition::VECCALC, 16>& ScratchPool()
    {
        return stagePool_;
    }
    // 摘要：返回计数记录与本组任务描述符占用的UB；两者均不能被packet scratch覆盖。
    __aicore__ inline uint32_t ReservedBytes() const
    {
        return statusCntAlign_ * kEpServerDedupCountRecordBytes + localTaskBytes_;
    }
    // A separate alias view leaves stagePool_'s allocation cursor untouched for
    // the original RecvCnt allocations. ProcessGroupLocal drains its reads first.
    __aicore__ inline TBufPool<TPosition::VECCALC,16>& LocalSendScratchPool()
    {
        LL_SEG_SCOPE(PREPARE_INPUT);
        tpipe_->InitBufPool(localSendPool_,overlap_.enabled ? overlap_.sendBytes : kEpServerDedupCountUbBytes-ReservedBytes(),stagePool_);
        return localSendPool_;
    }
    __aicore__ inline uint32_t LocalTaskCount() const { return localTaskCount_; }
    __aicore__ inline LocalTensor<uint32_t> LocalRouteIndices() { return localRouteBuf_.Get<uint32_t>(); }
    __aicore__ inline LocalTensor<uint32_t> LocalExpertQ() { return localExpertQBuf_.Get<uint32_t>(); }
    // Valid after CompleteCumsum and until this core's TPipe is reset.
    // 摘要：返回完整前缀的 UB 视图，仅在 CompleteCumsum 完成后且 TPipe 重置前有效。
    __aicore__ inline LocalTensor<int32_t> Prefix()
    {
        return prefixBuf_.Get<int32_t>();
    }
    // 摘要：等待本核全部流水收尾，并恢复初始化前的 CTRL[60] 设置。
    __aicore__ inline void Finish()
    {
        PipeBarrier<PIPE_ALL>();
        SetCtrlSpr<60, 60>(originalOverflowMode_);
    }

private:
    static constexpr uint32_t UB_ALIGN = 32U;
    static constexpr uint32_t UB_ALIGN_DATA_COUNT = 8U;
    static constexpr uint32_t SIZE_ALIGN_256 = 256U;
    static constexpr uint32_t CNT_GM_STRIDE_BLOCKS = kEpServerDedupCountGmStride / UB_ALIGN;
    uint32_t aivUsedAllToAll_ = 0, coreNum_ = 0;
    uint32_t aivUsedCumSum_ = 0;

    template <HardEvent event>
    // 摘要：用指定硬事件同步两个流水阶段，保护临时张量的生产与消费顺序。
    __aicore__ inline void SyncFunc()
    {
        const TEventID id = overlap_.enabled ?
            TEventID(7) : tpipe_->FetchEventID(event);
        SetFlag<event>(id);
        WaitFlag<event>(id);
    }
    // 摘要：将连续区间均分到各核，余数优先分配给前面的核，返回半开区间及长度。
    __aicore__ inline void SplitToCore(uint32_t total, uint32_t cores, uint32_t core, uint32_t& start, uint32_t& end,
                                       uint32_t& count)
    {
        const uint32_t base = total / cores;
        const uint32_t remainder = total % cores;
        count = base + (core < remainder ? 1U : 0U);
        start = core * base + (core < remainder ? core : remainder);
        end = start + count;
    }
    // 摘要：对原始路由执行逐专家匹配与归约，得到该专家的路由条数。
    __aicore__ inline void CalTokenSendExpertCnt(uint32_t dstExpertId, int32_t calCnt, int32_t& curExpertCnt);
    // 摘要：将专家计数按目标 rank 分组，写到远端“本地专家 × 来源 rank”的 32B 记录槽。
    __aicore__ inline void SendExpertCntRange(uint32_t start, uint32_t end, uint32_t record = 0U);
    // 摘要：向量筛选本核专家区间内的路由，再用直方图生成每专家的 ready 标记和计数。
    __aicore__ inline void CountByVectorRange(uint32_t start, uint32_t end, uint32_t maskBytes, uint32_t record = 0U);
    // 摘要：逐个专家扫描路由并计算计数，作为小规模输入的统计路径。
    __aicore__ inline void CountByExpert(uint32_t start, uint32_t end, uint32_t record = 0U);
    __aicore__ inline void BuildLocalTasks(uint32_t start, uint32_t end);
    // 摘要：清除本核已收取记录的 ready 标记，保留计数字段，为后续窗口复用做准备。
    __aicore__ inline void WaitDispatchClearStatus();
    // 摘要：提取计数字段，计算本片累计前缀，并向其他接收核发布本片总数。
    __aicore__ inline void GatherSumRecvCnt(LocalTensor<float>& gatherMaskOutTensor,
                                            LocalTensor<uint32_t>& gatherTmpTensor,
                                            LocalTensor<float>& statusSumOutTensor);
    // 摘要：将本核片段总数和就绪标记写入本 rank 的跨核汇总矩阵；空片段也必须发布。
    __aicore__ inline void PublishRecvCntSum(float sumOfRecvCnt);
    // 摘要：等待所有接收核总数就绪，求本核之前各片段之和，再清理本核消费的汇总标记。
    __aicore__ inline uint32_t GetCumSum(uint32_t newAivId);
    // 摘要：轮询本片所有来源计数的 ready 标记；到齐后提取计数、发布片段总数并清除 ready。
    // 摘要：为本片局部前缀加上跨核偏移，写出 sendCountsOut 与各核前缀副本，再发布完成标记。
    __aicore__ inline void CalRecvAndSetFlag();
    // 摘要：等待所有接收核的前缀写出标记到齐，并清除本核对应的标记行。
    __aicore__ inline bool TryCumSumFlag();
    __aicore__ inline void IssuePrefixLoad();
    __aicore__ inline void FinishPrefixLoad();
    enum class PrefixState : uint32_t { Unprepared, Polling, Loading, Ready };
    PrefixState prefixState_=PrefixState::Unprepared;
#ifdef DEEPEP_LL_STAGE_TIMER
    uint64_t prefixBegin_=0U, prefixLoadBegin_=0U;
#endif

    LlOverlapLayout overlap_{};
    TBufPool<TPosition::VECCALC,16> recvCountPool_;
    __aicore__ inline TBufPool<TPosition::VECCALC,16>& RecvPool() {
        return overlap_.enabled ? recvCountPool_ : stagePool_;
    }
    TPipe* tpipe_ = nullptr;
    const __gm__ DispatchDedup::CommArgs* args_ = nullptr;
    EpServerDedupCountLayout layout_{};
    uint64_t stateOffset_ = 0;
    int64_t originalOverflowMode_ = 0;
    uint32_t epRankId_ = 0, epWorldSize_ = 0, aivId_ = 0;
    uint32_t axisBS_ = 0, axisK_ = 0, expertIdsCnt_ = 0;
    uint32_t rscvStatusNum_ = 0, moeExpertNumPerRank_ = 0, expertTokenNumsType_ = 0;
    uint32_t statusCntAlign_ = 0;
    uint32_t startStatusIndex_ = 0, endStatusIndex_ = 0, recStatusNumPerCore_ = 0;
    uint32_t activeCumSumCores_ = 0;
    LlGroupCountPlan countPlan_{};
    uint32_t localTaskBytes_ = 0U, localTaskCount_ = 0U;
    TBuf<> localRouteBuf_, localExpertQBuf_;
    GM_ADDR recvCntWorkspaceGM_ = nullptr;
    GM_ADDR sendCountsOutGM_ = nullptr;
    GlobalTensor<int32_t> expertIdsGMTensor_;
    GlobalTensor<int64_t> expertTokenNumsOutGMTensor_;
    GlobalTensor<float> windowInstatusFp32Tensor_, selfRankWinInGMTensor_;
    LocalTensor<int32_t> statusTensor_, validExpertIdsTensor_;
    LocalTensor<float> workLocalTensor_, statusFp32Tensor_, statusCleanFp32Tensor_;
    // Both views outlive all their tensors. The record buffer is outside their
    // shared address range and remains immutable until the phase is retired.
    TBufPool<TPosition::VECCALC, 16> stagePool_;
    TBufPool<TPosition::VECCALC, 5> countPool_;
    TBufPool<TPosition::VECCALC,16> localSendPool_;
    LlStageTimer::DeviceTimer* timer_ = nullptr;
    TBuf<> expertIdsBuf_, dstExpBuf_, subExpBuf_, workBuf_, statusBuf_, histogramBuf_;
    TBuf<> waitStatusBuf_, gatherMaskOutBuf_, sumCoreBuf_, sumLocalBuf_, sumContinueBuf_, scalarBuf_;
    TBuf<> prefixBuf_, readyBuf_, cleanBuf_, tokenNumBuf_;
};

// 摘要：统计完整原始路由的专家计数并异步发往专家所属 rank，同时保留仍被 MTE3 使用的记录 UB。
__aicore__ inline void ServerDedupCounts::SendCnt()
{
    statusCntAlign_=countPlan_.count==0U ? 0U : Ceil(countPlan_.count,UB_ALIGN_DATA_COUNT)*UB_ALIGN_DATA_COUNT;
    const uint32_t recordBytes=statusCntAlign_*kEpServerDedupCountRecordBytes;
    if (recordBytes) tpipe_->InitBuffer(statusBuf_,recordBytes);
    const uint32_t routeBytes=Ceil(expertIdsCnt_*sizeof(int32_t),SIZE_ALIGN_256)*SIZE_ALIGN_256;
    if (countPlan_.local && countPlan_.count) {
        // Persistent descriptors are outside the overlay used by packet packing.
        localTaskBytes_=2U*routeBytes;
        tpipe_->InitBuffer(localRouteBuf_,routeBytes);
        tpipe_->InitBuffer(localExpertQBuf_,routeBytes);
    }
    tpipe_->InitBufPool(stagePool_,overlap_.enabled ? overlap_.sendBytes : kEpServerDedupCountUbBytes-ReservedBytes());
    if (overlap_.enabled && overlap_.countBytes) tpipe_->InitBufPool(recvCountPool_,overlap_.countBytes);
    if (!countPlan_.count) return;
    LL_SEG_SCOPE(SEND_COUNTS);
    LL_SEG_BEGIN(initCountBuffers, INIT_COUNT_BUFFERS);
    tpipe_->InitBufPool(countPool_,4U*routeBytes+statusCntAlign_*sizeof(int32_t),stagePool_);
    countPool_.InitBuffer(expertIdsBuf_,routeBytes);
    countPool_.InitBuffer(dstExpBuf_,routeBytes);
    countPool_.InitBuffer(subExpBuf_,routeBytes);
    countPool_.InitBuffer(workBuf_,routeBytes);
    countPool_.InitBuffer(histogramBuf_,statusCntAlign_*sizeof(int32_t));
    validExpertIdsTensor_=expertIdsBuf_.Get<int32_t>();
    workLocalTensor_=workBuf_.Get<float>();
    statusTensor_=statusBuf_.Get<int32_t>();
    DataCopyPad(validExpertIdsTensor_,expertIdsGMTensor_,
                {1U,expertIdsCnt_*uint32_t(sizeof(int32_t)),0U,0U,0U},{false,0U,0U,0});
    SyncFunc<HardEvent::MTE2_V>();
    uint64_t flagMask[2]={0x0101010101010101ULL,0ULL};
    Duplicate<int32_t>(statusTensor_,0,statusCntAlign_*UB_ALIGN_DATA_COUNT);
    PipeBarrier<PIPE_V>();
    Duplicate<int32_t>(statusTensor_,0x3f800000,flagMask,statusCntAlign_/8U,1U,8U);
    SyncFunc<HardEvent::V_S>();
    LL_SEG_END(initCountBuffers, INIT_COUNT_BUFFERS);
    const uint32_t maskBytes=Ceil(Ceil(expertIdsCnt_,8U),UB_ALIGN)*UB_ALIGN;
    for (uint32_t part=0;part<countPlan_.size;++part) {
        const auto range=countPlan_.ranges[part];
        if (range.local) {
            // Local range is first, so BuildLocalTasks retains record base zero.
            BuildLocalTasks(range.begin,range.end);
        } else {
            if (expertIdsCnt_>=SIZE_ALIGN_256/sizeof(int32_t) && 2U*maskBytes<=routeBytes)
                CountByVectorRange(range.begin,range.end,maskBytes,range.record);
            else CountByExpert(range.begin,range.end,range.record);
        }
    }
    // Records stay immutable while their DMA overlaps either sender.
    SyncFunc<HardEvent::S_MTE3>();
    for (uint32_t part=0;part<countPlan_.size;++part) {
        const auto range=countPlan_.ranges[part];
        SendExpertCntRange(range.begin,range.end,range.record);
    }
}

// 摘要：向量筛选本核专家区间内的路由，再用直方图生成每专家的 ready 标记和计数。
__aicore__ inline void ServerDedupCounts::CountByVectorRange(uint32_t start, uint32_t end, uint32_t maskBytes, uint32_t record)
{
        LL_SEG_SCOPE(COUNT_COMPUTE);
    LocalTensor<int32_t> histogram = histogramBuf_.Get<int32_t>();
    LocalTensor<int32_t> selected = dstExpBuf_.Get<int32_t>();
    LocalTensor<uint8_t> lower = subExpBuf_.Get<uint8_t>();
    LocalTensor<uint8_t> upper = lower[maskBytes];
    SyncFunc<HardEvent::S_V>();
    Compares(lower, validExpertIdsTensor_, int32_t(start), CMPMODE::GE, expertIdsCnt_);
    Compares(upper, validExpertIdsTensor_, int32_t(end), CMPMODE::LT, expertIdsCnt_);
    PipeBarrier<PIPE_V>();
    And(lower.ReinterpretCast<uint16_t>(), lower.ReinterpretCast<uint16_t>(), upper.ReinterpretCast<uint16_t>(),
        int32_t(maskBytes / sizeof(uint16_t)));
    PipeBarrier<PIPE_V>();
    uint64_t selectedCount = 0;
    GatherMask(selected, validExpertIdsTensor_, lower.ReinterpretCast<uint32_t>(), true, expertIdsCnt_,
               {1U, 1U, 0U, 0U}, selectedCount);
    SyncFunc<HardEvent::V_S>();
    auto* output=reinterpret_cast<__ubuf__ uint32_t*>(histogram.GetPhyAddr());
    auto* input=reinterpret_cast<__ubuf__ uint32_t*>(selected.GetPhyAddr());
    // selected is already compact and padded to a 256B allocation. Only one
    // segment decision is made here, never one overflow check per route.
    if (selectedCount<=kLlHistogramSegment)
        asc_vf_call<CountFrequency<true>>(output,input,uint32_t(selectedCount),start,end-start);
    else
        asc_vf_call<CountFrequency<false>>(output,input,uint32_t(selectedCount),start,end-start);
    SyncFunc<HardEvent::V_S>();
    for (uint32_t i = 0; i < end - start; ++i) {
        statusTensor_.SetValue((record+i) * UB_ALIGN_DATA_COUNT + 1U, histogram.GetValue(i));
    }
}

// 摘要：逐个专家扫描路由并计算计数，作为小规模输入的统计路径。
__aicore__ inline void ServerDedupCounts::CountByExpert(uint32_t start, uint32_t end, uint32_t record)
{
        LL_SEG_SCOPE(COUNT_COMPUTE);
    for (uint32_t expert = start; expert < end; ++expert) {
        int32_t count = 0;
        CalTokenSendExpertCnt(expert, expertIdsCnt_, count);
        statusTensor_.SetValue((record+expert-start) * UB_ALIGN_DATA_COUNT + 1U, count);
    }
}

// 摘要：将专家计数按目标 rank 分组，写到远端“本地专家 × 来源 rank”的 32B 记录槽。

// Stable GatherMask keeps original route order. The existing count update also
// produces q: no per-route prefix reduction, remote exchange or scalar owner test.
__aicore__ inline void ServerDedupCounts::BuildLocalTasks(uint32_t start,uint32_t end)
{
    LL_SEG_SCOPE(COUNT_COMPUTE);
    auto lower=subExpBuf_.Get<uint8_t>();
    auto upper=workBuf_.Get<uint8_t>();
    auto selected=dstExpBuf_.Get<int32_t>();
    auto histogram=histogramBuf_.Get<int32_t>();
    Compares(lower,validExpertIdsTensor_,int32_t(start),CMPMODE::GE,expertIdsCnt_);
    Compares(upper,validExpertIdsTensor_,int32_t(end),CMPMODE::LT,expertIdsCnt_);
    PipeBarrier<PIPE_V>();
    const uint32_t maskWords=Ceil(expertIdsCnt_,16U);
    And(lower.ReinterpretCast<uint16_t>(),lower.ReinterpretCast<uint16_t>(),
        upper.ReinterpretCast<uint16_t>(),maskWords);
    CreateVecIndex(selected,int32_t(0),expertIdsCnt_);
    PipeBarrier<PIPE_V>();
    uint64_t selectedCount=0;
    auto routeIndices=localRouteBuf_.Get<uint32_t>();
    GatherMask(routeIndices,selected.ReinterpretCast<uint32_t>(),lower.ReinterpretCast<uint32_t>(),
               true,expertIdsCnt_,{1U,1U,0U,0U},selectedCount);
    Duplicate<int32_t>(histogram,0,statusCntAlign_);
    SyncFunc<HardEvent::V_S>();
    localTaskCount_=uint32_t(selectedCount);
    auto expertQ=localExpertQBuf_.Get<uint32_t>();
    for (uint32_t task=0;task<localTaskCount_;++task) {
        const uint32_t index=routeIndices.GetValue(task);
        const uint32_t expert=uint32_t(validExpertIdsTensor_.GetValue(index));
        const uint32_t q=uint32_t(histogram.GetValue(expert-start));
        histogram.SetValue(expert-start,q+1U);
        expertQ.SetValue(task,expert|((index<axisK_ ? 0U : q)<<16U));
    }
    for (uint32_t expert=start;expert<end;++expert)
        statusTensor_.SetValue((expert-start)*UB_ALIGN_DATA_COUNT+1U,histogram.GetValue(expert-start));
}

__aicore__ inline void ServerDedupCounts::SendExpertCntRange(uint32_t start, uint32_t end, uint32_t record)
{
        LL_SEG_SCOPE(COUNT_SEND_ISSUE);
    GlobalTensor<int32_t> peer;
    for (uint32_t expert = start; expert < end;) {
        const uint32_t dst = expert / moeExpertNumPerRank_;
        const uint32_t localExpert = expert % moeExpertNumPerRank_;
        const uint32_t count =
            end - expert < moeExpertNumPerRank_ - localExpert ? end - expert : moeExpertNumPerRank_ - localExpert;
        const uint64_t offset = (uint64_t(localExpert) * epWorldSize_ + epRankId_) * kEpServerDedupCountGmStride;
        peer.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(args_->peerMems[dst] + stateOffset_ + offset));
        // DataCopy strides are GM count slots: one record per source rank.
        const DataCopyParams copy{uint16_t(count), 1U, 0U, uint16_t(epWorldSize_ * CNT_GM_STRIDE_BLOCKS - 1U)};
        DataCopy<int32_t>(peer, statusTensor_[(record+expert-start) * UB_ALIGN_DATA_COUNT], copy);
        expert += count;
    }
}

// 摘要：将本 rank 的计数记录分给接收核，等待本片计数到齐并计算局部前缀与总数。
__aicore__ inline void ServerDedupCounts::PrepareRecvCnt()
{
    // 接收端控制区共有 localExperts×worldSize（等于 experts）条记录，
    // 按“本地专家 → 来源 rank”排列；每个接收核负责一段连续记录。
    // Each count owns a 512B GM slot, so contiguous ownership can split at any
    // record. Keep all nonempty receivers active and their loads within one record.
    activeCumSumCores_ = LlMakePrefixPartition(rscvStatusNum_,0U).active;
    RecvPool().InitBuffer(scalarBuf_, 3U * UB_ALIGN);
    const uint32_t receiver = aivId_ - aivUsedAllToAll_;
    // Inactive receivers only consume the final prefix in CompleteCumsum.
    if (receiver >= activeCumSumCores_) return;
    LL_SEG_BEGIN(recvBuffersBegin, INIT_COUNT_BUFFERS);
    const auto partition=LlMakePrefixPartition(rscvStatusNum_,receiver);
    startStatusIndex_=partition.begin;
    recStatusNumPerCore_=partition.count;
    endStatusIndex_=partition.begin+partition.count;
    RecvPool().InitBuffer(waitStatusBuf_, Ceil(recStatusNumPerCore_ * UB_ALIGN + SIZE_ALIGN_256, SIZE_ALIGN_256) * SIZE_ALIGN_256);
    RecvPool().InitBuffer(gatherMaskOutBuf_, Ceil(rscvStatusNum_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN);
    // Vector publication/cleanup touches complete groups of eight 32B records.
    // Padding is UB-only; matrix strides and participation use actual core counts.
    const uint32_t sumBytes = Ceil(coreNum_, 8U) * SIZE_ALIGN_256;
    RecvPool().InitBuffer(sumCoreBuf_, sumBytes);
    RecvPool().InitBuffer(sumLocalBuf_, Ceil(aivUsedCumSum_, 8U) * SIZE_ALIGN_256);
    RecvPool().InitBuffer(sumContinueBuf_, Ceil(aivUsedCumSum_ * sizeof(float), UB_ALIGN) * UB_ALIGN);
    // 等待本片 ready 到齐后提取 count，并计算局部累计值、发布本片总数。
    // 本阶段不会等待 token payload；即使分到空片段也要发布总数 0，参与后续跨核汇总。
    LL_SEG_END(recvBuffersBegin, INIT_COUNT_BUFFERS);
}

__aicore__ inline void ServerDedupCounts::RecvCnt()
{
    PrepareRecvCnt();
    while (!TryRecvCnt()) {}
}

// 摘要：等待所有前缀片段就绪，加载本核前缀副本，并由首个接收核输出各专家计数。
__aicore__ inline void ServerDedupCounts::PreparePrefix()
{
    if (prefixState_!=PrefixState::Unprepared) return;
    LL_SEG_SCOPE(PREFIX_COMPLETE);
#ifdef DEEPEP_LL_STAGE_TIMER
    prefixBegin_=timer_ ? timer_->Begin(LlStageTimer::PREFIX_READY_SPAN) : 0U;
#endif
    // Every AIV has a private ready row and UB copy of the shared GM prefix.
    // Retire send/count writes before reusing their UB. No cross-core barrier.
    if (!overlap_.Async()) { LL_SEG_SCOPE(TRANSITION_DRAIN);
    PipeBarrier<PIPE_ALL>();
    }
    if (!overlap_.enabled) tpipe_->Reset();
    // Ready flags use integer bit patterns reduced as FP32 subnormals.
    // Reassert the mode after Reset before the count-protocol reduction.
    SetCtrlSpr<60, 60>(0);
    tpipe_->InitBuffer(scalarBuf_, 3U * UB_ALIGN);
    tpipe_->InitBuffer(readyBuf_, Ceil(activeCumSumCores_ * UB_ALIGN + SIZE_ALIGN_256, SIZE_ALIGN_256) * SIZE_ALIGN_256);
    tpipe_->InitBuffer(cleanBuf_, activeCumSumCores_ * UB_ALIGN);
    tpipe_->InitBuffer(prefixBuf_, Ceil(rscvStatusNum_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN);
    statusFp32Tensor_ = readyBuf_.Get<float>();
    statusCleanFp32Tensor_ = cleanBuf_.Get<float>();
    prefixState_=PrefixState::Polling;
}

// A single readiness snapshot or completion of an already issued load. No
// retry loop here: the caller may submit its next local-send quantum instead.
__aicore__ inline bool ServerDedupCounts::TryCompletePrefix()
{
    if (prefixState_==PrefixState::Ready) return true;
    LL_SEG_SCOPE(PREFIX_COMPLETE);
    if (prefixState_==PrefixState::Loading) {
        FinishPrefixLoad();
        return true;
    }
    if (prefixState_!=PrefixState::Polling) { AscendC::Trap(); return false; }
    if (!TryCumSumFlag()) return false;
    IssuePrefixLoad();
    prefixState_=PrefixState::Loading;
    return false;
}

__aicore__ inline void ServerDedupCounts::IssuePrefixLoad()
{
#ifdef DEEPEP_LL_STAGE_TIMER
    prefixLoadBegin_=timer_ ? timer_->Begin(LlStageTimer::PREFIX_LOAD) : 0U;
#endif
    LocalTensor<int32_t> prefix = prefixBuf_.Get<int32_t>();
    GlobalTensor<int32_t> workspace;
    workspace.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(recvCntWorkspaceGM_));
    const DataCopyExtParams copy{1U, rscvStatusNum_ * uint32_t(sizeof(int32_t)), 0U, 0U, 0U};
    const DataCopyPadExtParams<int32_t> pad{false, 0U, 0U, 0};
    LL_SEG_SCOPE(PREFIX_LOAD_ISSUE);
    DataCopyPad(prefix, workspace, copy, pad);
    SetFlag<HardEvent::MTE2_S>(kLlPrefixLoadEvent);
}

__aicore__ inline void ServerDedupCounts::FinishPrefixLoad()
{
    { LL_SEG_SCOPE(PREFIX_LOAD_WAIT);
    WaitFlag<HardEvent::MTE2_S>(kLlPrefixLoadEvent);
    }
#ifdef DEEPEP_LL_STAGE_TIMER
    if (timer_) timer_->End(LlStageTimer::PREFIX_LOAD,prefixLoadBegin_);
#endif
    const uint32_t receiver=aivId_;
    LocalTensor<int32_t> prefix=prefixBuf_.Get<int32_t>();
    // 仅接收组的第 0 核写公开专家计数，避免多核重复写同一输出。
    // 每个专家最后一个来源 rank 的前缀即该专家结束位置；type=0 输出累计数，
    // 其他已验证类型输出与上一专家累计数之差，即该专家自己的接收条数。
    if (receiver == aivUsedAllToAll_) {
        LL_SEG_SCOPE(WRITE_COUNTS_OUTPUT);
        tpipe_->InitBuffer(tokenNumBuf_, Ceil(moeExpertNumPerRank_ * sizeof(int64_t), UB_ALIGN) * UB_ALIGN);
        LocalTensor<int64_t> expertCounts = tokenNumBuf_.Get<int64_t>();
        int64_t previous = 0;
        for (uint32_t expert = 0; expert < moeExpertNumPerRank_; ++expert) {
            const int64_t cumulative = prefix.GetValue((expert + 1U) * epWorldSize_ - 1U);
            expertCounts.SetValue(expert, expertTokenNumsType_ == 0U ? cumulative : cumulative - previous);
            previous = cumulative;
        }
        SyncFunc<HardEvent::S_MTE3>();
        const DataCopyExtParams output{1U, moeExpertNumPerRank_ * uint32_t(sizeof(int64_t)), 0U, 0U, 0U};
        DataCopyPad(expertTokenNumsOutGMTensor_, expertCounts, output);
    }
    prefixState_=PrefixState::Ready;
#ifdef DEEPEP_LL_STAGE_TIMER
    if (timer_) timer_->End(LlStageTimer::PREFIX_READY_SPAN,prefixBegin_);
#endif
}

// S cores and serial fallback still use blocking completion. The R cooperative
// path calls PreparePrefix/TryCompletePrefix around useful local-send work.
__aicore__ inline void ServerDedupCounts::CompleteCumsum()
{
    PreparePrefix();
    LL_SEG_SCOPE(PREFIX_COMPLETE);
    while (!TryCumSumFlag()) {}
    IssuePrefixLoad();
    FinishPrefixLoad();
}

// 摘要：对原始路由执行逐专家匹配与归约，得到该专家的路由条数。
__aicore__ inline void ServerDedupCounts::CalTokenSendExpertCnt(uint32_t dstExpertId, int32_t calCnt,
                                                                int32_t& curExpertCnt)
{
    if (calCnt < axisK_) {
        curExpertCnt = 0;
        return;
    }
    LocalTensor<int32_t> dstExpIdTensor = dstExpBuf_.Get<int32_t>();
    LocalTensor<int32_t> subExpIdTensor = subExpBuf_.Get<int32_t>();
    Duplicate<int32_t>(dstExpIdTensor, dstExpertId, calCnt);
    PipeBarrier<PIPE_V>();
    Sub(subExpIdTensor, validExpertIdsTensor_, dstExpIdTensor, calCnt);
    PipeBarrier<PIPE_V>();
    LocalTensor<float> tmpFp32 = subExpIdTensor.ReinterpretCast<float>();
    LocalTensor<float> tmpoutFp32 = dstExpIdTensor.ReinterpretCast<float>();
    Abs(tmpoutFp32, tmpFp32, calCnt);
    PipeBarrier<PIPE_V>();
    Mins(subExpIdTensor, dstExpIdTensor, 1, calCnt);
    PipeBarrier<PIPE_V>();
    ReduceSum<float>(tmpoutFp32, tmpFp32, workLocalTensor_, calCnt);
    SyncFunc<AscendC::HardEvent::V_S>();
    int32_t curOtherExpertCnt = dstExpIdTensor(0);
    if (calCnt >= curOtherExpertCnt) {
        curExpertCnt = calCnt - curOtherExpertCnt;
    } else {
        curExpertCnt = 0;
    }
}

// 摘要：清除本核已收取记录的 ready 标记，保留计数字段，为后续窗口复用做准备。
__aicore__ inline void ServerDedupCounts::WaitDispatchClearStatus()
{
        LL_SEG_SCOPE(CLEAR_RECV_STATUS);
    DataCopyParams intriOutParams{static_cast<uint16_t>(recStatusNumPerCore_), 1, 0,
                                  static_cast<uint16_t>(CNT_GM_STRIDE_BLOCKS - 1U)};
    uint64_t duplicateMask[2] = {0x0101010101010101ULL, 0ULL};
    LocalTensor<int32_t> cleanStateTensor = waitStatusBuf_.Get<int32_t>();
    SyncFunc<AscendC::HardEvent::S_V>();
    Duplicate<int32_t>(cleanStateTensor, 0, duplicateMask, Ceil(recStatusNumPerCore_, 8), 1, 8);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(windowInstatusFp32Tensor_[startStatusIndex_ * kEpServerDedupCountGmStride / sizeof(float)],
             cleanStateTensor.ReinterpretCast<float>(), intriOutParams);
    // The snapshot is no longer modified after this issue. Its DMA can overlap
    // cumsum; the caller retires all count-stage writes before receiver Reset.
}

// 摘要：提取计数字段，计算本片累计前缀，并向其他接收核发布本片总数。
__aicore__ inline void ServerDedupCounts::GatherSumRecvCnt(LocalTensor<float>& gatherMaskOutTensor,
                                                           LocalTensor<uint32_t>& gatherTmpTensor,
                                                           LocalTensor<float>& statusSumOutTensor)
{
        LL_SEG_SCOPE(GATHER_RECV_COUNTS);
    gatherTmpTensor.SetValue(0, 2U);
    uint32_t mask = 2U;
    SyncFunc<AscendC::HardEvent::S_V>();

    uint64_t recvCnt = 0;
    GatherMask(gatherMaskOutTensor, statusFp32Tensor_, gatherTmpTensor, true, mask,
               {1, (uint16_t)recStatusNumPerCore_, 1, 0}, recvCnt);
    PipeBarrier<PIPE_V>();

    uint32_t recStatusNumPerCoreInner = Ceil(recStatusNumPerCore_ * sizeof(float), UB_ALIGN) * UB_ALIGN / sizeof(float);
    SumParams sumParams{1, recStatusNumPerCoreInner, recStatusNumPerCore_};
    Sum(statusSumOutTensor, gatherMaskOutTensor, sumParams);
    SyncFunc<AscendC::HardEvent::V_S>();
    float sumOfRecvCnt = statusSumOutTensor.ReinterpretCast<float>().GetValue(0);

    // Compute the relative prefix while outgoing cnt may still be in flight.
    // Gather preserved the integer count bit patterns used by the FP32 Sum.
    LocalTensor<int32_t> prefix = gatherMaskOutTensor.ReinterpretCast<int32_t>();
    uint32_t cumulative = 0;
    for (uint32_t i = 0; i < recStatusNumPerCore_; ++i) {
        cumulative += uint32_t(prefix.GetValue(i));
        prefix.SetValue(i, cumulative);
    }
    if constexpr (!kLlPrefixR0) PublishRecvCntSum(sumOfRecvCnt);
}

// 摘要：将本核片段总数和就绪标记写入本 rank 的跨核汇总矩阵；空片段也必须发布。
__aicore__ inline void ServerDedupCounts::PublishRecvCntSum(float sumOfRecvCnt)
{
    uint32_t newAivId = aivId_ - aivUsedAllToAll_;

    LocalTensor<float> sumCoreFP32Tensor = sumCoreBuf_.Get<float>();
    uint64_t maskArrayCount[2] = {0x0101010101010101, 0};
    uint8_t repeatTimes = Ceil(activeCumSumCores_, 8);

    Duplicate<float>(sumCoreFP32Tensor, sumOfRecvCnt, maskArrayCount, repeatTimes, 1, 8);
    uint64_t maskArrayFlag[2] = {0x0202020202020202, 0};
    Duplicate<float>(sumCoreFP32Tensor, static_cast<float>(1.0), maskArrayFlag, repeatTimes, 1, 8);
    // Keep the matrix row stride, but publish only to active cumsum consumers.
    DataCopyParams sumIntriParams{static_cast<uint16_t>(activeCumSumCores_), 1, 0, 0};
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(selfRankWinInGMTensor_[(layout_.cumsumOffset + newAivId * aivUsedCumSum_ * UB_ALIGN) / sizeof(float)],
             sumCoreFP32Tensor, sumIntriParams);
    // sumCoreBuf_ stays immutable until CalRecvAndSetFlag reuses it. Polling
    // incoming sums must not wait here for this core's outgoing cnt.
}

// 摘要：等待所有接收核总数就绪，求本核之前各片段之和，再清理本核消费的汇总标记。
__aicore__ inline uint32_t ServerDedupCounts::GetCumSum(uint32_t newAivId)
{
        LL_SEG_SCOPE(GET_CUMSUM);
    uint32_t offset = 0;
    DataCopyParams sumIntriParams{static_cast<uint16_t>(activeCumSumCores_), 1,
                                  static_cast<uint16_t>(aivUsedCumSum_ - 1), 0};
    LocalTensor<float> sumLocalTensor = sumLocalBuf_.Get<float>();
    LocalTensor<uint32_t> gatherSumPattern = scalarBuf_.GetWithOffset<uint32_t>(UB_ALIGN / sizeof(uint32_t), 0);
    LocalTensor<float> sumContinueTensor = sumContinueBuf_.Get<float>();
    LocalTensor<float> recvCntSumOutTensor = scalarBuf_.GetWithOffset<float>(UB_ALIGN / sizeof(float), UB_ALIGN);

    uint32_t mask = 2;
    uint64_t recvCnt = 0;
    uint32_t innerSumParams = Ceil(activeCumSumCores_ * sizeof(float), UB_ALIGN) * UB_ALIGN / sizeof(float);
    SumParams sumParams{1, innerSumParams, activeCumSumCores_};
    int32_t cumSumFlag = 0;
    gatherSumPattern.SetValue(0, 2);
    SyncFunc<AscendC::HardEvent::S_V>();

    LL_SEG_BEGIN(corePollBegin, POLL_CORE_SUMS);
    while (true) {
        LL_SEG_COUNT(POLL_CORE_SUMS_ITERATIONS, 1U);
        DataCopy(sumLocalTensor, selfRankWinInGMTensor_[(layout_.cumsumOffset + newAivId * UB_ALIGN) / sizeof(float)],
                 sumIntriParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        GatherMask(sumContinueTensor, sumLocalTensor, gatherSumPattern, true, mask,
                   {1, static_cast<uint16_t>(activeCumSumCores_), 1, 0}, recvCnt);
        PipeBarrier<PIPE_V>();
        Sum(recvCntSumOutTensor, sumContinueTensor, sumParams);
        SyncFunc<AscendC::HardEvent::V_S>();
        cumSumFlag = static_cast<int32_t>(recvCntSumOutTensor.GetValue(0));

        if (cumSumFlag == activeCumSumCores_) {
            break;
        }
    }
    LL_SEG_END(corePollBegin, POLL_CORE_SUMS);
    LL_SEG_BEGIN(reduceBegin, REDUCE_CORE_PREFIX);
    if (newAivId != 0U) {
        mask = 1;
        recvCnt = 0;
        gatherSumPattern.SetValue(0, 1);
        SyncFunc<AscendC::HardEvent::S_V>();
        GatherMask(sumContinueTensor, sumLocalTensor, gatherSumPattern, true, mask,
                   {1, static_cast<uint16_t>(newAivId), 1, 0}, recvCnt);
        PipeBarrier<PIPE_V>();
        uint32_t innerCumSumParams = Ceil(newAivId * sizeof(float), UB_ALIGN) * UB_ALIGN / sizeof(float);
        SumParams cumSumParams{1, innerCumSumParams, newAivId};
        Sum(recvCntSumOutTensor, sumContinueTensor, cumSumParams);
        SyncFunc<AscendC::HardEvent::V_S>();
        offset = uint32_t(recvCntSumOutTensor.ReinterpretCast<int32_t>().GetValue(0));
    }

    LL_SEG_END(reduceBegin, REDUCE_CORE_PREFIX);
    LL_SEG_BEGIN(clearBegin, CLEAR_CORE_SUMS);
    LocalTensor<float> sumCoreFp32Tensor = sumLocalBuf_.Get<float>();

    uint8_t repeatTimes = Ceil(activeCumSumCores_, 8);

    Duplicate<float>(sumCoreFp32Tensor, static_cast<float>(0), 64, repeatTimes, 1, 8);
    DataCopyParams cleanParams{static_cast<uint16_t>(activeCumSumCores_), 1, 0,
                               static_cast<uint16_t>(aivUsedCumSum_ - 1)};
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(selfRankWinInGMTensor_[(layout_.cumsumOffset + newAivId * UB_ALIGN) / sizeof(float)], sumCoreFp32Tensor,
             cleanParams);
    LL_SEG_END(clearBegin, CLEAR_CORE_SUMS);
    return offset;
}

// 摘要：轮询本片所有来源计数的 ready 标记；到齐后提取计数、发布片段总数并清除 ready。
// One bounded snapshot. Once true, the caller must not call this again:
// its count flags have been consumed and its sum has been published.
__aicore__ inline bool ServerDedupCounts::TryRecvCnt()
{
    if (recStatusNumPerCore_==0U) return true;
    LL_SEG_SCOPE(WAIT_RECV_COUNTS);
    LL_SEG_SCOPE(POLL_RECV_COUNTS);
    LL_SEG_COUNT(POLL_RECV_COUNTS_ITERATIONS, 1U);
    // No Reset occurs between cooperative steps; Init preserves CTRL[60]=0.
    auto gather=gatherMaskOutBuf_.Get<float>();
    auto pattern=scalarBuf_.GetWithOffset<uint32_t>(UB_ALIGN/sizeof(uint32_t),0);
    auto sum=scalarBuf_.GetWithOffset<float>(UB_ALIGN/sizeof(float),UB_ALIGN);
    statusFp32Tensor_=waitStatusBuf_.Get<float>();
    DataCopy(statusFp32Tensor_,windowInstatusFp32Tensor_[startStatusIndex_*kEpServerDedupCountGmStride/sizeof(float)],
             DataCopyParams{uint16_t(recStatusNumPerCore_),1,uint16_t(CNT_GM_STRIDE_BLOCKS-1U),0});
    SyncFunc<HardEvent::MTE2_V>();
    if (recStatusNumPerCore_<=255U) {
        ReduceSum(sum,statusFp32Tensor_,gather,1U,recStatusNumPerCore_,1U);
    } else {
        // R0 can own all 1024 records; the basic ReduceSum repeat field is
        // only 8 bits. Compact ready fields, then use the long-count Sum API.
        pattern.SetValue(0U,1U);
        SyncFunc<HardEvent::S_V>();
        uint64_t selected=0;
        GatherMask(gather,statusFp32Tensor_,pattern,true,1U,
                   {1U,uint16_t(recStatusNumPerCore_),1U,0U},selected);
        PipeBarrier<PIPE_V>();
        Sum(sum,gather,SumParams{1U,Ceil(recStatusNumPerCore_,8U)*8U,recStatusNumPerCore_});
    }
    SyncFunc<HardEvent::V_S>();
    if (sum.GetValue(0)!=float(recStatusNumPerCore_)) return false;
    GatherSumRecvCnt(gather,pattern,sum);
    WaitDispatchClearStatus();
    return true;
}

// 摘要：为本片局部前缀加上跨核偏移，写出 sendCountsOut 与各核前缀副本，再发布完成标记。
__aicore__ inline void ServerDedupCounts::CalRecvAndSetFlag()
{
        LL_SEG_SCOPE(BUILD_RECV_CUMSUM);
    if (recStatusNumPerCore_ == 0U) return;
    LocalTensor<int32_t> outCountLocal = gatherMaskOutBuf_.Get<int32_t>();
    uint32_t newAivId = aivId_ - aivUsedAllToAll_;
    // 当前 UB 中是片段内前缀；先等待所有片段总数，再累加编号更小接收核的总数。
    // 这样得到全 rank 接收表中的基址，各核可独立修正自己负责的片段。
    uint32_t offset=0U;
    if constexpr (!kLlPrefixR0) offset=GetCumSum(newAivId);

    if (recStatusNumPerCore_ != 0U) {
        if (offset && recStatusNumPerCore_>=32U) {
            // An affine update, unlike the scalar recurrence that built the
            // local prefix. Preserve scalar->vector and vector->MTE3 ordering.
            SyncFunc<AscendC::HardEvent::S_V>();
            Adds(outCountLocal,outCountLocal,int32_t(offset),recStatusNumPerCore_);
            SyncFunc<AscendC::HardEvent::V_MTE3>();
        } else {
            if (offset) for (uint32_t i = 0; i < recStatusNumPerCore_; ++i) {
                outCountLocal.SetValue(i, uint32_t(outCountLocal.GetValue(i)) + offset);
            }
            SyncFunc<AscendC::HardEvent::S_MTE3>();
        }
        GlobalTensor<int32_t> sendCountsGlobal, workspaceGlobal;
        sendCountsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sendCountsOutGM_));
        workspaceGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(recvCntWorkspaceGM_));
        DataCopyExtParams dataCopyOutParams{1U, static_cast<uint32_t>(recStatusNumPerCore_ * sizeof(int32_t)), 0U, 0U,
                                            0U};
        // sendCountsOut 写的是累计前缀，而非单条记录的原始发送数量。
        DataCopyPad(sendCountsGlobal[startStatusIndex_], outCountLocal, dataCopyOutParams);

        LL_SEG_BEGIN(broadcastBegin, BROADCAST_PREFIX);
        // Disjoint aligned producer slices form one shared GM table.
        DataCopyPad(workspaceGlobal[startStatusIndex_], outCountLocal, dataCopyOutParams);
        LL_SEG_END(broadcastBegin, BROADCAST_PREFIX);
    }
    LL_SEG_BEGIN(flagsBegin, PUBLISH_PREFIX_FLAGS);
    uint8_t repeatTimes = Ceil(coreNum_, 8);
    DataCopyParams sumIntriParams{static_cast<uint16_t>(coreNum_), 1, 0,
                                  static_cast<uint16_t>(aivUsedCumSum_ - 1)};
    LocalTensor<int32_t> syncOnCoreTensor = sumCoreBuf_.Get<int32_t>();
    LocalTensor<float> syncOnCoreFP32Tensor = sumCoreBuf_.Get<float>();

    // 发布前缀完成标记前，先等待旧片段总数发送不再读取 sumCoreBuf_，
    // 再复用它构造标记；前缀写入与完成标记之间保留原 MTE3 顺序约束。
    // This is the first overwrite of the asynchronous local-sum publication.
    SyncFunc<AscendC::HardEvent::MTE3_V>();
    Duplicate<int32_t>(syncOnCoreTensor, static_cast<int32_t>(1), SIZE_ALIGN_256 / sizeof(int32_t), repeatTimes, 1, 8);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    PipeBarrier<PIPE_MTE3>();

    DataCopy(selfRankWinInGMTensor_[(layout_.cumsumFlagOffset + newAivId * UB_ALIGN) / sizeof(float)],
             syncOnCoreFP32Tensor, sumIntriParams);
    LL_SEG_END(flagsBegin, PUBLISH_PREFIX_FLAGS);
}

// 摘要：等待所有接收核的前缀写出标记到齐，并清除本核对应的标记行。
__aicore__ inline bool ServerDedupCounts::TryCumSumFlag()
{
        LL_SEG_SCOPE(WAIT_CUMSUM);
    int32_t cumSumFlag = 0;
    int32_t targetFlag = activeCumSumCores_ * UB_ALIGN_DATA_COUNT;
    uint32_t cumSumFlagOffset =
        (layout_.cumsumFlagOffset + aivId_ * aivUsedCumSum_ * UB_ALIGN) / sizeof(float);
    uint32_t innerSumParams = activeCumSumCores_ * UB_ALIGN / sizeof(float);
    SumParams sumFlagParams{1, innerSumParams, activeCumSumCores_ * UB_ALIGN_DATA_COUNT};
    LocalTensor<float> statusSumOutTensor = scalarBuf_.Get<float>();

    {
        LL_SEG_COUNT(WAIT_CUMSUM_ITERATIONS, 1U);
        DataCopy(statusFp32Tensor_, selfRankWinInGMTensor_[cumSumFlagOffset], activeCumSumCores_ * UB_ALIGN_DATA_COUNT);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Sum(statusSumOutTensor, statusFp32Tensor_, sumFlagParams);
        SyncFunc<AscendC::HardEvent::V_S>();
        cumSumFlag = statusSumOutTensor.ReinterpretCast<int32_t>().GetValue(0);

        if (cumSumFlag != targetFlag) return false;
    }
    Duplicate<float>(statusCleanFp32Tensor_, static_cast<float>(0), activeCumSumCores_ * UB_ALIGN_DATA_COUNT);
    SyncFunc<HardEvent::S_MTE3>();

    SyncFunc<HardEvent::V_MTE3>();
    DataCopy(selfRankWinInGMTensor_[cumSumFlagOffset], statusCleanFp32Tensor_,
             activeCumSumCores_ * UB_ALIGN_DATA_COUNT);
    return true;
}

}  // namespace DispatchDedup
#endif
