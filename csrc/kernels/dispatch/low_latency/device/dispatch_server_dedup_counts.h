// Copyright (c) 2026, Lu Lu
// Modified by ryan_li 2026

#ifndef DISPATCH_SERVER_DEDUP_COUNTS_H
#define DISPATCH_SERVER_DEDUP_COUNTS_H

#include "kernel_operator.h"
#include "dispatch_frequency_histogram.h"
#include "dispatch_prefix_scan.h"
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
// 本类没有 Process()：R核发布计数，R0计算完整前缀。
// 仅R核发布SendCnt，R0执行RecvCnt/CalcCumsum；所有核CompleteCumsum读取同一张前缀。
// 计数覆盖原始专家路由（含后续 relay 路由），不按去重后的跨 server 包数计数。
class ServerDedupCounts {
public:
    // 摘要：绑定路由与计数输出，选择本轮控制区；保存并调整 CTRL[60]，供整数位模式的 FP32 归约使用。
    __aicore__ inline void Init(GM_ADDR commArgs, GM_ADDR expertIds, GM_ADDR expertTokenNumsOut, GM_ADDR sendCountsOut,
                                uint32_t bs, uint32_t topK, uint32_t experts, uint32_t expertTokenNumsType,
                                int64_t magic, TPipe* pipe, LlStageTimer::DeviceTimer* timer = nullptr, bool enableDedup = true)
    {
        timer_ = timer;
        buildLocalTasks_=enableDedup;
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
        prefixEpoch_ = uint32_t(magic);
        prefixReadyGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(
            args_->peerMems[epRankId_]+uint64_t(kLlSendCoreNum)*kEpServerDedupMagicStride+32U));
        prefixReadyGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
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
    // 只有R0接收全部计数，其他R核直接继续本组发送。
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
    // S-only bounded reader while its input queue is still alive. The caller
    // leases both events from TPipe after queue initialization; no MTE2_S 6/7.
    __aicore__ inline bool TryLoadPrefixOnce(TEventID mte2Vector, TEventID vectorScalar);
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
    __aicore__ inline void BuildLocalTasks(uint32_t start,uint32_t end,uint32_t record);
    // 摘要：清除本核已收取记录的 ready 标记，保留计数字段，为后续窗口复用做准备。
    __aicore__ inline void WaitDispatchClearStatus();
    // R0轮询全部来源计数，提取计数并清理本轮通知。
    __aicore__ inline void CalRecvAndSetFlag();
    // 等待R0发布当前epoch，不清理共享标记；下轮由R0交替发布。
    __aicore__ inline void WaitCumSumFlag();
    __aicore__ inline void EnsurePrefixReader()
    {
        if (prefixReaderAllocated_) return;
        tpipe_->InitBuffer(readyBuf_,32U);
        tpipe_->InitBuffer(prefixBuf_,Ceil(rscvStatusNum_*4U,32U)*32U);
        prefixReaderAllocated_=true;
    }

    LlOverlapLayout overlap_{};
    bool buildLocalTasks_=true;
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
    LlGroupCountPlan countPlan_{};
    uint32_t localTaskBytes_ = 0U, localTaskCount_ = 0U;
    TBuf<> localRouteBuf_, localExpertQBuf_;
    GM_ADDR recvCntWorkspaceGM_ = nullptr;
    GM_ADDR sendCountsOutGM_ = nullptr;
    GlobalTensor<int32_t> expertIdsGMTensor_;
    GlobalTensor<int64_t> expertTokenNumsOutGMTensor_;
    GlobalTensor<float> windowInstatusFp32Tensor_, selfRankWinInGMTensor_;
    LocalTensor<int32_t> statusTensor_, validExpertIdsTensor_;
    LocalTensor<float> workLocalTensor_, statusFp32Tensor_;
    GlobalTensor<uint32_t> prefixReadyGm_;
    uint32_t prefixEpoch_=0U;
    bool prefixReaderAllocated_=false, prefixLoaded_=false;
    TBuf<> prefixPublishBuf_;
    // Both views outlive all their tensors. The record buffer is outside their
    // shared address range and remains immutable until the phase is retired.
    TBufPool<TPosition::VECCALC, 16> stagePool_;
    TBufPool<TPosition::VECCALC, 5> countPool_;
    TBufPool<TPosition::VECCALC,16> localSendPool_;
    LlStageTimer::DeviceTimer* timer_ = nullptr;
    TBuf<> expertIdsBuf_, dstExpBuf_, subExpBuf_, workBuf_, statusBuf_, histogramBuf_;
    TBuf<> waitStatusBuf_, gatherMaskOutBuf_, sumContinueBuf_, scalarBuf_;
    TBuf<> prefixBuf_, readyBuf_, tokenNumBuf_;
};

// 摘要：统计完整原始路由的专家计数并异步发往专家所属 rank，同时保留仍被 MTE3 使用的记录 UB。
__aicore__ inline void ServerDedupCounts::SendCnt()
{
    statusCntAlign_=countPlan_.count==0U ? 0U : Ceil(countPlan_.count,UB_ALIGN_DATA_COUNT)*UB_ALIGN_DATA_COUNT;
    const uint32_t recordBytes=statusCntAlign_*kEpServerDedupCountRecordBytes;
    if (recordBytes) tpipe_->InitBuffer(statusBuf_,recordBytes);
    const uint32_t routeBytes=Ceil(expertIdsCnt_*sizeof(int32_t),SIZE_ALIGN_256)*SIZE_ALIGN_256;
    if (buildLocalTasks_ && countPlan_.local && countPlan_.count) {
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
        if (buildLocalTasks_ && range.local) BuildLocalTasks(range.begin,range.end,range.record);
        else if (expertIdsCnt_>=SIZE_ALIGN_256/sizeof(int32_t) && 2U*maskBytes<=routeBytes)
            CountByVectorRange(range.begin,range.end,maskBytes,range.record);
        else CountByExpert(range.begin,range.end,range.record);
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
__aicore__ inline void ServerDedupCounts::BuildLocalTasks(uint32_t start,uint32_t end,uint32_t record)
{
    LL_SEG_SCOPE(COUNT_COMPUTE);
    SyncFunc<HardEvent::S_V>();
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
        statusTensor_.SetValue((record+expert-start)*UB_ALIGN_DATA_COUNT+1U,histogram.GetValue(expert-start));
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

// 只有R0接收全部计数，其他R核直接继续本组发送。
__aicore__ inline void ServerDedupCounts::PrepareRecvCnt()
{
    recStatusNumPerCore_=0U;
    if (aivId_!=kLlSendCoreNum) return;
    LL_SEG_SCOPE(INIT_COUNT_BUFFERS);
    startStatusIndex_=0U;endStatusIndex_=rscvStatusNum_;recStatusNumPerCore_=rscvStatusNum_;
    RecvPool().InitBuffer(scalarBuf_,96U);
    RecvPool().InitBuffer(waitStatusBuf_,Ceil(rscvStatusNum_*32U+256U,256U)*256U);
    RecvPool().InitBuffer(gatherMaskOutBuf_,Ceil(rscvStatusNum_*4U,256U)*256U);
    RecvPool().InitBuffer(sumContinueBuf_,Ceil(rscvStatusNum_*4U,32U)*32U);
    RecvPool().InitBuffer(prefixPublishBuf_,32U);
}

__aicore__ inline void ServerDedupCounts::RecvCnt()
{
    PrepareRecvCnt();
    while (!TryRecvCnt()) {}
}

// 摘要：等待所有前缀片段就绪，加载本核前缀副本，并由首个接收核输出各专家计数。
__aicore__ inline void ServerDedupCounts::CompleteCumsum()
{
    LL_SEG_SCOPE(PREFIX_COMPLETE);
    const uint32_t receiver=aivId_;
    if (!overlap_.Async()) { LL_SEG_SCOPE(TRANSITION_DRAIN); PipeBarrier<PIPE_ALL>(); }
    if (!overlap_.enabled) tpipe_->Reset();
    SetCtrlSpr<60,60>(0);
    EnsurePrefixReader();
    LocalTensor<int32_t> prefix = prefixBuf_.Get<int32_t>();
    if (!prefixLoaded_) {
    WaitCumSumFlag();
    GlobalTensor<int32_t> workspace;
    workspace.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(recvCntWorkspaceGM_));
    workspace.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    const DataCopyExtParams copy{1U, rscvStatusNum_ * uint32_t(sizeof(int32_t)), 0U, 0U, 0U};
    const DataCopyPadExtParams<int32_t> pad{false, 0U, 0U, 0};
    { LL_SEG_SCOPE(PREFIX_LOAD);
    DataCopyPad(prefix, workspace, copy, pad);
    SyncFunc<HardEvent::MTE2_S>();
    }
    prefixLoaded_=true;
    }
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
    // No listening table or payload receive/copy is constructed in this increment.
}

__aicore__ inline bool ServerDedupCounts::TryLoadPrefixOnce(TEventID mte2Vector, TEventID vectorScalar)
{
    if (prefixLoaded_) return true;
    EnsurePrefixReader();
    auto ready=readyBuf_.Get<uint32_t>();
    { LL_SEG_SCOPE(WAIT_CUMSUM);
      LL_SEG_COUNT(WAIT_CUMSUM_ITERATIONS,1U);
      DataCopy(ready,prefixReadyGm_,8U);
      // A finite MTE2 -> V -> S join protects this scalar read without
      // stealing the sender's pending-input MTE2_S event. No PIPE_ALL/drain.
      SetFlag<HardEvent::MTE2_V>(mte2Vector);
      WaitFlag<HardEvent::MTE2_V>(mte2Vector);
      SetFlag<HardEvent::V_S>(vectorScalar);
      WaitFlag<HardEvent::V_S>(vectorScalar);
    }
    if (ready.GetValue(0U)!=prefixEpoch_) return false;
    GlobalTensor<int32_t> workspace;
    workspace.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(recvCntWorkspaceGM_));
    workspace.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    { LL_SEG_SCOPE(PREFIX_LOAD);
      DataCopyPad(prefixBuf_.Get<int32_t>(),workspace,
                  {1U,rscvStatusNum_*uint32_t(sizeof(int32_t)),0U,0U,0U},{false,0U,0U,0});
      SetFlag<HardEvent::MTE2_V>(mte2Vector);
      WaitFlag<HardEvent::MTE2_V>(mte2Vector);
      SetFlag<HardEvent::V_S>(vectorScalar);
      WaitFlag<HardEvent::V_S>(vectorScalar);
    }
    prefixLoaded_=true;
    return true;
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

// R0轮询全部来源计数，提取计数并清理本轮通知。
// One bounded snapshot. Once true, the caller must not call this again:
// its count flags have been consumed; CalcCumsum publishes the prefix.
__aicore__ inline bool ServerDedupCounts::TryRecvCnt()
{
    if (!recStatusNumPerCore_) return true;
    LL_SEG_SCOPE(WAIT_RECV_COUNTS);
    LL_SEG_SCOPE(POLL_RECV_COUNTS);
    LL_SEG_COUNT(POLL_RECV_COUNTS_ITERATIONS,1U);
    statusFp32Tensor_=waitStatusBuf_.Get<float>();
    DataCopy(statusFp32Tensor_,windowInstatusFp32Tensor_,
             DataCopyParams{uint16_t(rscvStatusNum_),1U,uint16_t(CNT_GM_STRIDE_BLOCKS-1U),0U});
    SyncFunc<HardEvent::MTE2_V>();
    auto gather=gatherMaskOutBuf_.Get<float>();
    auto pattern=scalarBuf_.GetWithOffset<uint32_t>(8U,0U);
    auto sum=scalarBuf_.GetWithOffset<float>(8U,32U);
    pattern.SetValue(0U,1U);
    SyncFunc<HardEvent::S_V>();
    uint64_t n=0;
    GatherMask(gather,statusFp32Tensor_,pattern,true,2U,{1U,uint16_t(rscvStatusNum_),1U,0U},n);
    PipeBarrier<PIPE_V>();
    // Gather first: E can exceed the uint8 repeat count of a strided ReduceSum.
    ReduceSum(sum,gather,sumContinueBuf_.Get<float>(),rscvStatusNum_);
    SyncFunc<HardEvent::V_S>();
    if (sum.GetValue(0U)!=float(rscvStatusNum_)) return false;
    pattern.SetValue(0U,2U);
    SyncFunc<HardEvent::S_V>();
    GatherMask(gather,statusFp32Tensor_,pattern,true,2U,{1U,uint16_t(rscvStatusNum_),1U,0U},n);
    SyncFunc<HardEvent::V_S>();
    WaitDispatchClearStatus();
    return true;
}

__aicore__ inline void ServerDedupCounts::CalRecvAndSetFlag()
{
    if (aivId_!=kLlSendCoreNum) return;
    LL_SEG_SCOPE(BUILD_RECV_CUMSUM);
    auto prefix=gatherMaskOutBuf_.Get<int32_t>();
    // Gather completion and scalar/Vector ownership precede the VF; data-ready
    // publication below still waits for the actual prefix DMA completion.
    SyncFunc<HardEvent::S_V>();
    asc_vf_call<R0PrefixScan>(reinterpret_cast<__ubuf__ uint32_t*>(prefix.GetPhyAddr()),rscvStatusNum_);
    SyncFunc<HardEvent::V_MTE3>();
    GlobalTensor<int32_t> workspace,output;
    workspace.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(recvCntWorkspaceGM_));
    workspace.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    output.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sendCountsOutGM_));
    SyncFunc<HardEvent::S_MTE3>();
    const DataCopyExtParams copy{1U,rscvStatusNum_*4U,0U,0U,0U};
    DataCopyPad(output,prefix,copy);
    { LL_SEG_SCOPE(BROADCAST_PREFIX);
      DataCopyPad(workspace,prefix,copy); } // One shared table, no per-core replicas.
    // Finish prefix and count-ready cleanup before releasing a single epoch.
    { LL_SEG_SCOPE(PREFIX_PUBLISH_WAIT); SyncFunc<HardEvent::MTE3_S>(); }
    LL_SEG_SCOPE(PUBLISH_PREFIX_FLAGS);
    auto ready=prefixPublishBuf_.Get<uint32_t>();
    Duplicate<uint32_t>(ready,0U,8U);
    SyncFunc<HardEvent::V_S>();
    ready.SetValue(0U,prefixEpoch_);
    SyncFunc<HardEvent::S_MTE3>();
    DataCopy(prefixReadyGm_,ready,8U);
    // Publication source is persistent until Finish (or fallback's PIPE_ALL).
}

// 等待R0发布当前epoch，不清理共享标记；下轮由R0交替发布。
__aicore__ inline void ServerDedupCounts::WaitCumSumFlag()
{
    LL_SEG_SCOPE(WAIT_CUMSUM);
    auto ready=readyBuf_.Get<uint32_t>();
    do {
        LL_SEG_COUNT(WAIT_CUMSUM_ITERATIONS,1U);
        DataCopy(ready,prefixReadyGm_,8U);
        SyncFunc<HardEvent::MTE2_S>();
    } while (ready.GetValue(0U)!=prefixEpoch_);
    // Fixed unbanked epoch alternates 1/0. Consumers never clear the producer flag.
}

}  // namespace DispatchDedup
#endif
