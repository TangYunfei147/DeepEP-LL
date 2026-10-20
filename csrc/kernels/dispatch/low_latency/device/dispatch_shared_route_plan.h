#pragma once
#include "kernel_operator.h"
#include "../common/ll_route_plan.h"
#include "../common/ep_memory_server_dedup_protocol.h"
namespace DispatchDedup {
using namespace AscendC;
template<HardEvent E> __aicore__ inline void RoutePlanSync(TPipe* pipe) {
    const TEventID event=pipe->FetchEventID(E); SetFlag<E>(event); WaitFlag<E>(event);
}

// Three LOCAL-device phase boundaries. Every launched core participates even
// when its token/expert interval is empty. No remote payload starts beforehand.
__aicore__ inline void BuildSharedRoutePlan(GM_ADDR base,GM_ADDR ids,uint32_t bs,uint32_t k,
                                           uint32_t experts,uint32_t ranks,uint32_t rank,bool enableDedup,TPipe* pipe) {
    const uint32_t core=GetBlockIdx(),workers=kLlActiveCoreNum;
    const uint32_t tokens=bs/workers+(core<bs%workers ? 1U : 0U);
    const uint32_t first=core*(bs/workers)+(core<bs%workers ? core : bs%workers);
    const uint32_t routes=tokens*k,groupExperts=experts/ranks*kLlRanksPerGroup;
    const uint32_t groups=ranks/kLlRanksPerGroup;
    const auto layout=LlRoutePlanMemory(uint64_t(bs)*k,experts);
    pipe->Reset();
    TBuf<> histBuf,idsBuf,qBuf,masksBuf,stateBuf,bucketBuf,totalsBuf;
    const uint32_t shardBytes=uint32_t(LlPlanAlign(routes ? uint64_t(routes)*4U : 1U,256U));
    pipe->InitBuffer(histBuf,experts*32U);
    pipe->InitBuffer(idsBuf,shardBytes); pipe->InitBuffer(qBuf,shardBytes);
    pipe->InitBuffer(masksBuf,shardBytes);
    pipe->InitBuffer(stateBuf,uint32_t(LlPlanAlign(groups*8U,32U)));
    auto hist=histBuf.Get<uint32_t>(),localIds=idsBuf.Get<uint32_t>();
    auto localQ=qBuf.Get<uint32_t>(),masks=masksBuf.Get<uint32_t>(),state=stateBuf.Get<uint32_t>();
    Duplicate<uint32_t>(hist,0U,experts*8U);
    Duplicate<uint32_t>(state,0U,uint32_t(LlPlanAlign(groups*8U,32U))/4U);
    GlobalTensor<uint32_t> input,matrix,totals,outIds,outQ,outMasks;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ids));
    matrix.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(base));
    totals.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(base+layout.totalsOffset));
    outIds.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(base+layout.idsOffset));
    outQ.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(base+layout.qOffset));
    outMasks.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(base+layout.masksOffset));
    if (routes) DataCopyPad(localIds,input[first*k],{1U,routes*4U,0U,0U,0U},{false,0U,0U,0U});
    RoutePlanSync<HardEvent::MTE2_S>(pipe); RoutePlanSync<HardEvent::V_S>(pipe);
    for (uint32_t t=0;t<tokens;++t) {
        for (uint32_t j=0;j<k;++j) {
            const uint32_t i=t*k+j,e=localIds.GetValue(i),g=e/groupExperts;
            localQ.SetValue(i,hist.GetValue(e*8U));
            hist.SetValue(e*8U,hist.GetValue(e*8U)+1U);
            const uint32_t old=(state.GetValue(g*2U)&0xffffU)==t+1U ? state.GetValue(g*2U+1U) : 0U;
            state.SetValue(g*2U,t+1U); state.SetValue(g*2U+1U,old|(uint32_t(1)<<j));
        }
        for (uint32_t j=0;j<k;++j) {
            const uint32_t i=t*k+j,g=localIds.GetValue(i)/groupExperts;
            const uint32_t mask=state.GetValue(g*2U+1U);
            masks.SetValue(i,mask);
            const uint32_t anchor=LlPlanFirst(mask);
            if (j==anchor) {
                const uint32_t remaining=mask&~(uint32_t(1)<<anchor);
                const uint32_t event=remaining ? LlPlanFirst(remaining) : anchor;
                const uint32_t leader=!enableDedup || !remaining || g==rank/kLlRanksPerGroup ? anchor :
                    LlChooseLeader(localIds[t*k],k,mask,experts/ranks);
                state.SetValue(g*2U,(t+1U)|(leader<<16U)|(event<<24U));
            }
            localQ.SetValue(i,localQ.GetValue(i)|(state.GetValue(g*2U)&0xffff0000U));
        }
    }
    RoutePlanSync<HardEvent::S_MTE3>(pipe);
    DataCopy(matrix[core*8U],hist,DataCopyParams{uint16_t(experts),1U,0U,uint16_t(workers-1U)});
    PipeBarrier<PIPE_ALL>(); SyncAll<true>();

    const uint32_t owned=experts/workers+(core<experts%workers ? 1U : 0U);
    const uint32_t eBegin=core*(experts/workers)+(core<experts%workers ? core : experts%workers);
    if (owned) {
        pipe->InitBuffer(bucketBuf,owned*workers*32U); pipe->InitBuffer(totalsBuf,owned*32U);
        auto bucket=bucketBuf.Get<uint32_t>(),counts=totalsBuf.Get<uint32_t>();
        DataCopy(bucket,matrix[eBegin*workers*8U],owned*workers*8U);
        Duplicate<uint32_t>(counts,0U,owned*8U);
        RoutePlanSync<HardEvent::MTE2_S>(pipe); RoutePlanSync<HardEvent::V_S>(pipe);
        for (uint32_t e=0;e<owned;++e) {
            uint32_t sum=0U;
            for (uint32_t c=0;c<workers;++c) {
                const uint32_t index=(e*workers+c)*8U,n=bucket.GetValue(index);
                bucket.SetValue(index,sum); sum+=n;
            }
            counts.SetValue(e*8U,0x3f800000U); counts.SetValue(e*8U+1U,sum);
        }
        RoutePlanSync<HardEvent::S_MTE3>(pipe);
        DataCopy(matrix[eBegin*workers*8U],bucket,owned*workers*8U);
        DataCopy(totals[eBegin*8U],counts,owned*8U);
    }
    PipeBarrier<PIPE_ALL>(); SyncAll<true>();
    // Fetch this token producer's expert offsets, then turn local q into q
    // in the complete original token order. First-token convention unchanged.
    DataCopy(hist,matrix[core*8U],DataCopyParams{uint16_t(experts),1U,uint16_t(workers-1U),0U});
    RoutePlanSync<HardEvent::MTE2_S>(pipe);
    for (uint32_t i=0;i<routes;++i) {
        const uint32_t packed=localQ.GetValue(i),e=localIds.GetValue(i);
        const uint32_t q=first*k+i<k ? 0U : (packed&0xffffU)+hist.GetValue(e*8U);
        localQ.SetValue(i,(packed&0xffff0000U)|q);
    }
    RoutePlanSync<HardEvent::S_MTE3>(pipe);
    if (routes) {
        const DataCopyExtParams copy{1U,routes*4U,0U,0U,0U};
        DataCopyPad(outIds[first*k],localIds,copy); DataCopyPad(outQ[first*k],localQ,copy);
        DataCopyPad(outMasks[first*k],masks,copy);
    }
    PipeBarrier<PIPE_ALL>(); SyncAll<true>(); pipe->Reset();
}
} // namespace DispatchDedup
