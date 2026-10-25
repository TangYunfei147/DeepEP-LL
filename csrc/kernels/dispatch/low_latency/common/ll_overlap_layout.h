#pragma once
#include "ep_memory_server_dedup_data_layout.h"
#include "ll_overlap_stage.h"
#include "ep_memory_server_dedup_count_layout.h"
#include "ll_receive_probe_pipeline.h"
#include "ll_early_receive.h"
namespace DispatchDedup {
#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
#define LL_OVERLAP_INLINE __aicore__ inline
#else
#define LL_OVERLAP_INLINE inline
#endif
struct LlOverlapLayout {
    uint32_t sendBytes=0, countBytes=0, sendLanes=0, recvLanes=0;
    uint64_t totalBytes=0;
    bool enabled=false;
    bool recvProbePrefetch=false;
    bool earlyReceive=false;
    LL_OVERLAP_INLINE bool Async() const { return enabled && kLlOverlapExperimentStage>=3U; }
    LL_OVERLAP_INLINE bool Cooperative() const { return enabled && kLlOverlapExperimentStage>=4U; }
};
LL_OVERLAP_INLINE uint64_t LlOverlapMax(uint64_t a,uint64_t b) { return a>b?a:b; }
LL_OVERLAP_INLINE LlOverlapLayout LlMakeOverlapLayout(const EpServerDedupDataLayout& d,
    uint32_t bs,uint32_t k,uint32_t e,uint32_t ranks,uint32_t rank,uint32_t core,
    EpServerDedupInputMode mode, bool enableDedup=true) {
    LlOverlapLayout p{};
    if constexpr (kLlOverlapExperimentStage<2U) return p;
    const auto plan=LlMakeGroupCountPlan(e,ranks,rank,core);
    const uint64_t routes=DispatchDedupAlign(uint64_t(bs)*k*4,256), q=DispatchDedupAlign(uint64_t(bs)*k*4,32);
    const uint64_t records=DispatchDedupAlign(plan.count,8), reserved=records*32+(enableDedup&&plan.local&&plan.count?2*routes:0);
    const bool quant=mode==EpServerDedupInputMode::Quantize16;
    const uint64_t input=quant?d.quantCount*2:d.packedPayloadBytes;
    const uint64_t scales=d.quantCount/32, work=DispatchDedupAlign(DispatchDedupAlign(scales,32)*4+scales*2,32);
    const uint64_t tokens=(bs+kLlSendCoreNum-1)/kLlSendCoreNum;
    const uint64_t groups=(ranks/kLlRanksPerGroup-1)<k/2 ? ranks/kLlRanksPerGroup-1:k/2;
    const uint64_t sendFixed=core<kLlSendCoreNum ?
        2*input+routes+(quant?LlOverlapMax(q,work):q)+LlOverlapMax(quant?LlOverlapMax(q,work):q,DispatchDedupAlign(uint64_t(e)*4,32))+q+(quant?d.packedPayloadBytes:0)+
        2*DispatchDedupAlign(tokens*k*4,32)+DispatchDedupAlign(tokens*4,32)+DispatchDedupAlign(tokens*k*16,32)+
        LlOverlapMax(64,DispatchDedupAlign(ranks/kLlRanksPerGroup*4,32))+DispatchDedupAlign(groups*4,32) :
        2*input+q+(quant?work+d.packedPayloadBytes:0);
    const uint64_t countScratch=plan.count?4*routes+records*4:0;
    const uint64_t countStage=core==kLlSendCoreNum ? DispatchSinglePrefixStageUbBytes(e):0;
    const uint64_t complete=32+DispatchDedupAlign(e*4,32)+DispatchDedupAlign(uint64_t(e/ranks)*8,32);
    const uint64_t maxRows=uint64_t(bs)*ranks*(k<e/ranks?k:e/ranks);
    const uint64_t receiveCores=enableDedup?kLlDispatchRecvCoreNum:kLlRecvCoreNum;
    const uint64_t tasks=(maxRows+receiveCores-1)/receiveCores;
    const uint64_t recvFixed=2*DispatchDedupAlign(tasks*4,32)+16*d.blockCount*32+256+
        16*DispatchDedupAlign((d.blockCount+7)/8,32)*4+256+16*36;
    const uint64_t fixed=reserved+countStage+complete+recvFixed;
    // Solve both inequalities for max(sendFixed + L*P, countScratch).
    // Two divisions once at initialization; no lane search or per-packet plan.
    if (fixed+sendFixed+d.tokenStride*2+64>kEpServerDedupCountUbBytes ||
        fixed+countScratch+d.tokenStride+64>kEpServerDedupCountUbBytes) return p;
    const uint64_t sendLimit=(kEpServerDedupCountUbBytes-fixed-sendFixed)/(2*d.tokenStride+64);
    const uint64_t countLimit=(kEpServerDedupCountUbBytes-fixed-countScratch)/(d.tokenStride+64);
    uint64_t lanes=sendLimit<countLimit?sendLimit:countLimit;
    if (lanes>6) lanes=6; // 6/7 are reserved for input/control event IDs.
    const uint64_t send=LlOverlapMax(sendFixed+lanes*d.tokenStride,countScratch);
    p={uint32_t(send),uint32_t(countStage),uint32_t(lanes),uint32_t(lanes),
       fixed+send+lanes*(d.tokenStride+64),true};
    // The original lane count is already fixed. Use only genuine spare UB for
    // the second stable-ID snapshot; flag/reduction storage remains shared.
    if (LlReceiveProbePrefetchFits(p.totalBytes,kEpServerDedupCountUbBytes)) {
        p.recvProbePrefetch=true;
        p.totalBytes+=kLlReceiveProbeExtraBytes;
    }
    // Only S participates. Retain the already selected TX/RX lanes and use a
    // separate tiny ID FIFO; payloads occupy the original receive lanes.
    if (enableDedup && core<kLlSendCoreNum && p.Async() &&
        LlEarlyReceiveFits(p.totalBytes,kEpServerDedupCountUbBytes,p.recvLanes)) {
        p.earlyReceive=true;
        p.totalBytes+=LlEarlyReceiveIdBytes(p.recvLanes);
    }
    return p;
}
#undef LL_OVERLAP_INLINE
}
