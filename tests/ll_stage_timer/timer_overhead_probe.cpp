// Standalone timer microbenchmark: no LL traffic, SDK, or protocol changes.
#include "kernel_operator.h"
#ifndef LL_TIMER_PROBE_NAME
#error Build through timer_overhead_build.py
#endif
#define TP_JOIN_(a,b) a##b
#define TP_JOIN(a,b) TP_JOIN_(a,b)
#define TP_SYMBOL(suffix) TP_JOIN(LL_TIMER_PROBE_NAME,suffix)
// Keep baseline/current DeviceTimer types distinct when linking all four TUs.
#define LlStageTimer TP_SYMBOL(_timer)
#include "ll_stage_timer_device.h"
using namespace AscendC;
using namespace LlStageTimer;
static_assert(KERNEL==0 && INIT==50 && SEND_SETUP==81 && Q_SEED_SUBMIT==82 &&
    Q_LOCAL_UPDATE==84 && TOKEN_INPUT==16 && TOKEN_INPUT_WAIT==64 &&
    TOKEN_METADATA==18 && TOKEN_PACK==20 && TOKEN_SEND_ISSUE==21 &&
    SEND_RETIRE_WAIT==87 && COUNT_COMPUTE==22 && COUNT_SEND_ISSUE==23 &&
    WAIT_RECV_COUNTS==6 && BUILD_RECV_CUMSUM==10 && PREFIX_COMPLETE==78 &&
    RX_FLAG_ISSUE==88 && RX_FLAG_REDUCE==90 && RX_FLAG_DECODE==91 &&
    RX_METADATA==94 && RX_PAYLOAD_OUTPUT==95 && RX_META_OUTPUT==96,
    "Update the Host checksum oracle if stage identifiers change");

// An observable scalar recurrence. It is identical in all four builds.
static __aicore__ inline uint64_t ProbeStep(DeviceTimer& timer, Stage stage, uint64_t value) {
#ifndef LL_TIMER_PROBE_SPARSE
    const uint64_t begin=timer.Begin(stage);
#endif
    value=value*1664525ULL+uint32_t(stage)+1013904223ULL;
#ifndef LL_TIMER_PROBE_SPARSE
    timer.End(stage,begin);
#endif
    return value;
}

extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__ void TP_SYMBOL(_kernel)(
    GM_ADDR timerAddress, GM_ADDR result, uint32_t pattern, uint32_t repeats) {
    // Ascend950/C310 uses native SyncAll; the SDK FFTS setter is a no-op on this target.
    const uint64_t entry=GetSystemCycle();
    // Intentionally before INIT, exactly as in the production entry.
    DeviceTimer timer;
    timer.Init(timerAddress,1U,1U);
#ifdef LL_TIMER_PROBE_SPARSE
    timer.MarkAt<ENTRY>(entry);
#endif
    const uint64_t initialized=GetSystemCycle();
#ifndef LL_TIMER_PROBE_SPARSE
    const uint64_t init=timer.Begin(INIT);
#endif
    uint64_t checksum=uint64_t(GetBlockIdx())+1U;
#ifndef LL_TIMER_PROBE_SPARSE
    timer.End(INIT,init);
#else
    timer.Mark<INIT_DONE>();
#endif
    const uint64_t beforeKernel=GetSystemCycle();
#ifndef LL_TIMER_PROBE_SPARSE
    const uint64_t kernel=timer.Begin(KERNEL);
#endif
    if(pattern==1U) {
#ifdef LL_TIMER_PROBE_SPARSE
        timer.Mark<COUNTS_SENT>();
        if(GetBlockIdx()<DispatchDedup::kLlSendCoreNum) {
            timer.Mark<SEND_PREPARED>();timer.Mark<FIRST_SEND>();
        } else {
            timer.Mark<COUNTS_RECEIVED>();timer.Mark<PREFIX_LOCAL_DONE>();
            timer.Mark<PREFIX_READY>();timer.Mark<RECV_PREPARED>();
            timer.Mark<FIRST_RECV_PROBE>();timer.Mark<FIRST_RECV_READY>();
        }
#endif
        for(uint32_t i=0;i<repeats;++i) {
            if(GetBlockIdx()<DispatchDedup::kLlSendCoreNum) {
                checksum=ProbeStep(timer,SEND_SETUP,checksum);
                checksum=ProbeStep(timer,Q_SEED_SUBMIT,checksum);
                checksum=ProbeStep(timer,Q_LOCAL_UPDATE,checksum);
                checksum=ProbeStep(timer,TOKEN_INPUT,checksum);
                checksum=ProbeStep(timer,TOKEN_INPUT_WAIT,checksum);
                checksum=ProbeStep(timer,TOKEN_METADATA,checksum);
                checksum=ProbeStep(timer,TOKEN_PACK,checksum);
                checksum=ProbeStep(timer,TOKEN_SEND_ISSUE,checksum);
                checksum=ProbeStep(timer,SEND_RETIRE_WAIT,checksum);
            } else {
                checksum=ProbeStep(timer,COUNT_COMPUTE,checksum);
                checksum=ProbeStep(timer,COUNT_SEND_ISSUE,checksum);
                checksum=ProbeStep(timer,WAIT_RECV_COUNTS,checksum);
                checksum=ProbeStep(timer,BUILD_RECV_CUMSUM,checksum);
                checksum=ProbeStep(timer,PREFIX_COMPLETE,checksum);
                checksum=ProbeStep(timer,RX_FLAG_ISSUE,checksum);
                checksum=ProbeStep(timer,RX_FLAG_REDUCE,checksum);
                checksum=ProbeStep(timer,RX_FLAG_DECODE,checksum);
                checksum=ProbeStep(timer,RX_METADATA,checksum);
                checksum=ProbeStep(timer,RX_PAYLOAD_OUTPUT,checksum);
                checksum=ProbeStep(timer,RX_META_OUTPUT,checksum);
            }
        }
#ifdef LL_TIMER_PROBE_SPARSE
        if(GetBlockIdx()<DispatchDedup::kLlSendCoreNum) timer.Mark<SEND_DONE>();
        else timer.Mark<RECV_DONE>();
#endif
    }
#ifndef LL_TIMER_PROBE_SPARSE
    timer.End(KERNEL,kernel);
#else
    timer.Mark<WORK_DONE>();
#endif
    const uint64_t afterKernel=GetSystemCycle();
    if(timer.Enabled()) {
#ifndef LL_TIMER_PROBE_SPARSE
        SyncAll<true>();
#endif
        timer.Flush();
    }
    const uint64_t afterFlush=GetSystemCycle();
    auto* words=reinterpret_cast<__gm__ uint64_t*>(result)+GetBlockIdx()*16U;
    words[0]=entry;words[1]=initialized;words[2]=beforeKernel;words[3]=afterKernel;
    words[4]=afterFlush;words[5]=checksum;words[6]=pattern;words[7]=repeats;
    GlobalTensor<uint32_t> output;
    output.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(words),32U);
    DataCacheCleanAndInvalid<uint32_t,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(output);
}

extern "C" void TP_SYMBOL(_launch)(void* stream,GM_ADDR timerAddress,GM_ADDR result,
                                    uint32_t pattern,uint32_t repeats) {
    TP_SYMBOL(_kernel)<<<56,nullptr,stream>>>(timerAddress,result,pattern,repeats);
}
extern "C" void TP_SYMBOL(_info)(uint32_t* info) {
    info[0]=kSchema;info[1]=kWordsPerCore;info[2]=kMetrics;
    info[3]=DispatchDedup::kLlSendCoreNum;
#ifdef DEEPEP_LL_STAGE_TIMER
    info[4]=1U;
#else
    info[4]=0U;
#endif
}
