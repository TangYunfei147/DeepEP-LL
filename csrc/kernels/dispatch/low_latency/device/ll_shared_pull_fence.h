#pragma once
#include "kernel_operator.h"
#include "../common/ep_memory_server_dedup_count_layout.h"
namespace DispatchDedup {
// All local AIV reads finish before core0 publishes its monotonically increasing
// completion epoch. Other ranks may observe a later epoch, but never a stale
// equal parity. Kernel completion retains the shared packet until every rank
// has finished reading this invocation. This prototype deliberately pays for
// that collective; the mode1 control measures the same cost without pull.
__aicore__ inline void LlSharedPullFinish(const __gm__ CommArgs* args,AscendC::TPipe* pipe) {
    using namespace AscendC;
    static_assert(kLlSharedPullFenceOffset+32U<=kEpServerDedupControlPrefixBytes,"fence bounds");
    SyncAll<true>();
    if(GetBlockIdx()!=0U) return;
    pipe->Reset();
    TBuf<> scratch;pipe->InitBuffer(scratch,32U);
    auto state=scratch.Get<uint32_t>();
    const auto readEvent=pipe->FetchEventID(HardEvent::MTE2_S);
    const auto scalarEvent=pipe->FetchEventID(HardEvent::S_MTE3);
    const auto writeEvent=pipe->FetchEventID(HardEvent::MTE3_S);
    GlobalTensor<uint32_t> counter;
    counter.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(args->peerMems[args->rank]+kLlSharedPullFenceOffset));
    counter.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    DataCopy(state,counter,8U);
    SetFlag<HardEvent::MTE2_S>(readEvent);WaitFlag<HardEvent::MTE2_S>(readEvent);
    const uint32_t previous=state.GetValue(0U);
    if(previous==0xffffffffU) { Trap();return; }
    const uint32_t epoch=previous+1U;
    state.SetValue(0U,epoch);
    SetFlag<HardEvent::S_MTE3>(scalarEvent);WaitFlag<HardEvent::S_MTE3>(scalarEvent);
    DataCopy(counter,state,8U);
    SetFlag<HardEvent::MTE3_S>(writeEvent);WaitFlag<HardEvent::MTE3_S>(writeEvent);
    for(uint32_t peer=0;peer<uint32_t(args->rankSize);++peer) {
        counter.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(args->peerMems[peer]+kLlSharedPullFenceOffset));
        counter.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        do {
            DataCopy(state,counter,8U);
            SetFlag<HardEvent::MTE2_S>(readEvent);WaitFlag<HardEvent::MTE2_S>(readEvent);
        } while(state.GetValue(0U)<epoch);
    }
}
}
