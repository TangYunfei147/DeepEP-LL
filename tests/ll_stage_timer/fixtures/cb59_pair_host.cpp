#include <algorithm>
#include <cassert>
#include <cstdint>
#include "csrc/kernels/dispatch/low_latency/common/ep_memory_server_dedup_layout.h"

using namespace DispatchDedup;

int main()
{
    static_assert(kLlCb59Pair && !kLlCb59Stream && !kLlCb59Small, "pair is an independent selection");
    static_assert(kLlCoreProtocolVersion == 23028U, "pair identity");
    unsigned admitted = 0, fallback = 0, pairImprovesBudget = 0;
    for (uint32_t ranks : {16U,32U})
    for (uint32_t b : {1U,8U,32U,64U})
    for (uint32_t experts : {32U,384U,1024U})
    for (uint32_t k : {1U,8U,32U})
    for (uint32_t h : {256U,3584U,7168U,16000U,32768U})
    for (auto mode : {EpServerDedupInputMode::Plain16,EpServerDedupInputMode::Quantize16,
                      EpServerDedupInputMode::PrequantizedFp8Packs}) {
        EpMemoryServerDedupLayout layout{};
        if (BuildDispatchServerDedupLayout(ranks,b,h,k,experts,&layout,mode,56U)!=0) continue;
        const bool old = Cb59Replay::PacketUbFits(layout,b,k,experts,ranks,56U);
        assert(DispatchDedupPacketUbFits(layout,b,k,experts,ranks,56U,true)==old);
        if (ranks!=16U || b>32U) {
            assert(DispatchDedupPacketUbFits(layout,b,k,experts,ranks,56U,false)==old);
            ++fallback;
            continue;
        }
        const bool fit=Cb59Replay::PairUbFits(layout,b,k,experts,ranks,56U);
        assert(DispatchDedupPacketUbFits(layout,b,k,experts,ranks,56U,false)==fit);
        if (!fit) continue;
        ++admitted;
        if (!Cb59Replay::StreamUbFits(layout,b,k,experts,ranks,56U)) ++pairImprovesBudget;
        const uint64_t reserved=LlStreamAlign((uint64_t(experts)+55U)/56U,8U)*32U;
        for (uint32_t receiver=0;receiver<28U;++receiver) {
            const auto own=LlStreamRecordSlice(experts,receiver), partner=LlStreamRecordSlice(experts,receiver^1U);
            const uint64_t total=std::min<uint64_t>(layout.capacityRows,uint64_t(own.Count()+partner.Count())*b*k);
            const uint64_t rows=receiver%2U ? (total+1U)/2U : total/2U;
            const uint64_t live=reserved+LlStreamCountBufferBytes(experts,ranks,receiver)+
                LlPairExtraBufferBytes(experts,receiver)+LlStreamReceiverBufferBytes(rows,layout.data.blockCount);
            assert(live+layout.data.tokenStride+64U<=Cb59Replay::kEpServerDedupCountUbBytes);
        }
    }
    assert(admitted && fallback && pairImprovesBudget);
}
