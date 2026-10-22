#pragma once
#include "ll_transport.h"
namespace LlTransport {
// One contiguous body plus one separate 32B commit. The receiver must not read
// body ready/count words until it has acquired the frame commit.
__aicore__ inline void CountFrame(const __gm__ Args* args, uint32_t peer,
                                 uint64_t frameOffset, uint64_t commitOffset,
                                 LocalTensor<uint8_t> records, uint32_t count, LocalTensor<uint8_t> scratch) {
    Check(args, peer, frameOffset, uint64_t(count)*32U);
    Check(args, peer, commitOffset, 32U);
    auto dst=Address(args,peer,frameOffset);
    auto commit=Address(args,peer,commitOffset);
    if (!RemoteWrite(args,peer)) {
        DataCopy(Tensor(dst,Rdma(args)),records,count*32U);
        Sync<HardEvent::MTE3_S>(7U);
        DataCopy(Tensor(commit,Rdma(args)),records,32U);
        return;
    }
    auto stage=Stage(args,count*32U);
    DataCopy(Tensor(stage,true),records,count*32U);
    Sync<HardEvent::MTE3_S>(7U);
    Put(args,peer,dst,stage,count*32U,scratch);
    Quiet(peer,scratch);
    Put(args,peer,commit,stage,32U,scratch);
    Quiet(peer,scratch);
}
}
