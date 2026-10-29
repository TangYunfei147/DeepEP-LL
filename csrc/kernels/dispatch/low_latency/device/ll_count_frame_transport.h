#pragma once
#include "ll_transport.h"
namespace LlTransport {
// Per-core CQ bookkeeping, independent of receiver publication. A payload
// quiet on the same peer/QP also retires the earlier count request.
class CountRetirement {
    uint32_t pending_[8]{};
public:
    __aicore__ inline void Init(const __gm__ Args*) { for (uint32_t i=0;i<8;++i) pending_[i]=0; }
    __aicore__ inline void Posted(uint32_t peer) { pending_[peer/32] |= 1U<<(peer%32); }
    __aicore__ inline bool Pending() const { for (uint32_t i=0;i<8;++i) if(pending_[i]) return true; return false; }
    __aicore__ inline void Finish(LocalTensor<uint8_t> scratch) {
        for(uint32_t i=0;i<8;++i) {
            for(uint32_t b=0;b<32;++b) if(pending_[i]&(1U<<b)) Quiet(i*32+b,scratch);
            pending_[i]=0;
        }
    }
};

// Count TX is isolated in each symmetric bank, outside count/prefix, combine
// controls and the payload region. One disjoint slot per destination rank.
constexpr uint64_t kCountTxBase = 1280ULL * 1024U;
static_assert(kCountTxBase >= 1024U * 1024U + 18432U + 36U * 512U,
              "count TX above existing combine controls");
static_assert(kCountTxBase + (1024U + 256U) * 32U < 2U * 1024U * 1024U, "count TX below payload");
// Remote rank-owned counts only. records includes a fully initialized tail
// record with ready=1 even for zeros. CQ retirement is deferred to the role's
// existing payload quiet or final core completion, never required by SendCnt.
__aicore__ inline void DirectCountFrame(const __gm__ Args* args, uint32_t peer,
                                       uint64_t frameOffset, LocalTensor<uint8_t> records,
                                       uint32_t count, LocalTensor<uint8_t> scratch, uint64_t window = 0U) {
    const uint32_t bytes = (count + 1U) * 32U;
    Check(args, peer, frameOffset, bytes);
    auto stage = args->peerMems[args->rank] + window + kCountTxBase + uint64_t(peer) * bytes;
    DataCopy(Tensor(stage,true), records, bytes);
    Sync<HardEvent::MTE3_S>(7U);
    Put(args,peer,Address(args,peer,frameOffset),stage,bytes,scratch);
}
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
