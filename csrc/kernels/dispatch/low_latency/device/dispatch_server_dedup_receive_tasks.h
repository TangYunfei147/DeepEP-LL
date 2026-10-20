#pragma once
#include "kernel_operator.h"
#include "../common/ll_receive_queue.h"
namespace DispatchDedup {
using namespace AscendC;
// A fixed-size descriptor tile; output rows always retain their global indices.
// No pending list, row reassignment or per-row ownership atomic is needed.
class ServerDedupReceiveTasks {
public:
    __aicore__ inline void Init(TPipe* pipe) {
        pipe->InitBuffer(tokenPosBuf_,kLlReceiveBatch*sizeof(uint32_t));
        tokenPos_=tokenPosBuf_.Get<uint32_t>();
    }
    __aicore__ inline void Load(GlobalTensor<uint32_t> positions,uint32_t begin,uint32_t count) {
        begin_=begin;count_=count;
        DataCopyPad(tokenPos_,positions[begin],{1U,count*4U,0U,0U,0U},{false,0U,0U,0U});
    }
    __aicore__ inline uint32_t Begin() const { return begin_; }
    __aicore__ inline uint32_t Count() const { return count_; }
    __aicore__ inline uint32_t Position(uint32_t j) { return tokenPos_.GetValue(j); }
private:
    uint32_t begin_=0U,count_=0U;
    TBuf<> tokenPosBuf_;
    LocalTensor<uint32_t> tokenPos_;
};
} // namespace DispatchDedup
