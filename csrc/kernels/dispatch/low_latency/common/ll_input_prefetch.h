#pragma once
#include <cstdint>

namespace DispatchDedup {
#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
#define LL_INPUT_INLINE __aicore__ inline
#else
#define LL_INPUT_INLINE inline
#endif
struct LlSendPacketTask {
    uint32_t token=0U,firstK=0U,expert=0U,q=0U,mask=0U;
};

// One dequeued current input + one queued next input. Queue occupancy is <=1,
// while the two underlying buffers can both be owned. No full task-list copy.
class LlOnePacketInput {
public:
    LL_INPUT_INLINE void Reset() { pending_=false; }
    LL_INPUT_INLINE bool HasPending() const { return pending_; }
    template<class Ops>
    LL_INPUT_INLINE void Push(Ops& ops,const LlSendPacketTask& next) {
        ops.Issue(next);
        if (pending_) ops.Consume(task_);
        // Dequeue only AFTER current compute/pack has been issued: its MTE2_V
        // dependency must not make current compute wait for the next load.
        ops.Take();
        task_=next;
        pending_=true;
    }
    template<class Ops>
    LL_INPUT_INLINE void Flush(Ops& ops) {
        if (pending_) ops.Consume(task_);
        pending_=false;
    }
private:
    LlSendPacketTask task_{};
    bool pending_=false;
};
#undef LL_INPUT_INLINE
} // namespace DispatchDedup
