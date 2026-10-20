#pragma once
#include <cstdint>
namespace DispatchDedup {
#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
#define LL_QUEUE_INLINE __aicore__ inline
#else
#define LL_QUEUE_INLINE inline
#endif
constexpr uint32_t kLlReceiveBatch = 16U;
constexpr uint32_t kLlReceiveLeaseBatches = 8U;
constexpr uint32_t kLlReceiveDoneOffset = 512U;
constexpr uint32_t kLlReceiveAdmissionOffset = 1024U;
constexpr uint32_t kLlReceiveHeaderBytes = 1536U;
static_assert(kLlReceiveBatch<=16U && kLlReceiveBatch>0U && kLlReceiveLeaseBatches>0U,"valid ready queue grain");
struct LlReceiveQueueLayout {
    uint64_t claimBytes=0, positionOffset=0, bytes=0;
};
LL_QUEUE_INLINE uint64_t LlReceiveCapacity(uint32_t bs,uint32_t k,uint32_t experts,uint32_t ranks) {
    const uint32_t local=experts/ranks;
    return uint64_t(ranks)*bs*(k<local ? k : local);
}
LL_QUEUE_INLINE LlReceiveQueueLayout LlReceiveQueueMemory(uint64_t capacity) {
    LlReceiveQueueLayout out;
    const uint64_t batches=(capacity+kLlReceiveBatch-1U)/kLlReceiveBatch;
    out.claimBytes=(batches*4U+511U)/512U*512U;
    out.positionOffset=kLlReceiveHeaderBytes+out.claimBytes;
    out.bytes=out.positionOffset+(capacity*4U+511U)/512U*512U;
    return out;
}
LL_QUEUE_INLINE uint32_t LlReceiveValidMask(uint32_t count) { return (uint32_t(1)<<count)-1U; }

// Atomics supplies Pointer/Add/Cas. The same scheduler is exercised with host
// std::atomic in CPU tests; native operations remain local-GM scalar atomics.
template<class Atomics> class LlReadyReceiveQueue {
public:
    using Pointer=typename Atomics::Pointer;
    LL_QUEUE_INLINE void Init(Pointer base,uint32_t rows,uint32_t workers) {
        scan_=remaining_=poll_=0U; allClaimed_=false; workers_=workers;
        base_=base; batches_=(rows+kLlReceiveBatch-1U)/kLlReceiveBatch;
    }
    // Limit tiny queues by batch count, using arrival order, not physical core
    // IDs. Admission is separate from the cyclic probe cursor, so a cold first
    // scan cannot exclude later workers by advancing beyond the queue length.
    LL_QUEUE_INLINE bool Enter() {
        if (!batches_ || AllClaimed()) return false;
        return batches_>=workers_ || Atomics::Add(base_+kLlReceiveAdmissionOffset/4U,1U)<batches_;
    }
    LL_QUEUE_INLINE bool AllClaimed() const {
        return allClaimed_ || Atomics::Add(base_+kLlReceiveDoneOffset/4U,0U)==batches_;
    }
    LL_QUEUE_INLINE bool Next(uint32_t& batch) {
        if (!batches_ || allClaimed_) return false;
        if (++poll_==kLlReceiveLeaseBatches) {
            poll_=0U;
            if (AllClaimed()) return false;
        }
        if (!remaining_) {
            // Tickets select cyclic probe windows, not exclusive task owners.
            // uint32 wrap only changes the scan phase; it cannot lose a task.
            scan_=Atomics::Add(base_,kLlReceiveLeaseBatches)%batches_;
            remaining_=kLlReceiveLeaseBatches;
        }
        batch=scan_++; if (scan_==batches_) scan_=0U;
        --remaining_;
        return true;
    }
    // Never own a not-ready row. At most two CAS attempts, then move on. This
    // avoids waiting cycles between unclaimed FIRST_HOP and terminal packets.
    LL_QUEUE_INLINE uint32_t Claim(uint32_t batch,uint32_t ready,uint32_t valid,uint32_t& observed) {
        ready&=valid;
        observed=0U;
        if (!ready) return 0U;
        Pointer word=base_+kLlReceiveHeaderBytes/4U+batch;
        uint32_t old=Atomics::Cas(word,0U,ready), acquired=0U;
        observed=old;
        if (!old) { acquired=ready; observed=ready; }
        else if (ready&~old) {
            const uint32_t current=Atomics::Cas(word,old,old|ready);
            observed=current;
            if (current==old) { acquired=ready&~old; observed=old|ready; }
        }
        if (acquired && ((old|acquired)&valid)==valid)
            allClaimed_=Atomics::Add(base_+kLlReceiveDoneOffset/4U,1U)+1U==batches_;
        return acquired;
    }
private:
    Pointer base_{};
    uint32_t batches_=0U,scan_=0U,remaining_=0U,poll_=0U,workers_=0U;
    bool allClaimed_=false;
};
#undef LL_QUEUE_INLINE
} // namespace DispatchDedup
