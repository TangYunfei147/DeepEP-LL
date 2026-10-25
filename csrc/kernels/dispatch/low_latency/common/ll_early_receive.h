#pragma once
#include <cstdint>
#include "ll_receive_priority.h"
namespace DispatchDedup {
#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
#define LL_EARLY_INLINE __aicore__ inline
#else
#define LL_EARLY_INLINE inline
#endif

// One policy per committed wheel, not another dispatch template dimension.
enum class LlEarlyReceivePolicy : uint32_t { Buffered = 1U, Immediate = 2U };
constexpr LlEarlyReceivePolicy kLlEarlyReceivePolicy = LlEarlyReceivePolicy::Immediate;
constexpr uint32_t kLlEarlyReceiveCadence = 2U;
constexpr uint32_t kLlEarlyReceiveQuantum = 2U;
constexpr uint32_t kLlEarlyReceiveProbe = 16U;

LL_EARLY_INLINE uint32_t LlEarlyReceiveIdBytes(uint32_t lanes) {
    return ((lanes * uint32_t(sizeof(uint32_t)) + 31U) / 32U) * 32U;
}
LL_EARLY_INLINE bool LlEarlyReceiveFits(uint64_t used, uint64_t capacity, uint32_t lanes) {
    return lanes != 0U && lanes <= 6U && used <= capacity &&
           LlEarlyReceiveIdBytes(lanes) <= capacity - used;
}

// The FH generator can still own exactly one deferred input. Once DIRECT
// generation starts, packet submissions above this target are real DIRECTs.
// Visit is called after Push, never after the final input Flush.
class LlSubmittedDirectCadence {
public:
    LL_EARLY_INLINE void Reset(uint32_t submitted, bool deferredFirstHop) {
        firstHopTarget_ = submitted + uint32_t(deferredFirstHop);
        next_ = kLlEarlyReceiveCadence;
    }
    LL_EARLY_INLINE bool Visit(uint32_t submitted) {
        if (submitted < firstHopTarget_ || submitted - firstHopTarget_ < next_) return false;
        next_ = submitted - firstHopTarget_ + kLlEarlyReceiveCadence;
        return true;
    }
private:
    uint32_t firstHopTarget_ = 0U, next_ = kLlEarlyReceiveCadence;
};

// Only the selected ready FIRST_HOPs leave pending. Pending positions remain
// unchanged through Accept, then Remove uses descending original positions.
// Advance even on a compact-only or unready window; final Run visits all tasks.
template<class Ops>
LL_EARLY_INLINE uint32_t LlTryEarlyFirstHopWindow(Ops& ops, uint32_t pending,
                                                uint32_t& cursor, uint32_t limit) {
    if (!pending || !limit) return 0U;
    if (cursor >= pending) cursor = 0U;
    const uint32_t remaining = pending - cursor;
    const uint32_t count = remaining < kLlEarlyReceiveProbe ? remaining : kLlEarlyReceiveProbe;
    const auto probe = ops.Probe(cursor, count);
    const uint32_t eligible = probe.readyMask & probe.keepFlagsMask;
    const auto order = LlMakeReceivePriority(count, eligible, eligible);
    const uint32_t accepted = order.count < limit ? order.count : limit;
    uint32_t selected = 0U;
    for (uint32_t i = 0U; i < accepted; ++i) selected |= uint32_t(1) << order.At(i);
    if (selected) {
        ops.Accept(count, selected);
        ops.Remove(cursor, count, selected);
    }
    const uint32_t after = pending - accepted;
    const uint32_t end = cursor + count;
    cursor = end < after ? end : 0U;
    return accepted;
}

#undef LL_EARLY_INLINE
} // namespace DispatchDedup
