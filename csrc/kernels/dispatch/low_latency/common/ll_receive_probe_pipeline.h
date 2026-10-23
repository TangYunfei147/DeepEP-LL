#pragma once
#include <cstdint>
namespace DispatchDedup {
#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
#define LL_PROBE_INLINE __aicore__ inline
#else
#define LL_PROBE_INLINE inline
#endif

constexpr uint32_t kLlReceiveProbeBatch = 16U;
constexpr uint32_t kLlReceiveProbeExtraBytes = kLlReceiveProbeBatch * sizeof(uint32_t);

// Decide AFTER selecting the original payload lanes. Optional flag lookahead
// must never take a payload lane or invalidate a previously accepted shape.
LL_PROBE_INLINE bool LlReceiveProbePrefetchFits(uint64_t used, uint64_t capacity) {
    return used <= capacity && capacity - used >= kLlReceiveProbeExtraBytes;
}

struct LlReceiveProbeResult {
    uint32_t readyMask = 0U;
    uint32_t keepFlagsMask = 0U;
};

// At most two adjacent snapshots. Pending is immutable until both snapshots
// have been consumed; their stable task IDs and original positions stay valid.
// B's flag DMA/reduction overlaps A consumption. Only one probe is outstanding,
// so flags/sums/work are reused after A's scalar decode; only IDs need two slots.
// The exact pending permutation differs from two old TryBatch calls: those
// compacted A before choosing B. Ownership and once-only consumption are retained.
template<class Ops>
LL_PROBE_INLINE bool LlRunReceiveProbePair(Ops& ops, uint32_t pending, uint32_t& scan) {
    if (!pending) return false;
    if (scan >= pending) scan = 0U;
    const uint32_t remaining = pending - scan;
    const uint32_t firstCount = remaining < kLlReceiveProbeBatch ? remaining : kLlReceiveProbeBatch;
    const uint32_t tail = remaining - firstCount;
    const uint32_t secondCount = tail < kLlReceiveProbeBatch ? tail : kLlReceiveProbeBatch;

    ops.StartProbe(scan, firstCount, 0U);
    const auto first = ops.FinishProbe(firstCount);
    // Queue A's first payload lanes before B flags, so lookahead does not get
    // ahead of the first current payload on the shared MTE2 engine.
    auto active = ops.PrimeReady(0U, firstCount, first);
    if (secondCount) ops.StartProbe(scan + firstCount, secondCount, kLlReceiveProbeBatch);
    ops.ConsumeReady(active);

    LlReceiveProbeResult second;
    if (secondCount) {
        second = ops.FinishProbe(secondCount);
        auto next = ops.PrimeReady(kLlReceiveProbeBatch, secondCount, second);
        ops.ConsumeReady(next);
        ops.RemoveProbe(scan + firstCount, secondCount, second.readyMask);
    }
    ops.RemoveProbe(scan, firstCount, first.readyMask);
    // Empty/partly stale snapshots must advance rather than wait for a terminal
    // whose FIRST_HOP may be elsewhere. A nonempty pair rechecks swapped tails.
    const bool ready = (first.readyMask | second.readyMask) != 0U;
    if (!ready) scan += firstCount + secondCount;
    return ready;
}

#undef LL_PROBE_INLINE
} // namespace DispatchDedup
