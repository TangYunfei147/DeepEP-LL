// Compile-time diagnostics only; keep all switches at zero for production.
#pragma once
#include "kernel_operator.h"
#include "ep_memory_server_dedup_count_layout.h"

// Edit these defaults, rebuild the production wheel, then restart every rank.
#ifndef LL_DISPATCH_DEBUG_STAGE
#define LL_DISPATCH_DEBUG_STAGE 0
#endif
#ifndef LL_DISPATCH_DEBUG_COUNTS
#define LL_DISPATCH_DEBUG_COUNTS 0
#endif
#ifndef LL_DISPATCH_DEBUG_PREFIX
#define LL_DISPATCH_DEBUG_PREFIX 0
#endif
#ifndef LL_DISPATCH_DEBUG_SEND
#define LL_DISPATCH_DEBUG_SEND 0
#endif
#ifndef LL_DISPATCH_DEBUG_RECV
#define LL_DISPATCH_DEBUG_RECV 0
#endif
#ifndef LL_DISPATCH_DEBUG_RANK
#define LL_DISPATCH_DEBUG_RANK -1  // -1: all ranks
#endif
#ifndef LL_DISPATCH_DEBUG_CORE
#define LL_DISPATCH_DEBUG_CORE -2  // -2: core0 + first receiver; -1: all; >=0: exact core
#endif
#ifndef LL_DISPATCH_DEBUG_POLL_EVERY
#define LL_DISPATCH_DEBUG_POLL_EVERY 65536U
#endif
#ifndef LL_DISPATCH_DEBUG_MAX_POLLS
#define LL_DISPATCH_DEBUG_MAX_POLLS 8U
#endif
#ifndef LL_DISPATCH_DEBUG_ITEMS
#define LL_DISPATCH_DEBUG_ITEMS 4U
#endif
#ifndef LL_DISPATCH_DEBUG_PACKETS
#define LL_DISPATCH_DEBUG_PACKETS 2U
#endif

#if LL_DISPATCH_DEBUG_STAGE || LL_DISPATCH_DEBUG_COUNTS || LL_DISPATCH_DEBUG_PREFIX || \
    LL_DISPATCH_DEBUG_SEND || LL_DISPATCH_DEBUG_RECV
#if defined(ASCENDC_DUMP) && ASCENDC_DUMP == 0
#error "LL dispatch diagnostics require ASCENDC_DUMP not to be zero"
#endif
static_assert(LL_DISPATCH_DEBUG_POLL_EVERY > 0, "debug polling interval must be positive");
namespace DispatchDedup {
__aicore__ inline bool LlDispatchDebugSelected(int rank, uint32_t core)
{
    return (LL_DISPATCH_DEBUG_RANK < 0 || rank == LL_DISPATCH_DEBUG_RANK) &&
           (LL_DISPATCH_DEBUG_CORE == -1 ||
            (LL_DISPATCH_DEBUG_CORE == -2 && (core == 0U || core == kEpServerDedupSendCoreNum)) ||
            (LL_DISPATCH_DEBUG_CORE >= 0 && core == uint32_t(LL_DISPATCH_DEBUG_CORE)));
}
// Bounded samples: first iteration, then every N completed polling iterations.
// Counters stop once the budget is exhausted; they are not a timeout.
__aicore__ inline bool LlDispatchDebugSample(uint32_t& polls, uint32_t& logs)
{
    if (logs >= LL_DISPATCH_DEBUG_MAX_POLLS) return false;
    const bool emit = polls == 0U;
    if (++polls == LL_DISPATCH_DEBUG_POLL_EVERY) polls = 0U;
    if (emit) ++logs;
    return emit;
}
}
#define LL_DLOG(enabled, rank, core, tag, a, b, c, d) do { \
    if ((enabled) && DispatchDedup::LlDispatchDebugSelected(int(rank), uint32_t(core))) { \
        AscendC::printf("[LLD] r=%d c=%u " tag " line=%u a=%llu b=%llu c=%llu d=%llu\n", \
            int(rank), uint32_t(core), uint32_t(__LINE__), \
            static_cast<unsigned long long>(a), static_cast<unsigned long long>(b), \
            static_cast<unsigned long long>(c), static_cast<unsigned long long>(d)); \
    } \
} while (0)
#else
// Disabled instrumentation does not evaluate arguments or read device state.
#define LL_DLOG(...) ((void)0)
#endif
