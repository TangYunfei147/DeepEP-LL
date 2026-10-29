#pragma once
#include "ll_count_frame_layout.h"

namespace DispatchDedup {
// Support up to 1024 records even when the dynamic split leaves few R cores.
// Each run is one expert row across source ranks; world>=32.
constexpr uint32_t kLlCountReadMaxRuns = 33U;
struct LlCountReadRun {
    uint64_t recordOffset = 0, commitOffset = 0;
    uint32_t first = 0, records = 0; // first is relative to this receiver's UB
};
struct LlCountReadPlan {
    LlCountReadRun runs[kLlCountReadMaxRuns]{};
    uint32_t size = 0;
    uint32_t commitSourceStrideBlocks = 0;
    uint32_t sourceStrideBlocks = 0; // gap in 32B units, excluding the copied block
};
#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
__aicore__ inline
#else
inline
#endif
bool BuildCountReadPlan(uint32_t experts, uint32_t world, uint32_t countCores,
                       uint32_t destination, uint32_t first, uint32_t count, LlCountReadPlan& out)
{
    out.size = 0;
    if (world < 32U || world > 256U || experts < world || experts > 1024U || experts % world ||
        !countCores || countCores > 72U || destination >= world || first > experts ||
        count > experts - first || count > 1024U) return false;
    out.commitSourceStrideBlocks = experts / world - 1U;
    out.sourceStrideBlocks = (experts / world) * 16U - 1U;
    for (uint32_t consumed = 0; consumed < count;) {
        if (out.size == kLlCountReadMaxRuns) return false;
        const uint32_t logical = first + consumed;
        const uint32_t source = logical % world;
        const uint32_t available = world - source;
        const uint32_t length = count - consumed < available ? count - consumed : available;
        const auto frame = CountFrame(experts, world, countCores, destination, source, logical / world);
        out.runs[out.size++] = {frame.recordOffset, frame.commitOffset, consumed, length};
        consumed += length;
    }
    return true;
}
}
