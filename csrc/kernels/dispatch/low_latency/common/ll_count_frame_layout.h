#pragma once
#include <cstdint>

namespace DispatchDedup {
// Same E*512B reservation, physical source-major frames; logical prefixes stay
// expert-major. One frame is the intersection of a count-core and target rank.
struct LlCountFrame {
    uint64_t recordOffset, commitOffset, frameOffset;
    uint32_t firstExpert, records;
};
#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
__aicore__ inline
#else
inline
#endif
LlCountFrame CountFrame(uint32_t experts, uint32_t world, uint32_t countCores,
                        uint32_t destination, uint32_t source, uint32_t localExpert)
{
    const uint32_t local = experts / world, expert = destination * local + localExpert;
    const uint32_t base = experts / countCores, extra = experts % countCores;
    const uint32_t wideEnd = (base + 1U) * extra;
    const uint32_t owner = expert < wideEnd ? expert / (base + 1U) : extra + (expert - wideEnd) / base;
    const uint32_t begin = owner * base + (owner < extra ? owner : extra);
    const uint32_t end = begin + base + (owner < extra ? 1U : 0U);
    const uint32_t first = begin > destination * local ? begin - destination * local : 0U;
    const uint32_t last = end < (destination + 1U) * local ? end - destination * local : local;
    const uint64_t frame = (uint64_t(source) * local + first) * 512U;
    return {frame + uint64_t(localExpert - first) * 32U,
            frame + uint64_t(last - first) * 512U - 32U, frame, first, last - first};
}
}
