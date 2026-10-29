#pragma once
#include <cstdint>

namespace DispatchDedup {
struct LlCountAssignment {
    uint32_t first, end, remoteRank, alignedRecords, remoteRecords;
};
#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
__aicore__ inline
#else
inline
#endif
LlCountAssignment CountAssignment(uint32_t rank, uint32_t world, uint32_t experts,
                                   uint32_t cores, uint32_t core, bool allCore, bool directRanks)
{
    const uint32_t countCores = allCore ? cores : cores / 2U;
    uint32_t first = 0U, end = 0U;
    if (allCore || core >= cores / 2U) {
        const uint32_t worker = allCore ? core : core - cores / 2U;
        const uint32_t base = experts / countCores, extra = experts % countCores;
        first = worker * base + (worker < extra ? worker : extra);
        end = first + base + (worker < extra ? 1U : 0U);
    }
    if (directRanks) {
        const uint32_t hostFirst = (rank / 16U) * (experts / 2U);
        const uint32_t hostEnd = hostFirst + experts / 2U;
        first = first < hostFirst ? hostFirst : first;
        end = end > hostEnd ? hostEnd : end;
        if (end < first) end = first;
    }
    const uint32_t records = end > first ? (experts + countCores - 1U) / countCores : 0U;
    const uint32_t remote = directRanks && core < 16U ? ((rank / 16U) ^ 1U) * 16U + core : world;
    const uint32_t remoteRecords = remote < world ? (experts / world + 1U + 7U) / 8U * 8U : 0U;
    return {first, end, remote, (records + 7U) / 8U * 8U, remoteRecords};
}
}
