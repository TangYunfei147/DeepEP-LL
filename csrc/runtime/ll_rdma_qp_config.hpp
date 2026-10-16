#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>

namespace deepep {
// Native ROCE capacity in SHMEM 4dd80e2129da42bd773982c2bb3758af1ed0bba8.
// This is separate from LL's 72-core control storage capacity.
constexpr uint32_t kLlNativeRoceQpLimit = 56;

inline void validate_ll_rdma_qps(uint32_t active_aivs, bool cross_host_rdma)
{
    if (cross_host_rdma && active_aivs > kLlNativeRoceQpLimit) {
        throw std::runtime_error(
            "Cross-host RDMA LL requires one independent ROCE QP per actual AIV: actual AIV count=" +
            std::to_string(active_aivs) + ", pinned SHMEM native QP limit=" +
            std::to_string(kLlNativeRoceQpLimit) +
            "; unsupported before SHMEM initialization (no QP sharing or automatic core reduction)");
    }
}
}  // namespace deepep
