#pragma once
#include "comm_args.h"
#include "ll_group_policy.h"
namespace LlRdma {
constexpr uint32_t kBackend = 1U;
constexpr uint32_t kCoreBytes = 128U * 1024U;
// QP count is queried active AIV count. Every physical core owns QP core_id.
// Native SHMEM capacity validation remains in Session before initialization.
// ABI v3 tail: addresses[0]=unbiased workspace, [1]=per-core staging,
// [2]=exclusive staging end (actual AIV count, independent of QP count).
// config[0]=backend, [1]=QP count. Remote peerMems are NULL in RDMA mode.
}
