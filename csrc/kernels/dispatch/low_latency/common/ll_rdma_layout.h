#pragma once
#include "comm_args.h"
#include "ll_group_policy.h"
namespace LlRdma {
constexpr uint32_t kBackend = 1U;
constexpr uint32_t kCoreBytes = 128U * 1024U;
constexpr uint32_t kQpCount = 8U;
constexpr uint32_t kCombineDirectQps = 4U;
constexpr uint32_t kCombinePartialQps = 4U;
// ABI v3 tail: addresses[0]=unbiased workspace, [1]=per-core staging,
// [2]=exclusive staging end (actual AIV count, independent of QP count).
// config[0]=backend, [1]=QP count. Remote peerMems are NULL in RDMA mode.
}
