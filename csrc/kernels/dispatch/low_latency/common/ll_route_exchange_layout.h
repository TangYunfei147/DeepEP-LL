#pragma once
#include <cstdint>
#include "ep_memory_server_dedup_count_layout.h"

namespace DispatchDedup {
// Fixed, private reservation in each bank, above combine's control region and
// below the common 2MiB payload boundary. Epoch counters use bank0 only.
constexpr uint64_t kRouteExchangeBase = 1280ULL * 1024U;
constexpr uint32_t kRouteExchangeMaxBytes = 256U * 32U * sizeof(int32_t);
constexpr uint64_t kRouteEpoch = kRouteExchangeBase;
constexpr uint64_t kRouteTx = kRouteEpoch + kLlMaxCoreNum * 512U;
constexpr uint64_t kRouteRx = kRouteTx + kRouteExchangeMaxBytes;
constexpr uint64_t kRouteSequence = kRouteRx + kRouteExchangeMaxBytes;
constexpr uint64_t kRouteCommit = kRouteSequence + 512U;
constexpr uint64_t kRouteAck = kRouteCommit + 512U;
constexpr uint64_t kRouteLastSent = kRouteAck + 512U;
constexpr uint64_t kRouteReadDone = kRouteLastSent + 512U;
constexpr uint64_t kRouteExchangeEnd = kRouteReadDone + kLlMaxCoreNum * 512U;
static_assert(kRouteExchangeEnd < kEpServerDedupControlBytes, "route exchange must not overlap payload");
static_assert(kRouteExchangeBase >= 1024U*1024U + 18432U + kLlMaxRecvCoreNum*512U,
              "route exchange must not overlap combine controls");
}
