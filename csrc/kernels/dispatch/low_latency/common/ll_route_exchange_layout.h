#pragma once
#include <cstdint>
#include "ep_memory_server_dedup_count_layout.h"

namespace DispatchDedup {
// Fixed, private reservation in each bank, above combine's control region and
// below the common 2MiB payload boundary. Epoch counters use bank0 only.
constexpr uint64_t kRouteExchangeBase = 1280ULL * 1024U;
constexpr uint32_t kRouteExchangeMaxBytes = 256U * 32U * sizeof(int32_t);
constexpr uint32_t kRouteFlagBytes = 32U;
// A marker outside the valid expert-ID domain distinguishes a tail from an
// old, longer route body's IDs when B/K changes between uses of the bank.
constexpr uint32_t kRouteFlagMagic = 0x52544553U;
constexpr uint64_t kRouteSlotBytes = (kRouteExchangeMaxBytes + kRouteFlagBytes + 511U) / 512U * 512U;
constexpr uint64_t kRouteEpoch = kRouteExchangeBase;
constexpr uint64_t kRouteTx = kRouteEpoch + kLlMaxCoreNum * 512U;
constexpr uint64_t kRouteRx = kRouteTx + kRouteSlotBytes;
constexpr uint64_t kRouteExchangeEnd = kRouteRx + kRouteSlotBytes;
static_assert(kRouteExchangeEnd < kEpServerDedupControlBytes, "route exchange must not overlap payload");
static_assert(kRouteExchangeBase >= 1024U*1024U + 18432U + kLlMaxRecvCoreNum*512U,
              "route exchange must not overlap combine controls");
}
