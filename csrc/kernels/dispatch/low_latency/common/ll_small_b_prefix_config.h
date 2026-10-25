#pragma once
#include "ll_core_config.h"
#ifndef DEEPEP_LL_SMALL_B_PREFIX_CORES
#define DEEPEP_LL_SMALL_B_PREFIX_CORES 0
#endif
namespace DispatchDedup {
static_assert(DEEPEP_LL_SMALL_B_PREFIX_CORES >= 0 && DEEPEP_LL_SMALL_B_PREFIX_CORES <= DEEPEP_LL_RECV_CORES,
              "LL small-B prefix producers must be 0 or in 1..receive cores");
// Zero preserves the baseline identity. Only EP16 small direct count/prefix
// work selects a smaller producer set; receive ownership and wire stay intact.
constexpr uint32_t kLlSmallBPrefixCores = DEEPEP_LL_SMALL_B_PREFIX_CORES;
static_assert(!kLlCb59Replay || kLlSmallBPrefixCores == 0U,
              "cb59 replay has no configurable P; unset DEEPEP_LL_SMALL_B_PREFIX_CORES or use 0");
}
