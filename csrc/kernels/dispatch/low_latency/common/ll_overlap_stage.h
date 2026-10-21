#pragma once
#include <cstdint>
namespace DispatchDedup {
// Deliberately committed per A/B revision, not an environment heuristic.
constexpr uint32_t kLlOverlapExperimentStage = 2U;
static_assert(kLlOverlapExperimentStage>=1 && kLlOverlapExperimentStage<=4,"invalid overlap A/B stage");
}
