#pragma once
#include <cstdint>
namespace DispatchDedup {
// Deliberately committed per A/B revision, not an environment heuristic.
constexpr uint32_t kLlOverlapExperimentStage = 5U;
static_assert(kLlOverlapExperimentStage==5U,"Stage5 source requires stage5; checkout historical commits for stages1..4");
// Prefix owns MTE2_S 5 only before receive starts; sender inputs own 6/7.
constexpr uint32_t kLlPrefixLoadEvent = 5U;
}
