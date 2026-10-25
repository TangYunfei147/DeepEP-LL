#pragma once
#include <cstdint>
#ifndef DEEPEP_LL_DISPATCH_CB59
#define DEEPEP_LL_DISPATCH_CB59 0
#endif
static_assert(DEEPEP_LL_DISPATCH_CB59 == 0 || DEEPEP_LL_DISPATCH_CB59 == 1,
              "LL dispatch implementation must be current or cb59");
namespace DispatchDedup {
constexpr bool kLlCb59Replay = DEEPEP_LL_DISPATCH_CB59 != 0;
constexpr uint32_t kLlDispatchImplementationId = kLlCb59Replay ? 1U : 0U;
constexpr const char* kLlDispatchImplementation = kLlCb59Replay ? "cb59" : "current";
constexpr const char* kLlImplementationName = kLlCb59Replay ? "cb59-replay" : "g4-small-b-parallel-prefix";
}
