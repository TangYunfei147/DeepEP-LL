#pragma once
#include <cstdint>
#ifndef DEEPEP_LL_DISPATCH_CB59
#define DEEPEP_LL_DISPATCH_CB59 0
#endif
static_assert(DEEPEP_LL_DISPATCH_CB59 >= 0 && DEEPEP_LL_DISPATCH_CB59 <= 2,
              "LL dispatch implementation must be current, cb59 or cb59-stream");
namespace DispatchDedup {
constexpr bool kLlCb59Replay = DEEPEP_LL_DISPATCH_CB59 != 0;
constexpr bool kLlCb59Stream = DEEPEP_LL_DISPATCH_CB59 == 2;
constexpr uint32_t kLlDispatchImplementationId = DEEPEP_LL_DISPATCH_CB59;
constexpr const char* kLlDispatchImplementation = kLlCb59Stream ? "cb59-stream" : (kLlCb59Replay ? "cb59" : "current");
constexpr const char* kLlImplementationName = kLlCb59Stream ? "cb59-stream-recv" : (kLlCb59Replay ? "cb59-replay" : "g4-small-b-parallel-prefix");
}
