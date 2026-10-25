#pragma once
#include <cstdint>
#ifndef DEEPEP_LL_DISPATCH_CB59
#define DEEPEP_LL_DISPATCH_CB59 0
#endif
static_assert(DEEPEP_LL_DISPATCH_CB59 >= 0 && DEEPEP_LL_DISPATCH_CB59 <= 4,
              "LL dispatch implementation must be current, cb59, cb59-stream, cb59-pair or cb59-small");
namespace DispatchDedup {
constexpr bool kLlCb59Replay = DEEPEP_LL_DISPATCH_CB59 != 0;
constexpr bool kLlCb59Stream = DEEPEP_LL_DISPATCH_CB59 == 2;
constexpr bool kLlCb59Pair = DEEPEP_LL_DISPATCH_CB59 == 3;
constexpr bool kLlCb59Small = DEEPEP_LL_DISPATCH_CB59 == 4;
constexpr uint32_t kLlDispatchImplementationId = DEEPEP_LL_DISPATCH_CB59;
constexpr const char* kLlDispatchImplementation = kLlCb59Small ? "cb59-small" : (kLlCb59Pair ? "cb59-pair" :
    (kLlCb59Stream ? "cb59-stream" : (kLlCb59Replay ? "cb59" : "current")));
constexpr const char* kLlImplementationName = kLlCb59Small ? "cb59-small-direct" : (kLlCb59Pair ? "cb59-pair-recv" :
    (kLlCb59Stream ? "cb59-stream-recv" : (kLlCb59Replay ? "cb59-replay" : "g4-small-b-parallel-prefix")));
}
