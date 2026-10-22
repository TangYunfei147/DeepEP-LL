#pragma once
#include "ll_stage_timer_layout.h"
#ifndef DEEPEP_LL_TIMER_PROFILE_ID
#define DEEPEP_LL_TIMER_PROFILE_ID 0
#endif
#if defined(DEEPEP_LL_TIMER_DEVICE_CONTEXT) || (defined(__CCE__) && defined(__CCE_IS_AICORE__))
#define LL_PROFILE_INLINE __aicore__ inline
#else
#define LL_PROFILE_INLINE inline
#endif
namespace LlStageTimer {
constexpr uint32_t kProfileId = DEEPEP_LL_TIMER_PROFILE_ID;
static_assert(kProfileId < 6U, "invalid LL timer profile");
constexpr const char* kProfileName = kProfileId == 0 ? "full" : kProfileId == 1 ? "overview" :
    kProfileId == 2 ? "prepare" : kProfileId == 3 ? "prefix" : kProfileId == 4 ? "send" : "recv";
// These few milestones/counters remain in every diagnostic build. No barriers
// are added. A missing counter means no call, not a measured zero-duration task.
LL_PROFILE_INLINE constexpr bool ProfileEnabled(Stage stage) {
    if (kProfileId == 0U) return true;
    switch (stage) {
        case INIT: case KERNEL: case TIMER_SETUP: case FINAL_DRAIN:
        case BEFORE_FIRST_SEND: case BEFORE_FIRST_RECV_POLL: case SEND_SUBMIT_END:
        case TX_PACKETS: case RX_PACKETS: case RX_READY: case RX_PROBED:
        case RX_EMPTY_POLLS: case RELAY_PACKETS: case OVERLAP_PATH:
        case SERIAL_FALLBACK: case SEND_LANES: case RECV_LANES: return true;
        default: break;
    }
    if (kProfileId == 1U) return false;
    if (kProfileId == 2U) {
        if (stage >= INIT_LAYOUT && stage <= PREFIX_CONTROL_INIT) return true;
        switch (stage) {
            case SEND_COUNTS: case INIT_COUNT_BUFFERS: case COUNT_COMPUTE: case COUNT_SEND_ISSUE:
            case SEND_SETUP: case PREPARE_INPUT: case Q_SEED_SUBMIT: case Q_SEED_WAIT:
            case Q_LOCAL_UPDATE: case ROUTE_CLASSIFY: case DIRECT_PLAN:
            case COUNT_SNAPSHOT_ISSUE: case COUNT_SNAPSHOT_WAIT: case COUNT_READY_CHECK:
            case COUNT_VALUE_EXTRACT: return true;
            default: return false;
        }
    }
    if (kProfileId == 3U) {
        if (stage >= COUNT_SNAPSHOT_ISSUE && stage <= PREFIX_GENERATION_LOAD) return true;
        switch (stage) {
            case COUNT_SETUP: case PREFIX_CONTROL_INIT: case CAL_CUMSUM: case SEND_COUNTS:
            case INIT_COUNT_BUFFERS: case COUNT_COMPUTE: case COUNT_SEND_ISSUE:
            case WAIT_RECV_COUNTS: case POLL_RECV_COUNTS: case BUILD_RECV_CUMSUM:
            case WAIT_CUMSUM: case PREFIX_COMPLETE: case PREFIX_LOAD: case WRITE_COUNTS_OUTPUT:
            case PREFIX_PUBLISH_WAIT: case PUBLISH_PREFIX_FLAGS:
            case POLL_CORE_SUMS_ITERATIONS: case POLL_RECV_COUNTS_ITERATIONS:
            case WAIT_CUMSUM_ITERATIONS: return true;
            default: return false;
        }
    }
    if (kProfileId == 4U) {
        if (stage >= SENDER_INPUT_INIT && stage <= SENDER_FLAG_INIT) return true;
        switch (stage) {
            case SEND_TOKENS: case PREPARE_INPUT: case SEND_SETUP: case SEND_DRAIN:
            case TOKEN_INPUT: case TOKEN_INPUT_WAIT: case TOKEN_QUANT: case TOKEN_QUANT_WAIT:
            case TOKEN_METADATA: case TOKEN_BUFFER_WAIT: case TOKEN_PACK: case TOKEN_SEND_ISSUE:
            case TX_PACK_READY_WAIT: case TX_METADATA_PUBLISH: case SEND_RETIRE_WAIT:
            case SEND_COMPLETE_WAIT: case Q_SEED_SUBMIT: case Q_SEED_WAIT: case Q_LOCAL_UPDATE:
            case ROUTE_CLASSIFY: case DIRECT_PLAN: return true;
            default: return false;
        }
    }
    switch (stage) {
        case RECV_PREPARE: case BUILD_EXPERT_TASKS: case FORMAT_OUTPUT: case PREFIX_LOAD:
        case RECV_POLL: case RECV_COPY: case RECV_BUFFER_WAIT: case RECV_INPUT: case RECV_OUTPUT:
        case RX_FLAG_ISSUE: case RX_FLAG_WAIT: case RX_FLAG_REDUCE: case RX_FLAG_DECODE:
        case RX_LANE_REUSE_WAIT: case RX_PAYLOAD_READY_WAIT: case RX_METADATA:
        case RX_PAYLOAD_OUTPUT: case RX_META_OUTPUT: case RX_PENDING_UPDATE:
        case RELAY_ISSUE: case RELAY_PRE_DRAIN: case RELAY_COMPLETE: case RECV_COMPLETE_WAIT:
        case CLEAR_LOCAL_FLAGS: case RECV_READY_TOKENS: return true;
        default: return false;
    }
}
// Constant stage arguments fold to one compact index. Disabled profiles reserve
// neither accumulator slots nor probe Scope state in the business kernel.
LL_PROFILE_INLINE constexpr uint32_t ProfileIndex(Stage stage) {
    uint32_t index = 0;
    for (uint32_t s = 0; s < uint32_t(stage); ++s)
        if (ProfileEnabled(Stage(s))) ++index;
    return index;
}
constexpr uint32_t kProfileMetrics = ProfileIndex(StageCount);
}

#undef LL_PROFILE_INLINE
