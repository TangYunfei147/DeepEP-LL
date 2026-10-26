#pragma once
#include <cstdint>

namespace LlReferenceProfile {
// Independent LL dispatch reference record; not the stage-timer schema.
constexpr uint32_t kSchema = 1U;
constexpr uint64_t kMagic = 0x4c4c4350ULL;
constexpr uint32_t kSlots = 8U;
constexpr uint32_t kFields = 8U;
constexpr uint32_t kWordsPerCore = kSlots * kFields;
constexpr const char* kNames[kWordsPerCore] = {
    "kernel_entry", "config_ready", "magic_begin", "magic_end", "counts_init_end", "role_init_end",
    "send_count_begin", "send_count_issue_end", "sender_begin", "sender_end_after_MTE3_S",
    "recv_count_begin", "recv_count_end", "cumsum_begin", "cumsum_issue_end",
    "complete_cumsum_begin", "complete_cumsum_issue_end", "count_drain_begin", "count_drain_end_PIPE_ALL",
    "receiver_begin", "receiver_end", "final_drain_begin", "final_drain_end_PIPE_ALL", "reserved", "reserved",
    "count_ready_poll_begin", "count_ready_poll_end_after_V_S", "cumsum_peer_join_begin",
    "cumsum_peer_join_clear_issue_end", "prefix_ready_wait_begin", "prefix_ready_wait_clear_issue_end",
    "receive_prefix_loaded_after_MTE2_S", "receive_tasks_built", "receive_buffers_ready",
    "receive_packet_poll_accum_cycles", "receive_load_consume_submit_accum_cycles",
    "receive_lane_wait_clear_issue_end", "receive_final_PIPE_ALL_end", "receive_task_count",
    "send_buffers_routes_ready", "send_packets_issue_end", "send_metadata_end_after_MTE3_S",
    "receive_probe_batch_count", "first_packet_ready", "send_route_count",
    "reserved", "reserved", "reserved", "reserved", "reserved", "reserved",
    "rank", "AIV_core", "EP", "active_aiv", "BS", "schema", "operation", "before_finalize",
    "valid_mask", "magic", "boundary_mode", "complete", "reserved", "reserved"
};
static_assert(kWordsPerCore * sizeof(uint64_t) == 512U, "one independent aligned record per AIV");
}  // namespace LlReferenceProfile
