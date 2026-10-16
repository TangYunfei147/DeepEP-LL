#pragma once
#include <cstdint>

namespace LlStageTimer {
constexpr uint32_t kSchema = 4U;
constexpr uint64_t kMagic = 0x4c4c5431ULL;
constexpr uint32_t kFields = 8U;
// Independent submitted-DMA byte counters, separate from timing events.
constexpr uint32_t kDataCounterOffset = 64U;
constexpr uint32_t kDataCounterBuckets = 5U;
constexpr uint32_t kDataCounterFields = 4U; // bytes, operations, read bytes, read operations
constexpr uint32_t kDataFlagsOffset = 84U;
constexpr uint64_t kDataNoComm = 1U;
#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
__aicore__ inline
#else
inline
#endif
constexpr uint32_t DataBucket(uint32_t rank, uint32_t peer, bool forwarding) {
    return rank == peer ? 0U : (rank / 16U == peer / 16U ? 1U : 3U) + uint32_t(forwarding);
}

enum Stage : uint32_t {
    Kernel = 1,
    Magic,
    CountInit,
    CountSend,
    SendPrepare,
    SendMetadata,
    CountReceive,
    CountPrefix,
    CountPublish,
    RecvPrepare,
    RecvWait,
    RecvCleanup,
    CountFinish,
    CombinePrepare,
    CombineNotify,
    CombineWaitReady,
    CombineDrain,
    CombineFinalDrain,
    SendClassify,
    SendRoute,
    RecvLaneWait,
    RecvGmToUb,
    RecvLoadWait,
    RecvMetadata,
    RecvOutputWait,
    RecvOutputWrite,
    RecvForward,
    SendInputGmToUb,
    SendInputWait,
    SendTail,
    SendLaneWait,
    SendMteWrite,
    SendRdmaStage,
    SendRdmaBodyPost,
    SendRdmaBodyWait,
    SendRdmaFlagPost,
    SendRdmaFlagWait,
    SendRdmaRestore,
    CombineReturnTask,
    CombineReduceTask,
    CombineOutputTask,
    CombineDataWait,
    CombineWaitResult,
    StageCount
};
constexpr const char* kNames[] = {"header",
    "kernel",
    "magic",
    "count_init",
    "count_send",
    "send_prepare",
    "send_metadata",
    "count_receive",
    "count_prefix",
    "count_publish",
    "recv_prepare",
    "recv_wait",
    "recv_cleanup",
    "count_finish",
    "combine_prepare",
    "combine_notify",
    "combine_wait_ready",
    "combine_drain",
    "combine_final_drain",
    "send_classify",
    "send_route",
    "recv_lane_wait",
    "recv_gm_to_ub",
    "recv_load_wait",
    "recv_metadata",
    "recv_output_wait",
    "recv_output_write",
    "recv_forward",
    "send_input_gm_to_ub",
    "send_input_wait",
    "send_tail",
    "send_lane_wait",
    "send_mte_write",
    "send_rdma_stage",
    "send_rdma_body_post",
    "send_rdma_body_wait",
    "send_rdma_flag_post",
    "send_rdma_flag_wait",
    "send_rdma_restore",
    "combine_return_task",
    "combine_reduce_task",
    "combine_output_task",
    "combine_data_wait",
    "combine_wait_result"};
// MegaMoE-style bounded per-type event arrays; no cumulative device timing records.
constexpr uint32_t kBaseWords = 128U;
constexpr uint32_t kEventFields = 3U; // start, end, logical task id
constexpr uint32_t kMaxDynamicEvents = 4096U;
constexpr uint32_t kEventErrorOffset = 85U;
#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
#define LL_EVENT_INLINE __aicore__ inline
#else
#define LL_EVENT_INLINE inline
#endif
LL_EVENT_INLINE constexpr uint32_t ControlOffset(uint32_t stage) { return 8U + stage - 1U; }
LL_EVENT_INLINE constexpr uint32_t EventCapacity(uint32_t stage) {
    return stage == RecvPrepare ? 4U :
           (stage == Kernel || stage == Magic || stage == CountInit || stage == CountSend || stage == SendPrepare || stage == SendMetadata || stage == CountReceive || stage == CountPrefix || stage == CountPublish || stage == RecvCleanup || stage == CountFinish || stage == CombinePrepare || stage == CombineNotify || stage == CombineWaitReady || stage == CombineDrain || stage == CombineFinalDrain ? 1U : kMaxDynamicEvents);
}
LL_EVENT_INLINE constexpr uint32_t EventOffset(uint32_t stage, uint32_t index) {
    uint32_t offset = kBaseWords;
    for (uint32_t s = 1U; s < stage; ++s) offset += EventCapacity(s) * kEventFields;
    return offset + index * kEventFields;
}
LL_EVENT_INLINE constexpr bool DependencyWait(uint32_t stage) {
    return stage == RecvWait || stage == RecvLaneWait || stage == RecvLoadWait ||
           stage == RecvOutputWait || stage == SendInputWait || stage == SendLaneWait ||
           stage == SendRdmaBodyWait || stage == SendRdmaFlagWait || stage == CombineWaitReady || stage == CombineDataWait || stage == CombineWaitResult;
}
#undef LL_EVENT_INLINE
constexpr uint32_t kWordsPerCore = (EventOffset(StageCount, 0U) + 63U) / 64U * 64U;
constexpr uint32_t kSlots = kWordsPerCore / kFields;
static_assert(kWordsPerCore % 64U == 0U, "independent 512B-aligned cores");
static_assert(8U + StageCount - 1U <= kDataCounterOffset, "stage/counter separation");
static_assert(kDataCounterOffset + kDataCounterBuckets * kDataCounterFields == kDataFlagsOffset, "counter layout");
static_assert(kDataFlagsOffset < kWordsPerCore, "timer capacity");
static_assert(sizeof(kNames) / sizeof(kNames[0]) == StageCount, "timer names");
}  // namespace LlStageTimer
