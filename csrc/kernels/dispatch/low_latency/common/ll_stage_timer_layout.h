#pragma once
#include <cstdint>

namespace LlStageTimer {
constexpr uint32_t kSchema = 1U;
constexpr uint64_t kMagic = 0x4c4c5431ULL;
constexpr uint32_t kSlots = 32U;
constexpr uint32_t kFields = 8U;
constexpr uint32_t kWordsPerCore = kSlots * kFields;
// 每核独占2048B，避免不同核对同一cache line写回。第0行是头，余下为阶段记录。
enum Stage : uint32_t {
    Kernel = 1,
    Magic,
    WorkerInit,
    CountInit,
    CountSend,
    Sender,
    SendPrepare,
    SendPayload,
    SendMetadata,
    CountReceive,
    CountPrefix,
    CountPublish,
    Receiver,
    RecvPrepare,
    RecvWait,
    RecvConsume,
    RecvCleanup,
    CountFinish,
    CombinePrepare,
    CombineNotify,
    CombineWaitReady,
    CombineDirect,
    CombinePartial,
    CombineAggregate,
    CombineDrain,
    CombineFinalDrain,
    SendClassify,
    SendPack,
    SendIssue,
    StageCount
};
constexpr const char* kNames[] = {"header",
                                  "kernel",
                                  "magic",
                                  "worker_init",
                                  "count_init",
                                  "count_send",
                                  "sender",
                                  "send_prepare",
                                  "send_payload",
                                  "send_metadata",
                                  "count_receive",
                                  "count_prefix",
                                  "count_publish",
                                  "receiver",
                                  "recv_prepare",
                                  "recv_wait",
                                  "recv_consume",
                                  "recv_cleanup",
                                  "count_finish",
                                  "combine_prepare",
                                  "combine_notify",
                                  "combine_wait_ready",
                                  "combine_direct",
                                  "combine_partial",
                                  "combine_aggregate",
                                  "combine_drain",
                                  "combine_final_drain",
                                  "send_classify",
                                  "send_pack",
                                  "send_issue"};
static_assert(StageCount <= kSlots, "timer capacity");
static_assert(sizeof(kNames) / sizeof(kNames[0]) == StageCount, "timer names");
}  // namespace LlStageTimer
