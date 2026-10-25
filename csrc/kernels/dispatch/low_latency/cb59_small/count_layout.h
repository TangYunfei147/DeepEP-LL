// Copyright (c) 2026, Lu Lu
#pragma once
#include "../cb59/count_layout.h"

// Small-direct specialization changes compile-time control flow only. Keep the
// historical count layout and 190 KiB arena, including all original UB setup.
namespace DispatchDedup::Cb59Small {
using Cb59Replay::DispatchServerDedupCountLayout;
using Cb59Replay::kEpServerDedupCountUbBytes;
using Cb59Replay::UB_SINGLE_DMA_SIZE_MAX;
using Cb59Replay::UB_SINGLE_PING_PONG_ADD_SIZE_MAX;
using Cb59Replay::PING_PONG_SIZE;
}  // namespace DispatchDedup::Cb59Small
