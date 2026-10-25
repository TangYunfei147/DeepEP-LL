#pragma once
#include "ll_stage_timer_layout.h"
#ifndef DEEPEP_LL_TIMER_PROFILE_ID
#define DEEPEP_LL_TIMER_PROFILE_ID 6
#endif
namespace LlStageTimer {
constexpr uint32_t kProfileId = DEEPEP_LL_TIMER_PROFILE_ID;
static_assert(kProfileId == 6U, "LL timer supports sparse milestones only; rebuild matching Host/device");
constexpr const char* kProfileName = "sparse";
}
