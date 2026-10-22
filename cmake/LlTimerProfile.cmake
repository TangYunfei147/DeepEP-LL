# Shared profile validation for Host and independently configured Device builds.
# The numeric identity is consumed by ll_stage_timer_profile.h in both compilers.
set(DEEPEP_LL_TIMER_PROFILE "full" CACHE STRING "LL timer profile: full, overview, prepare, prefix, send, recv")
set_property(CACHE DEEPEP_LL_TIMER_PROFILE PROPERTY STRINGS full overview prepare prefix send recv)
set(_ll_timer_profiles full overview prepare prefix send recv)
list(FIND _ll_timer_profiles "${DEEPEP_LL_TIMER_PROFILE}" DEEPEP_LL_TIMER_PROFILE_ID)
if(DEEPEP_LL_TIMER_PROFILE_ID LESS 0)
    message(FATAL_ERROR "DEEPEP_LL_TIMER_PROFILE must be one of full, overview, prepare, prefix, send, recv")
endif()
if(NOT DEEPEP_LL_STAGE_TIMER)
    # A profile cannot affect an OFF binary or divide its cache identity.
    set(DEEPEP_LL_TIMER_PROFILE "full" CACHE STRING "LL timer profile: full, overview, prepare, prefix, send, recv" FORCE)
    set(DEEPEP_LL_TIMER_PROFILE_ID 0)
endif()
