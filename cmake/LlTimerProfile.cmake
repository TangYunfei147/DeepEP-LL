# Shared profile validation for Host and independently configured Device builds.
# The numeric identity is consumed by ll_stage_timer_profile.h in both compilers.
set(DEEPEP_LL_TIMER_PROFILE "sparse" CACHE STRING "LL timer profile: sparse")
set_property(CACHE DEEPEP_LL_TIMER_PROFILE PROPERTY STRINGS sparse)
if(NOT DEEPEP_LL_TIMER_PROFILE STREQUAL "sparse")
    message(FATAL_ERROR "DEEPEP_LL_TIMER_PROFILE must be sparse; legacy full/phase profiles are offline-analysis only. Unset the old value and rebuild.")
endif()
# Preserve the identity reserved for the sparse milestone schema, including OFF.
# The six legacy IDs are retained only by the offline decoder.
set(DEEPEP_LL_TIMER_PROFILE_ID 6)
