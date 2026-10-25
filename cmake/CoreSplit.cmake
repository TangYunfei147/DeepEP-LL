# Shared by Host and direct Device configuration.
set(DEEPEP_LL_SEND_CORES 28 CACHE STRING "LL sender AIV count")
set(DEEPEP_LL_RECV_CORES 28 CACHE STRING "LL receiver AIV count")
set(DEEPEP_LL_SMALL_B_PREFIX_CORES 0 CACHE STRING "EP16 small direct prefix producers; 0 uses all receive cores")
foreach(_key DEEPEP_LL_SEND_CORES DEEPEP_LL_RECV_CORES)
    if(NOT "${${_key}}" MATCHES "^[1-9][0-9]*$")
        message(FATAL_ERROR "${_key} must be a positive integer")
    endif()
endforeach()
math(EXPR _ll_total "${DEEPEP_LL_SEND_CORES}+${DEEPEP_LL_RECV_CORES}")
if(NOT _ll_total EQUAL 56)
    message(FATAL_ERROR "LL send + receive cores must equal 56")
endif()
if(NOT "${DEEPEP_LL_SMALL_B_PREFIX_CORES}" MATCHES "^(0|[1-9][0-9]?)$"
   OR DEEPEP_LL_SMALL_B_PREFIX_CORES GREATER DEEPEP_LL_RECV_CORES)
    message(FATAL_ERROR "DEEPEP_LL_SMALL_B_PREFIX_CORES must be 0 or an integer in 1..DEEPEP_LL_RECV_CORES")
endif()
add_compile_definitions(DEEPEP_LL_SEND_CORES=${DEEPEP_LL_SEND_CORES} DEEPEP_LL_RECV_CORES=${DEEPEP_LL_RECV_CORES}
    DEEPEP_LL_SMALL_B_PREFIX_CORES=${DEEPEP_LL_SMALL_B_PREFIX_CORES})
