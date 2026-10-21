# Copyright (c) 2026, Lu Lu
# Modified by candy_cloud_fly 2026

cmake_minimum_required(VERSION 3.20)
project(deepep_device LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
include("${CMAKE_CURRENT_LIST_DIR}/../Dependencies.cmake")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${SHMEM_ROOT}/deepep-sdk.json")
set(_sdk_include "${ASCEND_HOME_PATH}/include")
set(_arch_include "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include")
set(_tikcpp "${ASCEND_HOME_PATH}/compiler/tikcpp")
# Some toolkit layouts expose a legacy rt.h compatibility header separately.
# Do not require that header to configure the CANN 9.2 -xasc target: the
# compiler/toolkit headers determine which runtime API is actually needed.
# Re-evaluate on configure rather than retaining a path from another SDK.
unset(DEVICE_RT_INCLUDE CACHE)
find_path(DEVICE_RT_INCLUDE rt.h HINTS
    "${_sdk_include}/runtime" "${_sdk_include}/experiment/runtime" "${_sdk_include}"
    "${_sdk_include}/acl/runtime" "${ASCEND_HOME_PATH}/runtime/include"
    "${_arch_include}/runtime" "${_arch_include}/experiment/runtime"
    "${_arch_include}/acl/runtime" "${_arch_include}"
    "${ASCEND_HOME_PATH}/pkg_inc/runtime"
    "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/runtime"
    "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/runtime/include"
    NO_DEFAULT_PATH)
# Share HT objects and retain both LL diagnostic variants. Only the selected
# variant is built, so repeated timer switches reuse dependency-tracked objects.
add_library(deepep_kernel_common OBJECT
    "${DEEPEP_SOURCE_DIR}/csrc/kernels/runtime/host_barrier.cpp"
    "${DEEPEP_SOURCE_DIR}/csrc/kernels/notify/notify.cpp"
    "${DEEPEP_SOURCE_DIR}/csrc/kernels/dispatch/dispatch.cpp"
    "${DEEPEP_SOURCE_DIR}/csrc/kernels/combine/combine.cpp"
    "${DEEPEP_SOURCE_DIR}/csrc/kernels/combine/reuse.cpp"
    "${DEEPEP_SOURCE_DIR}/csrc/kernels/combine/combine_ep8_mte.cpp")
set(_ll_sources
    "${DEEPEP_SOURCE_DIR}/csrc/kernels/dispatch/low_latency.cpp"
    "${DEEPEP_SOURCE_DIR}/csrc/kernels/combine/low_latency.cpp")
# One dispatch mode/group per translation unit keeps each device compilation
# bounded; all supported modes still enter the final production library.
foreach(_mode prequant_e4m3 fp16 fp16_e4m3 fp16_e5m2 bf16 bf16_e4m3 bf16_e5m2)
    foreach(_group 4 16)
        list(APPEND _ll_sources
            "${DEEPEP_SOURCE_DIR}/csrc/kernels/dispatch/low_latency/variants/${_mode}_g${_group}.cpp")
    endforeach()
endforeach()
add_library(deepep_kernel_ll_plain OBJECT ${_ll_sources})
add_library(deepep_kernel_ll_timer OBJECT ${_ll_sources})
set_target_properties(deepep_kernel_ll_plain deepep_kernel_ll_timer PROPERTIES EXCLUDE_FROM_ALL TRUE)
target_compile_definitions(deepep_kernel_ll_timer PRIVATE DEEPEP_LL_STAGE_TIMER=1)
if(DEEPEP_LL_STAGE_TIMER)
    set(_ll_variant deepep_kernel_ll_timer)
else()
    set(_ll_variant deepep_kernel_ll_plain)
endif()
add_library(deepep_kernels SHARED
    $<TARGET_OBJECTS:deepep_kernel_common> $<TARGET_OBJECTS:${_ll_variant}>)

add_library(deepep_kernel_options INTERFACE)
target_compile_options(deepep_kernel_options INTERFACE "$<$<NOT:$<CONFIG:Debug>>:-O2>" -xasc --npu-arch=dav-3510
    --cce-auto-infer-kernel-type=false
    "-ffile-prefix-map=${DEEPEP_SOURCE_DIR}=."
    "-ffile-prefix-map=${ASCEND_HOME_PATH}=CANN_SDK"
    "-ffile-prefix-map=${SHMEM_ROOT}=SHMEM_SDK")
# Preserve the original SHARED target's automatically added export definition.
target_compile_definitions(deepep_kernel_options INTERFACE CATLASS_ARCH=3510 USE_SIMT L2_CACHE_HINT
    DEEPEP_LL_RDMA=1 ACLSHMEMI_RDMA_K_BACKEND_HNS_1825=1 deepep_kernels_EXPORTS)
target_include_directories(deepep_kernel_options INTERFACE
    "${_sdk_include}" "${_arch_include}"
    "${_tikcpp}" "${_tikcpp}/tikcfw"
    "${_tikcpp}/tikcfw/impl" "${_tikcpp}/tikcfw/interface"
    "${SHMEM_ROOT}/include" "${SHMEM_ROOT}/src" "${SHMEM_ROOT}/src/device"
    "${SHMEM_ROOT}/src/device_simt")
if(DEVICE_RT_INCLUDE)
    target_include_directories(deepep_kernel_options INTERFACE "${DEVICE_RT_INCLUDE}")
    message(STATUS "Device runtime compatibility headers: ${DEVICE_RT_INCLUDE}")
else()
    message(STATUS "No standalone rt.h found; using the selected CANN -xasc toolchain headers")
endif()
foreach(_objects deepep_kernel_common deepep_kernel_ll_plain deepep_kernel_ll_timer)
    target_link_libraries(${_objects} PRIVATE deepep_kernel_options)
endforeach()

target_link_options(deepep_kernels PRIVATE --cce-fatobj-link -Wl,-z,relro,-z,now,-z,noexecstack)
target_link_libraries(deepep_kernels PRIVATE ascend_runtime ascendcl shmem ${CMAKE_DL_LIBS})
set_target_properties(deepep_kernels PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "$ORIGIN" INSTALL_RPATH_USE_LINK_PATH FALSE)
