# Copyright (c) 2026, Lu Lu
# Modified by lishaoxun 2026

# Shared by direct Host and Device CMake entry points. This branch has one
# compiled capability; backend selection happens at runtime.
string(TOUPPER "${DEEPEP_LL_RDMA}" _legacy_ll_rdma)
if(DEFINED DEEPEP_LL_RDMA AND NOT _legacy_ll_rdma MATCHES "^(ON|1|TRUE)$")
    message(WARNING "DEEPEP_LL_RDMA is deprecated and ignored: NDR always compiles HNS_1825 support. Use --backend mte at runtime or kernel/ll-configurable-groups for a build without RDMA.")
endif()
set(DEEPEP_LL_RDMA ON CACHE INTERNAL "Fixed NDR capability" FORCE)

foreach(sdk ASCEND_HOME_PATH SHMEM_ROOT)
    if(NOT IS_DIRECTORY "${${sdk}}")
        message(FATAL_ERROR "${sdk} must identify an installed SDK")
    endif()
endforeach()
set(_rdma_manifest "${SHMEM_ROOT}/deepep-sdk.json")
if(NOT EXISTS "${_rdma_manifest}")
    message(FATAL_ERROR "RDMA LL requires an identified HNS_1825 SDK")
endif()
file(READ "${_rdma_manifest}" _rdma_identity)
string(JSON _rdma_backend ERROR_VARIABLE _rdma_error GET "${_rdma_identity}" build rdma_backend)
string(JSON _rdma_patch ERROR_VARIABLE _rdma_patch_error GET "${_rdma_identity}" build rdma_patch)
if(_rdma_error OR _rdma_patch_error OR NOT _rdma_backend STREQUAL "HNS_1825" OR NOT _rdma_patch STREQUAL "deepep-roce-72-v1")
    message(FATAL_ERROR "Build the NDR SDK with python scripts/build_shmem.py")
endif()
# Resolve from the selected SDK on every configure, including direct CMake use.
unset(ASCENDCL_LIBRARY CACHE)
unset(ASCEND_RUNTIME_LIBRARY CACHE)
unset(SHMEM_LIBRARY CACHE)
set(_cann_library_paths "${ASCEND_HOME_PATH}/lib64"
    "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64")
find_library(ASCENDCL_LIBRARY NAMES ascendcl PATHS ${_cann_library_paths} NO_DEFAULT_PATH REQUIRED)
find_library(ASCEND_RUNTIME_LIBRARY NAMES runtime PATHS ${_cann_library_paths} NO_DEFAULT_PATH REQUIRED)
find_library(SHMEM_LIBRARY NAMES shmem PATHS "${SHMEM_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
add_library(ascendcl SHARED IMPORTED)
set_target_properties(ascendcl PROPERTIES IMPORTED_LOCATION "${ASCENDCL_LIBRARY}")
add_library(ascend_runtime SHARED IMPORTED)
set_target_properties(ascend_runtime PROPERTIES IMPORTED_LOCATION "${ASCEND_RUNTIME_LIBRARY}")
add_library(shmem SHARED IMPORTED)
set_target_properties(shmem PROPERTIES IMPORTED_LOCATION "${SHMEM_LIBRARY}")
