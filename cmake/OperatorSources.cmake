# One selection shared by the Host and Device build trees. HT is optional;
# the LL mode partition and its production compile flags remain unchanged.
option(DEEPEP_BUILD_HT "Build high-throughput notify/dispatch/combine" ON)
set(DEEPEP_CORE_HOST_SOURCES
    csrc/bindings/module.cpp
    csrc/runtime/session.cpp
    csrc/ops/low_latency.cpp)
set(DEEPEP_CORE_DEVICE_SOURCES
    csrc/kernels/dispatch/low_latency.cpp
    csrc/kernels/dispatch/low_latency/variants/prequant_e4m3.cpp
    csrc/kernels/dispatch/low_latency/variants/fp16.cpp
    csrc/kernels/dispatch/low_latency/variants/fp16_e4m3.cpp
    csrc/kernels/dispatch/low_latency/variants/fp16_e5m2.cpp
    csrc/kernels/dispatch/low_latency/variants/bf16.cpp
    csrc/kernels/dispatch/low_latency/variants/bf16_e4m3.cpp
    csrc/kernels/dispatch/low_latency/variants/bf16_e5m2.cpp
    csrc/kernels/combine/low_latency.cpp)
set(DEEPEP_CORE_CAPABILITIES
    runtime-mte runtime-shmem-huge1g-v1 event
    elastic-dispatch-low-latency-v1 elastic-combine-low-latency-v1)
if(DEEPEP_BUILD_HT)
    list(APPEND DEEPEP_CORE_HOST_SOURCES
        csrc/ops/notify.cpp csrc/ops/dispatch.cpp csrc/ops/combine.cpp)
    list(APPEND DEEPEP_CORE_DEVICE_SOURCES
        csrc/kernels/notify/notify.cpp
        csrc/kernels/dispatch/dispatch.cpp
        csrc/kernels/combine/combine.cpp
        csrc/kernels/combine/reuse.cpp
        csrc/kernels/combine/combine_ep8_mte.cpp)
    list(APPEND DEEPEP_CORE_CAPABILITIES
        runtime-urma runtime-mpi-bootstrap
        elastic-notify-metadata-v7 elastic-notify-profile-v1
        elastic-dispatch-expanded-v1 elastic-combine-expanded-v3
        elastic-combine-rank-return-v1 elastic-combine-return-slots-v2
        elastic-combine-chunk-tuning-v1 elastic-combine-profile-v4
        elastic-combine-peers-v1 elastic-combine-chunks-v2 runtime-combine-sparse-qp-v1)
endif()
