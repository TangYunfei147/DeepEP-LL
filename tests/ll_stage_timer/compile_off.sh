#!/usr/bin/env bash
# Compile all production NDR LL translation units with tracing absent; no NPU execution.
set -eo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$HERE/../../scripts/ll_timer_env.sh"
set -u
OUT=$(mktemp -d "$LL_TIMER_ROOT/.tmp/ll-timer-off-XXXXXXXX")
SDK="$LL_TIMER_ROOT/third-party/shmem/install/shmem"
test -f "$SDK/lib/libshmem.so"
SOURCES=(dispatch/low_latency.cpp combine/low_latency.cpp)
for mode in prequant_e4m3 fp16 fp16_e4m3 fp16_e5m2 bf16 bf16_e4m3 bf16_e5m2; do
  for group in 4 16; do
    SOURCES+=("dispatch/low_latency/variants/${mode}_g${group}.cpp")
  done
done
# NDR has one compiled transport capability. Runtime MTE is selected at runtime,
# so compiling with RDMA absent would not test the production configuration.
for src in "${SOURCES[@]}"; do
  bisheng -O2 -std=c++17 -fPIC -xasc --npu-arch=dav-3510 --cce-auto-infer-kernel-type=false \
    -DCATLASS_ARCH=3510 -DUSE_SIMT -DL2_CACHE_HINT -DDEEPEP_LL_RDMA=1 -DACLSHMEMI_RDMA_K_BACKEND_HNS_1825=1 \
    -I"$ASCEND_HOME_PATH/include" -I"$ASCEND_HOME_PATH/asc/include" \
    -I"$SDK/include" -I"$SDK/src" -I"$SDK/src/device" -I"$SDK/src/device_simt" \
    -c "$LL_TIMER_ROOT/csrc/kernels/$src" -o "$OUT/${src//\//_}.o"
done
printf 'PASS all %s production NDR LL translation units compile with timer OFF: %s\n' "${#SOURCES[@]}" "$OUT"
