#!/usr/bin/env bash
# Compile all specialized production LL units with tracing absent; no NPU execution.
set -eo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$HERE/../../scripts/ll_timer_env.sh"
set -u
OUT=$(mktemp -d "$LL_TIMER_ROOT/.tmp/ll-timer-off-XXXXXXXX")
SDK="$LL_TIMER_ROOT/third-party/shmem/install/shmem"
test -f "$SDK/lib/libshmem.so"
for source in "$LL_TIMER_ROOT/csrc/kernels/dispatch/low_latency.cpp" \
              "$LL_TIMER_ROOT"/csrc/kernels/dispatch/low_latency/variants/*.cpp \
              "$LL_TIMER_ROOT/csrc/kernels/combine/low_latency.cpp"; do
  name=$(basename "$(dirname "$source")")-$(basename "$source")
  bisheng -O2 -std=c++17 -fPIC -xasc --npu-arch=dav-3510 --cce-auto-infer-kernel-type=false \
    -DCATLASS_ARCH=3510 -DUSE_SIMT -DL2_CACHE_HINT \
    -I"$ASCEND_HOME_PATH/include" -I"$ASCEND_HOME_PATH/asc/include" \
    -I"$SDK/include" -I"$SDK/src" -I"$SDK/src/device" -I"$SDK/src/device_simt" \
    -c "$source" -o "$OUT/$name.o"
done
printf 'PASS all specialized production LL units compile with timer OFF: %s\n' "$OUT"
