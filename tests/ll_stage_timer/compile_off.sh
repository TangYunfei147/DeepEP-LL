#!/usr/bin/env bash
# Compile all specialized production LL units with tracing absent; no NPU execution.
set -eo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$HERE/../../scripts/ll_timer_env.sh"
set -u
LL_OFF_SEND=${DEEPEP_LL_SEND_CORES:-28}
LL_OFF_RECV=${DEEPEP_LL_RECV_CORES:-28}
LL_OFF_PREFIX=${DEEPEP_LL_SMALL_B_PREFIX_CORES-0}
LL_OFF_IMPL=${DEEPEP_LL_DISPATCH_IMPL-current}
if [[ ! $LL_OFF_SEND =~ ^[1-9][0-9]?$ || ! $LL_OFF_RECV =~ ^[1-9][0-9]?$ ]] ||
   (( LL_OFF_SEND + LL_OFF_RECV != 56 )); then
  printf '%s\n' 'SEND_CORES and RECV_CORES must be positive and sum to 56' >&2
  exit 1
fi
if [[ ! $LL_OFF_PREFIX =~ ^(0|[1-9][0-9]?)$ ]] || (( LL_OFF_PREFIX > LL_OFF_RECV )); then
  printf '%s\n' 'DEEPEP_LL_SMALL_B_PREFIX_CORES must be 0 or an integer in [1, RECV_CORES]' >&2
  exit 1
fi
case "$LL_OFF_IMPL" in
  current) LL_OFF_CB59=0 ;;
  cb59|cb59-stream|cb59-pair|cb59-small)
    if (( LL_OFF_SEND != 28 || LL_OFF_RECV != 28 || LL_OFF_PREFIX != 0 )); then
      printf '%s\n' "$LL_OFF_IMPL requires SEND_CORES=28, RECV_CORES=28 and SMALL_B_PREFIX_CORES=0" >&2
      exit 1
    fi
    LL_OFF_CB59=1
    if [[ $LL_OFF_IMPL == cb59-stream ]]; then LL_OFF_CB59=2; fi
    if [[ $LL_OFF_IMPL == cb59-pair ]]; then LL_OFF_CB59=3; fi
    if [[ $LL_OFF_IMPL == cb59-small ]]; then LL_OFF_CB59=4; fi ;;
  *) printf '%s\n' 'DEEPEP_LL_DISPATCH_IMPL must be current, cb59, cb59-stream, cb59-pair or cb59-small' >&2; exit 1 ;;
esac
OUT=$(mktemp -d "$LL_TIMER_ROOT/.tmp/ll-timer-off-XXXXXXXX")
SDK="$LL_TIMER_ROOT/third-party/shmem/install/shmem"
test -f "$SDK/lib/libshmem.so"
for source in "$LL_TIMER_ROOT/csrc/kernels/dispatch/low_latency.cpp" \
              "$LL_TIMER_ROOT"/csrc/kernels/dispatch/low_latency/variants/*.cpp \
              "$LL_TIMER_ROOT/csrc/kernels/combine/low_latency.cpp"; do
  name=$(basename "$(dirname "$source")")-$(basename "$source")
  bisheng -O2 -std=c++17 -fPIC -xasc --npu-arch=dav-3510 --cce-auto-infer-kernel-type=false \
    -DCATLASS_ARCH=3510 -DUSE_SIMT -DL2_CACHE_HINT \
    -DDEEPEP_LL_SEND_CORES="$LL_OFF_SEND" -DDEEPEP_LL_RECV_CORES="$LL_OFF_RECV" \
    -DDEEPEP_LL_SMALL_B_PREFIX_CORES="$LL_OFF_PREFIX" \
    -DDEEPEP_LL_DISPATCH_CB59="$LL_OFF_CB59" \
    -I"$ASCEND_HOME_PATH/include" -I"$ASCEND_HOME_PATH/asc/include" \
    -I"$SDK/include" -I"$SDK/src" -I"$SDK/src/device" -I"$SDK/src/device_simt" \
    -c "$source" -o "$OUT/$name.o"
done
printf 'PASS all specialized production LL units compile with timer OFF: %s\n' "$OUT"
