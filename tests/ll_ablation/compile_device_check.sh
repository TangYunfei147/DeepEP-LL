#!/usr/bin/env bash
# Compile only, no SDK rebuild/link/install or device execution.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
root=$(cd "$here/../.." && pwd -P)
sdk=${ASCEND_HOME_PATH:?source CANN first}
out=$(mktemp -d "$root/.tmp/ablation-device-XXXXXXXX")
header="$root/csrc/kernels/dispatch/low_latency/common/ll_ablation_config.h"
cp "$header" "$out/config.saved"
trap 'cp "$out/config.saved" "$header"' EXIT
for mode in ${ABLATION_CHECK_MODES:-0 2 3 4}; do
  sed "s/kLlAblationMode = [0-4];/kLlAblationMode = $mode;/" "$out/config.saved" > "$header"
  for timer in OFF ON; do
    mkdir "$out/m${mode}-${timer}"
    flags=()
    if [ "$timer" = ON ]; then flags+=(-DDEEPEP_LL_STAGE_TIMER=1); fi
    begin_seconds=$SECONDS
    pids=()
    failed=0
    for source in "$root"/csrc/kernels/dispatch/low_latency/variants/*.cpp; do
      target="$out/m${mode}-${timer}/$(basename "$source").o"
      bisheng -O2 -std=c++17 -fPIC -xasc --npu-arch=dav-3510 \
        --cce-auto-infer-kernel-type=false -DCATLASS_ARCH=3510 -DUSE_SIMT -DL2_CACHE_HINT \
        "${flags[@]}" -I"$sdk/include" -I"$sdk/asc/include" \
        -I"$root/third-party/shmem/include" -I"$root/third-party/shmem/src" \
        -I"$root/third-party/shmem/src/device" -I"$root/third-party/shmem/src/device_simt" \
        -c "$source" -o "$target" > "$target.log" 2>&1 &
      pids+=("$!")
      if [ "${#pids[@]}" -eq 4 ]; then
        for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
        pids=()
      fi
    done
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    if [ "$failed" -ne 0 ]; then tail -40 "$out/m${mode}-${timer}/"*.log; exit 1; fi
    echo "PASS mode=$mode timer=$timer seven entries seconds=$((SECONDS-begin_seconds))"
  done
done
echo "DEVICE_OBJECTS=$out"
