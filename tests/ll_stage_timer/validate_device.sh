#!/usr/bin/env bash
# Timer primitive only: no LL domain or SHMEM collective is fabricated.
set -eo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$HERE/../../scripts/ll_timer_env.sh"
set -u
export CUDA_VISIBLE_DEVICES=0,1 ASCEND_RT_VISIBLE_DEVICES=0,1 ASCEND_VISIBLE_DEVICES=0,1 PROBE_VISIBLE_DEVICES=0,1
OUT=$(mktemp -d "$LL_TIMER_ROOT/.tmp/ll-timer-device-XXXXXXXX")
printf '%s\n' "$OUT"
INC=(-I"$ASCEND_HOME_PATH/asc/include" -I"$ASCEND_HOME_PATH/include")
for enabled in 0 1; do
  FLAGS=(); if [[ "$enabled" == 1 ]]; then FLAGS=(-DDEEPEP_LL_STAGE_TIMER=1); fi
  bisheng -O2 -std=c++17 -xasc --npu-arch=dav-3510 --cce-auto-infer-kernel-type=false \
    "${FLAGS[@]}" "${INC[@]}" -c "$HERE/timer_smoke.cpp" -o "$OUT/timer-$enabled.o"
  bisheng -O2 -std=c++17 "${INC[@]}" "$HERE/timer_smoke_host.cpp" "$OUT/timer-$enabled.o" \
    --cce-fatobj-link -L"$ASCEND_HOME_PATH/lib64" -lascendcl -lruntime -o "$OUT/timer-$enabled"
  for mode in 1; do "$OUT/timer-$enabled" "$OUT/raw-$enabled-$mode.json" "$mode" "$enabled"; done
done
python3 "$HERE/check_device.py" "$OUT"
