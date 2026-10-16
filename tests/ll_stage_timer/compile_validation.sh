#!/usr/bin/env bash
# Compile/CPU checks only. No device initialization or kernel launch.
set -eo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$project_root"
source scripts/ll_timer_env.sh
: "${SHMEM_ROOT:?Provide an existing identified native SDK explicitly; this script does not rebuild it}"
export DEEPEP_NPU_ARCH=Ascend950
export DEEPEP_LL_STAGE_TIMER=ON
export DEEPEP_BUILD_MEGAMOE=OFF DEEPEP_BUILD_EP_GMM_FUSED=OFF
export CMAKE_BUILD_PARALLEL_LEVEL=4
export PROBE_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1 ASCEND_RT_VISIBLE_DEVICES=0,1 ASCEND_VISIBLE_DEVICES=0,1
export LD_LIBRARY_PATH="$SHMEM_ROOT/lib:${LD_LIBRARY_PATH:-}"
log_dir="$project_root/.logs/event-validation"
mkdir -p "$log_dir" "$project_root/.artifacts/event-wheels"
phase=cpu
trap 'code=$?; printf "FAILED phase=%s code=%s\n" "$phase" "$code"; tail -n 60 "$log_dir/$phase.log"; exit "$code"' ERR
python -m pytest tests/ll_stage_timer tests/ll_rdma/test_dispatch_variants.py tests/ll_rdma/test_transport.py -q \
  --tb=short --basetemp="$project_root/.tmp/event-cpu" > "$log_dir/cpu.log" 2>&1
cat "$log_dir/cpu.log"
phase=wheel
python -m pip wheel -v --no-build-isolation --no-deps . \
  --wheel-dir "$project_root/.artifacts/event-wheels" > "$log_dir/wheel.log" 2>&1
printf 'PASS full timer ON wheel compile\n'
phase=off
cmake -S cmake/device -B .tmp/event-device-off -DCMAKE_BUILD_TYPE=Release \
  -DDEEPEP_LL_STAGE_TIMER=OFF -DDEEPEP_SOURCE_DIR="$project_root" \
  -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" -DSHMEM_ROOT="$SHMEM_ROOT" \
  -DCMAKE_CXX_COMPILER="$ASCEND_HOME_PATH/tools/bisheng_compiler/bin/bisheng" > "$log_dir/off.log" 2>&1
cmake --build .tmp/event-device-off --parallel 4 >> "$log_dir/off.log" 2>&1
printf 'PASS timer OFF device compile; no NPU execution\n'
