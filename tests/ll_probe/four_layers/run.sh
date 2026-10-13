#!/usr/bin/env bash
set -eo pipefail
ROOT=/home/tyf/Q4_remote_test
source "$ROOT/env_1009/cann-9.3.0/set_env.sh"
set -u
cd "$ROOT/DeepEP-LL/tests/ll_probe/four_layers"
export LD_LIBRARY_PATH="$ROOT/DeepEP-LL/tests/ll_probe/build/shmem/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${PROBE_VISIBLE_DEVICES:-0,1}" ASCEND_RT_VISIBLE_DEVICES="${PROBE_VISIBLE_DEVICES:-0,1}" ASCEND_VISIBLE_DEVICES="${PROBE_VISIBLE_DEVICES:-0,1}"
export TMPDIR="$ROOT/.tmp" TMP="$ROOT/.tmp" TEMP="$ROOT/.tmp" XDG_CACHE_HOME="$ROOT/.cache"
export ASCEND_WORK_PATH="$PWD/build/runtime_logs" ASCEND_PROCESS_LOG_PATH="$PWD/build/runtime_logs" ASCEND_CACHE_PATH="$ROOT/.cache/ascend"
export SHMEM_LOG_PATH="$PWD/build/runtime_logs/shmem"
mkdir -p "$SHMEM_LOG_PATH"
mkdir -p "$ASCEND_WORK_PATH" "$ASCEND_CACHE_PATH"
exec "$@"
