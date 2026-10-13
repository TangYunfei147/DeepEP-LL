#!/usr/bin/env bash
set -eo pipefail
ROOT=/home/tyf/Q4_remote_test
source "$ROOT/env_1009/cann-9.3.0/set_env.sh"
set -u
cd "$ROOT/DeepEP-LL/tests/ll_probe"
export LD_LIBRARY_PATH="$PWD/build/shmem/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0,1 ASCEND_RT_VISIBLE_DEVICES=0,1 ASCEND_VISIBLE_DEVICES=0,1
export TMPDIR="$ROOT/.tmp" TMP="$ROOT/.tmp" TEMP="$ROOT/.tmp"
export XDG_CACHE_HOME="$ROOT/.cache" ASCEND_CACHE_PATH="$ROOT/.cache/ascend"
export ASCEND_WORK_PATH="$PWD/build/runtime_logs" ASCEND_PROCESS_LOG_PATH="$PWD/build/runtime_logs"
export SHMEM_LOG_PATH="$PWD/build/runtime_logs/shmem"
mkdir -p "$SHMEM_LOG_PATH"
mkdir -p "$TMPDIR" "$ASCEND_CACHE_PATH" "$ASCEND_WORK_PATH"
python3 scan_shmem.py "$@"
