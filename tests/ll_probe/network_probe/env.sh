#!/usr/bin/env bash
# All writable state stays under an explicit authorized root.
set -eo pipefail
export PROBE_ROOT="${PROBE_ROOT:-/home/tyf/Q4_remote_test}"
if [[ "$PROBE_ROOT" != /* ]]; then echo 'PROBE_ROOT must be absolute' >&2; exit 2; fi
if [[ -n "${CANN_ENV:-}" ]]; then
    source "$CANN_ENV"
elif [[ -f "$PROBE_ROOT/env_1009/cann-9.3.0/set_env.sh" ]]; then
    source "$PROBE_ROOT/env_1009/cann-9.3.0/set_env.sh"
fi
set -u
export TMPDIR="$PROBE_ROOT/.tmp" TMP="$PROBE_ROOT/.tmp" TEMP="$PROBE_ROOT/.tmp"
export XDG_CACHE_HOME="$PROBE_ROOT/.cache" CCACHE_DIR="$PROBE_ROOT/.cache/ccache"
export CUDA_CACHE_PATH="$PROBE_ROOT/.cache/cuda" TORCH_EXTENSIONS_DIR="$PROBE_ROOT/.cache/torch_extensions"
export TRITON_CACHE_DIR="$PROBE_ROOT/.cache/triton" PIP_CACHE_DIR="$PROBE_ROOT/.cache/pip"
export SHMEM_LOG_PATH="$PROBE_ROOT/.logs/shmem" ASCEND_WORK_PATH="$PROBE_ROOT/.logs/ascend"
export SHMEM_LOG_LEVEL="${SHMEM_LOG_LEVEL:-INFO}"
export ASCEND_PROCESS_LOG_PATH="$PROBE_ROOT/.logs/ascend" PYTHONDONTWRITEBYTECODE=1
export PROBE_VISIBLE_DEVICES="${PROBE_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES="$PROBE_VISIBLE_DEVICES" ASCEND_RT_VISIBLE_DEVICES="$PROBE_VISIBLE_DEVICES"
export ASCEND_VISIBLE_DEVICES="$PROBE_VISIBLE_DEVICES"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$SHMEM_LOG_PATH" "$ASCEND_WORK_PATH"
