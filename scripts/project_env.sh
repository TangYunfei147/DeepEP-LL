#!/usr/bin/env bash
# Source this file. All defaults follow this checkout, not the caller's cwd.
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
export REPO_ROOT DEEPEP_PROJECT_ROOT="$REPO_ROOT"
project_path() {
    python3 -c 'import pathlib,sys; p=pathlib.Path(sys.argv[2]).expanduser(); print((p if p.is_absolute() else pathlib.Path(sys.argv[1])/p).resolve())' "$REPO_ROOT" "$1"
}
PROBE_ROOT=$(project_path "${PROBE_ROOT:-.}")
export PROBE_ROOT
# Set caches before loading an externally supplied SDK environment script.
_project_runtime_dirs() {
    export TMPDIR="$PROBE_ROOT/.tmp" TMP="$PROBE_ROOT/.tmp" TEMP="$PROBE_ROOT/.tmp"
    export XDG_CACHE_HOME="$PROBE_ROOT/.cache" CCACHE_DIR="$PROBE_ROOT/.cache/ccache"
    export CUDA_CACHE_PATH="$PROBE_ROOT/.cache/cuda" TORCH_EXTENSIONS_DIR="$PROBE_ROOT/.cache/torch_extensions"
    export TRITON_CACHE_DIR="$PROBE_ROOT/.cache/triton" PIP_CACHE_DIR="$PROBE_ROOT/.cache/pip"
    export ASCEND_CACHE_PATH="$PROBE_ROOT/.cache/ascend"
    export SHMEM_LOG_PATH="$PROBE_ROOT/.logs/shmem" ASCEND_WORK_PATH="$PROBE_ROOT/.logs/ascend"
    export ASCEND_PROCESS_LOG_PATH="$PROBE_ROOT/.logs/ascend" PYTHONDONTWRITEBYTECODE=1
    mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$SHMEM_LOG_PATH" "$ASCEND_WORK_PATH" "$PROBE_ROOT/results"
}
_project_runtime_dirs
if [[ -n "${CANN_ENV:-}" ]]; then
    CANN_ENV=$(project_path "$CANN_ENV")
    if [[ ! -f "$CANN_ENV" ]]; then echo "CANN_ENV not found: $CANN_ENV" >&2; return 2; fi
    # Vendor set_env.sh need not support nounset.
    _project_restore_u=0
    case "$-" in *u*) _project_restore_u=1; set +u ;; esac
    source "$CANN_ENV"
    if [[ "$_project_restore_u" == 1 ]]; then set -u; fi
    export CANN_ENV
fi
if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then
    ASCEND_HOME_PATH=$(project_path "$ASCEND_HOME_PATH")
    export ASCEND_HOME_PATH
fi
_project_runtime_dirs
# Device visibility belongs to the caller or an explicit deployment/validation
# entry point. Shared SDK/path setup must not restrict or expand devices.
unset SHMEM_ROOT CATLASS_ROOT
export SHMEM_LIB_DIR="$REPO_ROOT/third-party/shmem/install/shmem/lib"
for _project_dep in "$REPO_ROOT/third-party/shmem" "$REPO_ROOT/third-party/catlass"; do
    if [[ "$(project_path "$_project_dep")" != "$_project_dep" ]]; then
        echo 'Dependencies must be in-repository checkouts, not symlinks to external SDKs' >&2; return 2
    fi
done
export LD_LIBRARY_PATH="$SHMEM_LIB_DIR:${LD_LIBRARY_PATH:-}"
require_cann() {
    if [[ -z "${ASCEND_HOME_PATH:-}" || ! -d "$ASCEND_HOME_PATH/include" ]]; then
        echo 'Load your CANN environment first, or set CANN_ENV to its set_env.sh (relative to the repository is supported).' >&2
        return 2
    fi
    if command -v bisheng >/dev/null 2>&1; then
        BISHENG=$(command -v bisheng)
    elif [[ -x "$ASCEND_HOME_PATH/tools/bisheng_compiler/bin/bisheng" ]]; then
        BISHENG="$ASCEND_HOME_PATH/tools/bisheng_compiler/bin/bisheng"
    else
        echo 'bisheng not found in the configured CANN environment' >&2; return 2
    fi
    export BISHENG
}
require_shmem_sdk() {
    if [[ ! -f "$SHMEM_LIB_DIR/libshmem.so" || ! -f "$SHMEM_LIB_DIR/../deepep-sdk.json" ]]; then
        echo 'Missing in-repository SHMEM SDK. From the project root run: python scripts/build_shmem.py' >&2
        return 2
    fi
    python3 "$REPO_ROOT/scripts/record_shmem_sdk.py" --sdk third-party/shmem/install/shmem --check >/dev/null
}
