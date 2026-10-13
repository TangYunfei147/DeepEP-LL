#!/usr/bin/env bash
set -eo pipefail
PROBE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$PROBE_DIR/../../scripts/project_env.sh"
require_cann
set -u
cd "$PROBE_DIR"
SOURCE="$REPO_ROOT/third-party/shmem"
LIB="$SHMEM_LIB_DIR"
require_shmem_sdk
"$BISHENG" -O2 -std=c++17 -I"$ASCEND_HOME_PATH/include" -I"$SOURCE/include" \
    shmem_probe.cpp build/transfer_probe.o --cce-fatobj-link \
    -L"$LIB" -L"$ASCEND_HOME_PATH/lib64" -Wl,-rpath,"$LIB" \
    -lshmem -lascendcl -lruntime -o build/shmem_probe 2>&1 | tee build/shmem-driver-link.log
