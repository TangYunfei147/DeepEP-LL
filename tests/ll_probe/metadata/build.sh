#!/usr/bin/env bash
set -eo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$HERE/../../../scripts/project_env.sh"
require_cann
set -u
cd "$HERE"
mkdir -p build
SHMEM="$REPO_ROOT/third-party/shmem"
LIB="$SHMEM_LIB_DIR"
require_shmem_sdk
python3 "$HERE/prepare_snapshot.py"
"$BISHENG" -O2 -std=c++17 -xasc --npu-arch=dav-3510 --cce-auto-infer-kernel-type=false \
  -I"$ASCEND_HOME_PATH/asc/include" -I"$ASCEND_HOME_PATH/include" \
  -I"$SHMEM/include" -I"$SHMEM/src" -c kernel/metadata_kernel.cpp -o build/metadata_kernel.o
"$BISHENG" -O2 -std=c++17 -xasc --npu-arch=dav-3510 --cce-auto-infer-kernel-type=false \
  -DPROBE_METADATA_DIAGNOSTIC=1 -I"$ASCEND_HOME_PATH/asc/include" -I"$ASCEND_HOME_PATH/include" \
  -I"$SHMEM/include" -I"$SHMEM/src" -c kernel/metadata_kernel.cpp -o build/metadata_check.o
"$BISHENG" -O2 -std=c++17 -I"$ASCEND_HOME_PATH/include" -I"$ASCEND_HOME_PATH/asc/include" host/metadata_host.cpp build/metadata_kernel.o build/metadata_check.o \
  --cce-fatobj-link -L"$ASCEND_HOME_PATH/lib64" -lascendcl -lruntime -lplatform -ltiling_api -lunified_dlog -o build/metadata_probe
