#!/usr/bin/env bash
set -eo pipefail
FOUR_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$FOUR_DIR/../../../scripts/project_env.sh"
require_cann
set -u
cd "$FOUR_DIR"
python3 "$FOUR_DIR/prepare_snapshot.py"
mkdir -p build
SHMEM="$REPO_ROOT/third-party/shmem"
LIB="$SHMEM_LIB_DIR"
require_shmem_sdk
"$BISHENG" -O2 -std=c++17 -xasc --npu-arch=dav-3510 --cce-auto-infer-kernel-type=false \
  -I"$ASCEND_HOME_PATH/asc/include" -I"$ASCEND_HOME_PATH/include" -c comm_kernel.cpp -o build/comm_kernel.o
"$BISHENG" -O2 -std=c++17 -I"$ASCEND_HOME_PATH/include" -I"$SHMEM/include" \
  comm_host.cpp build/comm_kernel.o --cce-fatobj-link -L"$LIB" -L"$ASCEND_HOME_PATH/lib64" \
  -Wl,-rpath,"$LIB" -lshmem -lascendcl -lruntime -o build/comm_probe
"$BISHENG" -O2 -std=c++17 -xasc --npu-arch=dav-3510 --cce-auto-infer-kernel-type=false \
  -I"$ASCEND_HOME_PATH/asc/include" -I"$ASCEND_HOME_PATH/include" \
  -I"$SHMEM/include" -I"$SHMEM/src" -c metadata_kernel.cpp -o build/metadata_kernel.o
"$BISHENG" -O2 -std=c++17 -xasc --npu-arch=dav-3510 --cce-auto-infer-kernel-type=false \
  -DPROBE_METADATA_DIAGNOSTIC=1 -I"$ASCEND_HOME_PATH/asc/include" -I"$ASCEND_HOME_PATH/include" \
  -I"$SHMEM/include" -I"$SHMEM/src" -c metadata_kernel.cpp -o build/metadata_check.o
"$BISHENG" -O2 -std=c++17 -I"$ASCEND_HOME_PATH/include" -I"$ASCEND_HOME_PATH/asc/include" metadata_host.cpp build/metadata_kernel.o build/metadata_check.o \
  --cce-fatobj-link -L"$ASCEND_HOME_PATH/lib64" -lascendcl -lruntime -lplatform -ltiling_api -lunified_dlog -o build/metadata_probe
