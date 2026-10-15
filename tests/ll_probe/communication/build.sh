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
"$BISHENG" -O2 -std=c++17 -xasc --npu-arch=dav-3510 --cce-auto-infer-kernel-type=false \
  -I"$ASCEND_HOME_PATH/asc/include" -I"$ASCEND_HOME_PATH/include" -c kernel/comm_kernel.cpp -o build/comm_kernel.o
"$BISHENG" -O2 -std=c++17 -I"$ASCEND_HOME_PATH/asc/include" -I"$ASCEND_HOME_PATH/include" -I"$SHMEM/include" \
  host/comm_host.cpp build/comm_kernel.o --cce-fatobj-link -L"$LIB" -L"$ASCEND_HOME_PATH/lib64" \
  -Wl,-rpath,"$LIB" -lshmem -lascendcl -lruntime -lplatform -ltiling_api -lunified_dlog -lstdc++ -o build/comm_probe

python3 "$REPO_ROOT/tests/ll_probe/communication/ndr_build.py"
