#!/usr/bin/env bash
set -eo pipefail
ROOT=/home/tyf/Q4_remote_test
source "$ROOT/env_1009/cann-9.3.0/set_env.sh"
set -u
export TMPDIR="$ROOT/.tmp" XDG_CACHE_HOME="$ROOT/.cache"
cd "$ROOT/DeepEP-LL/tests/ll_probe/four_layers"
mkdir -p build
SHMEM="$ROOT/DeepEP-LL/third-party/shmem"
LIB="$ROOT/DeepEP-LL/tests/ll_probe/build/shmem/lib"
bisheng -O2 -std=c++17 -xasc --npu-arch=dav-3510 --cce-auto-infer-kernel-type=false \
  -I"$ASCEND_HOME_PATH/asc/include" -I"$ASCEND_HOME_PATH/include" -c comm_kernel.cpp -o build/comm_kernel.o
bisheng -O2 -std=c++17 -I"$ASCEND_HOME_PATH/include" -I"$SHMEM/include" \
  comm_host.cpp build/comm_kernel.o --cce-fatobj-link -L"$LIB" -L"$ASCEND_HOME_PATH/lib64" \
  -Wl,-rpath,"$LIB" -lshmem -lascendcl -lruntime -o build/comm_probe
bisheng -O2 -std=c++17 -xasc --npu-arch=dav-3510 --cce-auto-infer-kernel-type=false \
  -I"$ASCEND_HOME_PATH/asc/include" -I"$ASCEND_HOME_PATH/include" \
  -I"$SHMEM/include" -I"$SHMEM/src" -c metadata_kernel.cpp -o build/metadata_kernel.o
bisheng -O2 -std=c++17 -xasc --npu-arch=dav-3510 --cce-auto-infer-kernel-type=false \
  -DPROBE_METADATA_DIAGNOSTIC=1 -I"$ASCEND_HOME_PATH/asc/include" -I"$ASCEND_HOME_PATH/include" \
  -I"$SHMEM/include" -I"$SHMEM/src" -c metadata_kernel.cpp -o build/metadata_check.o
bisheng -O2 -std=c++17 -I"$ASCEND_HOME_PATH/include" metadata_host.cpp build/metadata_kernel.o build/metadata_check.o \
  --cce-fatobj-link -L"$ASCEND_HOME_PATH/lib64" -lascendcl -lruntime -o build/metadata_probe
