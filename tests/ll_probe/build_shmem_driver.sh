#!/usr/bin/env bash
set -eo pipefail
ROOT=/home/tyf/Q4_remote_test
source "$ROOT/env_1009/cann-9.3.0/set_env.sh"
set -u
export TMPDIR="$ROOT/.tmp" XDG_CACHE_HOME="$ROOT/.cache"
cd "$ROOT/DeepEP-LL/tests/ll_probe"
SOURCE="$ROOT/DeepEP-LL/third-party/shmem"
LIB="$PWD/build/shmem/lib"
test -f "$LIB/libshmem.so"
bisheng -O2 -std=c++17 -I"$ASCEND_HOME_PATH/include" -I"$SOURCE/include" \
    shmem_probe.cpp build/transfer_probe.o --cce-fatobj-link \
    -L"$LIB" -L"$ASCEND_HOME_PATH/lib64" -Wl,-rpath,"$LIB" \
    -lshmem -lascendcl -lruntime -o build/shmem_probe 2>&1 | tee build/shmem-driver-link.log
