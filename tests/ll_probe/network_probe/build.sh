#!/usr/bin/env bash
set -eo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/env.sh"
REPO=$(cd "$HERE/../../.." && pwd)
case "$REPO/" in "$PROBE_ROOT/"*) ;; *) echo 'Checkout must be inside PROBE_ROOT' >&2; exit 2;; esac
SHMEM="$REPO/third-party/shmem"
SHMEM_LIB_DIR="${SHMEM_LIB_DIR:-$REPO/tests/ll_probe/build/shmem/lib}"
export SHMEM_LIB_DIR
export LD_LIBRARY_PATH="$SHMEM_LIB_DIR:${LD_LIBRARY_PATH:-}"
test -f "$SHMEM_LIB_DIR/libshmem.so"
: "${ASCEND_HOME_PATH:?Load CANN via CANN_ENV or the existing environment}"
mkdir -p "$HERE/build"
INC=(-I"$REPO/csrc" -I"$SHMEM/include" -I"$SHMEM/src" -I"$ASCEND_HOME_PATH/include" -I"$ASCEND_HOME_PATH/asc/include")
bisheng -O2 -std=c++17 -xasc --npu-arch=dav-3510 --cce-auto-infer-kernel-type=false \
    "${INC[@]}" -c "$HERE/copy.cpp" -o "$HERE/build/copy.o"
bisheng -O2 -std=c++17 "${INC[@]}" "$HERE/probe.cpp" "$REPO/csrc/runtime/session.cpp" \
    "$HERE/build/copy.o" --cce-fatobj-link -L"$SHMEM_LIB_DIR" -Wl,-rpath,"$SHMEM_LIB_DIR" \
    -L"$ASCEND_HOME_PATH/lib64" -lshmem -lascendcl -lruntime -ldl -lpthread -o "$HERE/build/network_probe"
python3 "$HERE/run.py" --write-build-manifest
