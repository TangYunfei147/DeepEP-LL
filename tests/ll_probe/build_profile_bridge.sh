#!/usr/bin/env bash
# Python profiler integration; all artifacts remain in the authorized checkout.
set -eo pipefail
ROOT=/home/tyf/Q4_remote_test
source "$ROOT/env_1009/cann-9.3.0/set_env.sh"
set -u
cd "$ROOT/DeepEP-LL/tests/ll_probe"
export TMPDIR="$ROOT/.tmp" XDG_CACHE_HOME="$ROOT/.cache"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" build four_layers/build
SHMEM="$ROOT/DeepEP-LL/third-party/shmem"
LIB="$PWD/build/shmem/lib"
test -f "$LIB/libshmem.so"
INC=(-I"$ASCEND_HOME_PATH/asc/include" -I"$ASCEND_HOME_PATH/include" -I"$SHMEM/include" -I"$SHMEM/src")
DEV=(-O2 -std=c++17 -fPIC -xasc --npu-arch=dav-3510 --cce-auto-infer-kernel-type=false)
HOST=(-O2 -std=c++17 -fPIC -shared -DLL_PROFILE_BRIDGE)
LINK=(--cce-fatobj-link -L"$LIB" -Wl,-rpath,"$LIB" -L"$ASCEND_HOME_PATH/lib64" -lshmem -lascendcl -lruntime)
python3 four_layers/prepare_snapshot.py
bisheng "${DEV[@]}" "${INC[@]}" -c transfer_probe.cpp -o build/transfer_profile.o
bisheng "${DEV[@]}" "${INC[@]}" -c four_layers/comm_kernel.cpp -o four_layers/build/comm_profile.o
bisheng "${DEV[@]}" "${INC[@]}" -c four_layers/metadata_kernel.cpp -o four_layers/build/metadata_profile.o
bisheng "${DEV[@]}" "${INC[@]}" -DPROBE_METADATA_DIAGNOSTIC=1 -c four_layers/metadata_kernel.cpp -o four_layers/build/metadata_check_profile.o
bisheng "${HOST[@]}" "${INC[@]}" peer_probe.cpp profile_bridge.cpp build/transfer_profile.o "${LINK[@]}" -o build/libpeer_profile.so
bisheng "${HOST[@]}" "${INC[@]}" shmem_probe.cpp profile_bridge.cpp build/transfer_profile.o "${LINK[@]}" -o build/libshmem_profile.so
bisheng "${HOST[@]}" "${INC[@]}" four_layers/comm_host.cpp profile_bridge.cpp four_layers/build/comm_profile.o "${LINK[@]}" -o four_layers/build/libcomm_profile.so
bisheng "${HOST[@]}" "${INC[@]}" four_layers/metadata_host.cpp profile_bridge.cpp four_layers/build/metadata_profile.o four_layers/build/metadata_check_profile.o "${LINK[@]}" -o four_layers/build/libmetadata_profile.so
