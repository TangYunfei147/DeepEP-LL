#!/usr/bin/env bash
set -eo pipefail
PROBE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$PROBE_DIR/../../scripts/project_env.sh"
require_cann
set -u
cd "$PROBE_DIR"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" build four_layers/build
SHMEM="$REPO_ROOT/third-party/shmem"
LIB="$SHMEM_LIB_DIR"
require_shmem_sdk
INC=(-I"$ASCEND_HOME_PATH/asc/include" -I"$ASCEND_HOME_PATH/include" -I"$SHMEM/include" -I"$SHMEM/src")
DEV=(-O2 -std=c++17 -fPIC -xasc --npu-arch=dav-3510 --cce-auto-infer-kernel-type=false)
HOST=(-O2 -std=c++17 -fPIC -shared -DLL_PROFILE_BRIDGE)
LINK=(--cce-fatobj-link -L"$LIB" -Wl,-rpath,"$LIB" -L"$ASCEND_HOME_PATH/lib64" -lshmem -lascendcl -lruntime)
python3 four_layers/prepare_snapshot.py
"$BISHENG" "${DEV[@]}" "${INC[@]}" -c transfer_probe.cpp -o build/transfer_profile.o
"$BISHENG" "${DEV[@]}" "${INC[@]}" -c four_layers/comm_kernel.cpp -o four_layers/build/comm_profile.o
"$BISHENG" "${DEV[@]}" "${INC[@]}" -c four_layers/metadata_kernel.cpp -o four_layers/build/metadata_profile.o
"$BISHENG" "${DEV[@]}" "${INC[@]}" -DPROBE_METADATA_DIAGNOSTIC=1 -c four_layers/metadata_kernel.cpp -o four_layers/build/metadata_check_profile.o
"$BISHENG" "${HOST[@]}" "${INC[@]}" peer_probe.cpp profile_bridge.cpp build/transfer_profile.o "${LINK[@]}" -o build/libpeer_profile.so
"$BISHENG" "${HOST[@]}" "${INC[@]}" shmem_probe.cpp profile_bridge.cpp build/transfer_profile.o "${LINK[@]}" -o build/libshmem_profile.so
"$BISHENG" "${HOST[@]}" "${INC[@]}" four_layers/comm_host.cpp profile_bridge.cpp four_layers/build/comm_profile.o "${LINK[@]}" -o four_layers/build/libcomm_profile.so
"$BISHENG" "${HOST[@]}" "${INC[@]}" four_layers/metadata_host.cpp profile_bridge.cpp four_layers/build/metadata_profile.o four_layers/build/metadata_check_profile.o "${LINK[@]}" -o four_layers/build/libmetadata_profile.so
