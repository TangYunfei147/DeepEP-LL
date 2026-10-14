#!/usr/bin/env bash
set -eo pipefail
PROBE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
source "$PROBE_DIR/../../scripts/project_env.sh"
require_cann
set -u
cd "$PROBE_DIR"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" bandwidth/build communication/build metadata/build
SHMEM="$REPO_ROOT/third-party/shmem"
LIB="$SHMEM_LIB_DIR"
require_shmem_sdk
INC=(-I"$ASCEND_HOME_PATH/asc/include" -I"$ASCEND_HOME_PATH/include" -I"$SHMEM/include" -I"$SHMEM/src")
DEV=(-O2 -std=c++17 -fPIC -xasc --npu-arch=dav-3510 --cce-auto-infer-kernel-type=false)
HOST=(-O2 -std=c++17 -fPIC -shared -DLL_PROFILE_BRIDGE)
LINK=(--cce-fatobj-link -L"$LIB" -Wl,-rpath,"$LIB" -L"$ASCEND_HOME_PATH/lib64" -lshmem -lascendcl -lruntime -lplatform -ltiling_api -lunified_dlog)
python3 metadata/prepare_snapshot.py
"$BISHENG" "${DEV[@]}" "${INC[@]}" -c bandwidth/kernel/transfer_probe.cpp -o bandwidth/build/transfer_profile.o
"$BISHENG" "${DEV[@]}" "${INC[@]}" -c communication/kernel/comm_kernel.cpp -o communication/build/comm_profile.o
"$BISHENG" "${DEV[@]}" "${INC[@]}" -c metadata/kernel/metadata_kernel.cpp -o metadata/build/metadata_profile.o
"$BISHENG" "${DEV[@]}" "${INC[@]}" -DPROBE_METADATA_DIAGNOSTIC=1 -c metadata/kernel/metadata_kernel.cpp -o metadata/build/metadata_check_profile.o
"$BISHENG" "${HOST[@]}" "${INC[@]}" bandwidth/host/peer_probe.cpp profiling/host/profile_bridge.cpp bandwidth/build/transfer_profile.o "${LINK[@]}" -o bandwidth/build/libpeer_profile.so
"$BISHENG" "${HOST[@]}" "${INC[@]}" bandwidth/host/shmem_probe.cpp profiling/host/profile_bridge.cpp bandwidth/build/transfer_profile.o "${LINK[@]}" -o bandwidth/build/libshmem_profile.so
"$BISHENG" "${HOST[@]}" "${INC[@]}" communication/host/comm_host.cpp profiling/host/profile_bridge.cpp communication/build/comm_profile.o "${LINK[@]}" -o communication/build/libcomm_profile.so
"$BISHENG" "${HOST[@]}" "${INC[@]}" metadata/host/metadata_host.cpp profiling/host/profile_bridge.cpp metadata/build/metadata_profile.o metadata/build/metadata_check_profile.o "${LINK[@]}" -o metadata/build/libmetadata_profile.so
