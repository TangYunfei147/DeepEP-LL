#!/usr/bin/env bash
set -eo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$HERE/env.sh"
set -u
SDK="$LL_RDMA_ROOT/third-party/shmem/install/shmem"
OUT="$LL_RDMA_ROOT/build/ll_rdma"
mkdir -p "$OUT"
python3 - "$SDK" <<'PY'
import json,sys
from pathlib import Path
m=json.loads((Path(sys.argv[1])/'deepep-sdk.json').read_text())
assert m['build'].get('rdma_backend')=='HNS_1825', 'Build the explicit HNS_1825 SDK first'
assert m['build'].get('rdma_patch')=='deepep-roce-72-v1', 'Build the SDK with the project ROCE QP policy patch'
PY
INC=(-I"$LL_RDMA_ROOT/csrc" -I"$ASCEND_HOME_PATH/asc/include" -I"$ASCEND_HOME_PATH/include"
     -I"$SDK/include" -I"$SDK/src" -I"$SDK/src/device")
FLAGS=(-DDEEPEP_LL_RDMA=1 -DACLSHMEMI_RDMA_K_BACKEND_HNS_1825=1 -DCATLASS_ARCH=3510 -DL2_CACHE_HINT)
bisheng -O2 -std=c++17 -xasc --npu-arch=dav-3510 --cce-auto-infer-kernel-type=false \
    "${FLAGS[@]}" "${INC[@]}" -c "$HERE/smoke_device.cpp" -o "$OUT/smoke_device.o"
bisheng -O2 -std=c++17 "${INC[@]}" "$HERE/smoke_host.cpp" "$OUT/smoke_device.o" --cce-fatobj-link \
    -L"$ASCEND_HOME_PATH/lib64" -L"$SDK/lib" -lascendcl -lruntime -lshmem -o "$OUT/rdma_smoke"
printf 'Built %s\n' "$OUT/rdma_smoke"
