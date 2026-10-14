#!/usr/bin/env bash
set -eo pipefail
PROBE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
source "$PROBE_DIR/../../scripts/project_env.sh"
require_cann
set -u
cd "$PROBE_DIR"
mkdir -p bandwidth/build
SDK="$ASCEND_HOME_PATH"
COMPILER="$BISHENG"
"$COMPILER" --version > bandwidth/build/compiler-version.txt
"$COMPILER" -O2 -std=c++17 -fPIC -xasc --npu-arch=dav-3510 \
  --cce-auto-infer-kernel-type=false \
  -I"$SDK/asc/include" -I"$SDK/include" \
  -c bandwidth/kernel/transfer_probe.cpp -o bandwidth/build/transfer_probe.o 2>&1 | tee bandwidth/build/compile.log
"$COMPILER" -O2 -std=c++17 -I"$SDK/include" \
  bandwidth/host/local_smoke.cpp bandwidth/build/transfer_probe.o --cce-fatobj-link \
  -L"$SDK/lib64" -Wl,-rpath,"$SDK/lib64" \
  -lascendcl -lruntime -o bandwidth/build/local_smoke 2>&1 | tee bandwidth/build/link.log
"$COMPILER" -O2 -std=c++17 -I"$SDK/include" \
  bandwidth/host/peer_probe.cpp bandwidth/build/transfer_probe.o --cce-fatobj-link \
  -L"$SDK/lib64" -Wl,-rpath,"$SDK/lib64" \
  -lascendcl -lruntime -o bandwidth/build/peer_probe 2>&1 | tee bandwidth/build/peer-link.log
