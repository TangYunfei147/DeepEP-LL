#!/usr/bin/env bash
set -eo pipefail
PROBE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$PROBE_DIR/../../scripts/project_env.sh"
require_cann
set -u
cd "$PROBE_DIR"
mkdir -p build
SDK="$ASCEND_HOME_PATH"
COMPILER="$BISHENG"
"$COMPILER" --version > build/compiler-version.txt
"$COMPILER" -O2 -std=c++17 -fPIC -xasc --npu-arch=dav-3510 \
  --cce-auto-infer-kernel-type=false \
  -I"$SDK/asc/include" -I"$SDK/include" \
  -c transfer_probe.cpp -o build/transfer_probe.o 2>&1 | tee build/compile.log
"$COMPILER" -O2 -std=c++17 -I"$SDK/include" \
  local_smoke.cpp build/transfer_probe.o --cce-fatobj-link \
  -L"$SDK/lib64" -Wl,-rpath,"$SDK/lib64" \
  -lascendcl -lruntime -o build/local_smoke 2>&1 | tee build/link.log
"$COMPILER" -O2 -std=c++17 -I"$SDK/include" \
  peer_probe.cpp build/transfer_probe.o --cce-fatobj-link \
  -L"$SDK/lib64" -Wl,-rpath,"$SDK/lib64" \
  -lascendcl -lruntime -o build/peer_probe 2>&1 | tee build/peer-link.log
