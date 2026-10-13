#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/tyf/Q4_remote_test
cd "$ROOT/DeepEP-LL/tests/ll_probe"
export TMPDIR="$ROOT/.tmp" XDG_CACHE_HOME="$ROOT/.cache"
export CUDA_VISIBLE_DEVICES=0,1 ASCEND_RT_VISIBLE_DEVICES=0,1 ASCEND_VISIBLE_DEVICES=0,1
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" build
SDK="$ROOT/env_1009/cann-9.3.0"
export ASCEND_HOME_PATH="$SDK"
COMPILER="$SDK/tools/bisheng_compiler/bin/bisheng"
"$COMPILER" --version > build/compiler-version.txt
"$COMPILER" -O2 -std=c++17 -fPIC -xasc --npu-arch=dav-3510 \
  --cce-auto-infer-kernel-type=false \
  -I"$SDK/x86_64-linux/asc/include" -I"$SDK/x86_64-linux/include" \
  -c transfer_probe.cpp -o build/transfer_probe.o 2>&1 | tee build/compile.log
"$COMPILER" -O2 -std=c++17 -I"$SDK/x86_64-linux/include" \
  local_smoke.cpp build/transfer_probe.o --cce-fatobj-link \
  -L"$SDK/x86_64-linux/lib64" -Wl,-rpath,"$SDK/x86_64-linux/lib64" \
  -lascendcl -lruntime -o build/local_smoke 2>&1 | tee build/link.log
"$COMPILER" -O2 -std=c++17 -I"$SDK/x86_64-linux/include" \
  peer_probe.cpp build/transfer_probe.o --cce-fatobj-link \
  -L"$SDK/x86_64-linux/lib64" -Wl,-rpath,"$SDK/x86_64-linux/lib64" \
  -lascendcl -lruntime -o build/peer_probe 2>&1 | tee build/peer-link.log
