#!/usr/bin/env bash
set -eo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
ROOT=$(cd -- "$HERE/../../.." && pwd -P)
source "$ROOT/scripts/project_env.sh"
require_cann
set -u
mkdir -p "$HERE/build"
cd "$HERE/build"
BYPASS=${MTE2_DEPTH_BYPASS:-0}
if [[ "$BYPASS" != 0 && "$BYPASS" != 1 ]]; then echo 'MTE2_DEPTH_BYPASS must be 0 or 1' >&2; exit 2; fi
GATE=${MTE2_DEPTH_GATE_REPEATS:-0}
case "$GATE" in 0|16|32|64|128) ;; *) echo 'MTE2_DEPTH_GATE_REPEATS must be 0/16/32/64/128' >&2; exit 2 ;; esac
"$BISHENG" --version > "$HERE/build/compiler-version.txt"
"$BISHENG" -O2 -std=c++17 -save-temps=obj -DMTE2_DEPTH_BYPASS="$BYPASS" \
  -DMTE2_DEPTH_GATE_REPEATS="$GATE" -xasc --npu-arch=dav-3510 \
  --cce-auto-infer-kernel-type=false -I"$ASCEND_HOME_PATH/asc/include" \
  -I"$ASCEND_HOME_PATH/include" -c "$HERE/kernel.cpp" \
  -o "$HERE/build/kernel.o" 2>&1 | tee "$HERE/build/compile.log"
"$BISHENG" -O2 -std=c++17 -I"$ASCEND_HOME_PATH/include" \
  "$HERE/host.cpp" "$HERE/build/kernel.o" --cce-fatobj-link \
  -L"$ASCEND_HOME_PATH/lib64" -Wl,-rpath,"$ASCEND_HOME_PATH/lib64" \
  -lascendcl -lruntime -o "$HERE/build/mte2_depth" 2>&1 | tee "$HERE/build/link.log"
"$ASCEND_HOME_PATH/tools/bisheng_compiler/bin/llvm-objdump" -d \
  "$HERE/build/kernel-cce-hiipu64-hisilicon-cce-dav-c310-vec.o" > "$HERE/build/kernel.dis"
python3 "$HERE/audit_asm.py" > "$HERE/build/asm-audit.log"
python3 "$HERE/record_build.py" --bypass "$BYPASS" --gate-repeats "$GATE"
