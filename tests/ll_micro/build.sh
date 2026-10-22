#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
sdk=${ASCEND_HOME_PATH:?source the machine CANN environment first}
compiler=${BISHENG:-$sdk/bin/bisheng}
mkdir -p "$here/build"
"$compiler" --version > "$here/build/compiler-version.txt"
"$compiler" -O2 -std=c++17 -xasc --npu-arch=dav-3510 --cce-auto-infer-kernel-type=false \
  -I"$sdk/asc/include" -I"$sdk/include" -c "$here/scalar.cpp" -o "$here/build/scalar.o"
"$compiler" -O2 -std=c++17 -I"$sdk/include" "$here/scalar_host.cpp" "$here/build/scalar.o" \
  --cce-fatobj-link -L"$sdk/lib64" -Wl,-rpath,"$sdk/lib64" -lascendcl -lruntime -o "$here/build/scalar_probe"
