#!/usr/bin/env bash
# Build experiment-only libraries, without installing or changing production SDK.
set -eo pipefail
ROOT=/home/tyf/Q4_remote_test
source "$ROOT/env_1009/cann-9.3.0/set_env.sh"
set -u
export TMPDIR="$ROOT/.tmp" XDG_CACHE_HOME="$ROOT/.cache" CCACHE_DIR="$ROOT/.cache/ccache"
export CUDA_VISIBLE_DEVICES=0,1 ASCEND_RT_VISIBLE_DEVICES=0,1 ASCEND_VISIBLE_DEVICES=0,1
SOURCE="$ROOT/DeepEP-LL/third-party/shmem"
BUILD="$ROOT/DeepEP-LL/tests/ll_probe/build/shmem"
test "$(git -C "$SOURCE" rev-parse HEAD)" = 4dd80e2129da42bd773982c2bb3758af1ed0bba8
test -z "$(git -C "$SOURCE" diff --name-only HEAD)"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$BUILD"
# Same JSON tag used by the pinned SHMEM build script. Never clear an existing tree.
if [ ! -f "$SOURCE/3rdparty/json/single_include/nlohmann/json.hpp" ]; then
    test ! -e "$SOURCE/3rdparty/json"
    mkdir -p "$SOURCE/3rdparty"
    (unset LD_LIBRARY_PATH LD_PRELOAD; git clone --branch v3.11.3 --depth 1 \
        https://gitcode.com/GitHub_Trending/js/json.git "$SOURCE/3rdparty/json")
fi
git -C "$SOURCE/3rdparty/json" rev-parse HEAD > "$BUILD/json-revision.txt"
cmake -S "$SOURCE" -B "$BUILD" -DSOC_TYPE=Ascend950 \
    -DENABLE_CANN_BUILD=ON -DCMAKE_BUILD_TYPE=Release \
    -DUSE_EXAMPLES=OFF -DUSE_UNIT_TEST=OFF \
    2>&1 | tee "$BUILD/configure.log"
cmake --build "$BUILD" --parallel 4 --target shmem aclshmem_bootstrap_config_store \
    2>&1 | tee "$BUILD/build.log"
git -C "$SOURCE" rev-parse HEAD > "$BUILD/shmem-revision.txt"
sha256sum "$BUILD/lib/libshmem.so" > "$BUILD/library-sha256.txt"
