#!/usr/bin/env bash
# Copyright (c) 2026, Lu Lu
# Modified by zhu-mingzhe71 2026
# Optional fused dependencies stay in this checkout; SHMEM uses the common builder.
set -eo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$SCRIPT_DIR/project_env.sh"
require_cann
set -u
PYTHON=${PYTHON:-python3}
CATCCOS_COMMIT=878c2e0a504be8c49c03b5e1157a7c714a628e29
CATCCOS_DIR="$REPO_ROOT/third-party/catccos"
if [[ "$(project_path "$CATCCOS_DIR")" != "$CATCCOS_DIR" ]]; then
    echo 'CatCCOS must not be an external symlink' >&2; exit 1
fi
if [[ ! -e "$CATCCOS_DIR/.git" ]]; then
    if [[ -e "$CATCCOS_DIR" ]]; then echo 'CatCCOS directory already exists; inspect it first' >&2; exit 1; fi
    (unset LD_LIBRARY_PATH LD_PRELOAD; git clone https://gitcode.com/cann/catccos.git "$CATCCOS_DIR")
    git -C "$CATCCOS_DIR" checkout --detach "$CATCCOS_COMMIT"
fi
if [[ "$(git -C "$CATCCOS_DIR" rev-parse HEAD)" != "$CATCCOS_COMMIT" ]]; then
    echo 'CatCCOS revision differs from the pinned commit; inspect it before updating' >&2; exit 1
fi
(unset LD_LIBRARY_PATH LD_PRELOAD; git -C "$CATCCOS_DIR" submodule update --init --recursive --depth 1 3rdparty/catlass)
cd "$REPO_ROOT"
(unset LD_LIBRARY_PATH LD_PRELOAD; git submodule update --init --recursive --depth 1)
unset SHMEM_ROOT CATLASS_ROOT
"$PYTHON" scripts/build_shmem.py --if-needed
printf '%s\n' 'Dependencies ready inside third-party/.' 'export CATCCOS_SOURCE_PATH=third-party/catccos' 'export SHMEM_SOURCE_PATH=third-party/shmem' 'unset SHMEM_ROOT CATLASS_ROOT' 'export DEEPEP_NPU_ARCH=Ascend950' 'export LD_LIBRARY_PATH="$PWD/third-party/shmem/install/shmem/lib:${LD_LIBRARY_PATH:-}"'
