#!/usr/bin/env bash
# Copyright (c) 2026, Lu Lu
# Modified by zhu-mingzhe71 2026
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=${PROJECT_DIR}
PYTHON=${PYTHON:-python3}
CATCCOS_REPOSITORY=https://gitcode.com/cann/catccos.git
CATCCOS_COMMIT=878c2e0a504be8c49c03b5e1157a7c714a628e29
CATCCOS_DIR=${CATCCOS_SOURCE_PATH:-"${REPO_ROOT}/third-party/catccos"}
SHMEM_DIR="${REPO_ROOT}/third-party/shmem"

if [[ ! -d "${CATCCOS_DIR}/.git" ]]; then
    if [[ -e "${CATCCOS_DIR}" ]]; then
        echo "error: ${CATCCOS_DIR} exists but is not a Git checkout" >&2
        exit 1
    fi
    git clone "${CATCCOS_REPOSITORY}" "${CATCCOS_DIR}"
    git -C "${CATCCOS_DIR}" checkout --detach "${CATCCOS_COMMIT}"
else
    ACTUAL_COMMIT=$(git -C "${CATCCOS_DIR}" rev-parse HEAD)
    if [[ "${ACTUAL_COMMIT}" != "${CATCCOS_COMMIT}" ]]; then
        echo "error: CATCCOS_SOURCE_PATH is at ${ACTUAL_COMMIT}, expected ${CATCCOS_COMMIT}" >&2
        echo "Use a clean checkout at the pinned commit or unset CATCCOS_SOURCE_PATH." >&2
        exit 1
    fi
fi

# Only CATLASS is needed from CatCCOS submodules; SHMEM follows the main lock.
git -C "${CATCCOS_DIR}" submodule update --init --recursive 3rdparty/catlass

# Resolve the checked-in submodule through Git. No forced checkout/reset is
# used, so Git refuses a conflicting source update instead of overwriting it.
git -C "${REPO_ROOT}" submodule update --init --recursive --depth 1 third-party/shmem

# Resolve paths before changing directories and before printing shell commands.
CATCCOS_DIR=$(cd -- "${CATCCOS_DIR}" && pwd)
SHMEM_DIR=$(cd -- "${SHMEM_DIR}" && pwd)
DEFAULT_SDK_DIR="${SHMEM_DIR}/install/shmem"
SDK_DIR="${DEFAULT_SDK_DIR}"
# External SDK paths are neither removed nor reused. The shared builder cleans
# only project-owned build/install/ci outputs and builds checked-out source.
unset SHMEM_ROOT SHMEM_SOURCE_PATH
"${PYTHON}" "${SCRIPT_DIR}/build_shmem.py"

echo "CatCCOS: ${CATCCOS_DIR}"
echo "SHMEM source: ${SHMEM_DIR}"
echo "Run these commands in the calling shell before building or testing DeepEP:"
printf 'export CATCCOS_SOURCE_PATH=%q\n' "${CATCCOS_DIR}"
printf 'export SHMEM_SOURCE_PATH=%q\n' "${SHMEM_DIR}"
printf 'export SHMEM_ROOT=%q\n' "${SDK_DIR}"
printf 'export LD_LIBRARY_PATH=%q:${LD_LIBRARY_PATH:-}\n' "${SDK_DIR}/lib"
