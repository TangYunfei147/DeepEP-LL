#!/usr/bin/env bash
set -eo pipefail
NETWORK_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$NETWORK_DIR/../../../scripts/project_env.sh"
export SHMEM_LOG_LEVEL="${SHMEM_LOG_LEVEL:-INFO}"
set -u
