#!/usr/bin/env bash
# Compatibility entry: use the same in-repository SDK as the production wheel.
set -eo pipefail
PROBE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$PROBE_DIR/../../scripts/project_env.sh"
require_cann
set -u
cd "$REPO_ROOT"
exec python3 scripts/build_shmem.py --if-needed "$@"
