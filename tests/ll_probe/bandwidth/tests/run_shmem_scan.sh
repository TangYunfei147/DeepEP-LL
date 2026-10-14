#!/usr/bin/env bash
set -eo pipefail
PROBE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
source "$PROBE_DIR/../../scripts/project_env.sh"
require_cann
set -u
cd "$PROBE_DIR"
cd "$REPO_ROOT"
exec python3 "$PROBE_DIR/bandwidth/tests/scan_shmem.py" "$@"
