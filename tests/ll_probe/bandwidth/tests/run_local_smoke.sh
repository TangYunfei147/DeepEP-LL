#!/usr/bin/env bash
set -eo pipefail
PROBE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
source "$PROBE_DIR/../../scripts/a5_validation_env.sh"
require_cann
set -u
cd "$PROBE_DIR"
timeout 45s bandwidth/build/local_smoke 2>&1 | tee bandwidth/build/local-smoke.log
