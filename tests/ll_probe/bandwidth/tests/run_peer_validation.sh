#!/usr/bin/env bash
set -eo pipefail
PROBE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
source "$PROBE_DIR/../../scripts/a5_validation_env.sh"
require_cann
set -u
cd "$PROBE_DIR"
timeout 45s bandwidth/build/peer_probe 15392 8 4 4096 3 0 0 2>&1 | tee bandwidth/build/peer-validation.log
