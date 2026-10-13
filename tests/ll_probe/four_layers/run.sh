#!/usr/bin/env bash
set -eo pipefail
FOUR_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$FOUR_DIR/../../../scripts/project_env.sh"
# Optional explicit probe selection; absence preserves the caller's visibility.
if [[ -n "${PROBE_VISIBLE_DEVICES:-}" ]]; then
    export ASCEND_RT_VISIBLE_DEVICES="$PROBE_VISIBLE_DEVICES"
    export ASCEND_VISIBLE_DEVICES="$PROBE_VISIBLE_DEVICES"
fi
require_cann
set -u
cd "$FOUR_DIR"
cd "$REPO_ROOT"
exec "$@"
