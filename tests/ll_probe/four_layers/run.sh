#!/usr/bin/env bash
set -eo pipefail
FOUR_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$FOUR_DIR/../../../scripts/project_env.sh"
require_cann
set -u
cd "$FOUR_DIR"
cd "$REPO_ROOT"
exec "$@"
