#!/usr/bin/env bash
set -eo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/env.sh"
REPO=$(cd "$HERE/../../.." && pwd)
export LD_LIBRARY_PATH="$REPO/third-party/shmem/install/shmem/lib:${LD_LIBRARY_PATH:-}"
cd "$REPO_ROOT"
exec python3 "$HERE/run.py" "$@"
