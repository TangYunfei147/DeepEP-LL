#!/usr/bin/env bash
set -eo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/env.sh"
REPO=$(cd "$HERE/../../.." && pwd)
export LD_LIBRARY_PATH="${SHMEM_LIB_DIR:-$REPO/tests/ll_probe/build/shmem/lib}:${LD_LIBRARY_PATH:-}"
exec python3 "$HERE/run.py" "$@"
