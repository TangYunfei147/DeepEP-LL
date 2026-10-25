#!/usr/bin/env bash
# Run on each target host. --plan/--help need neither CANN nor a device.
set -e -o pipefail
NDR_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
NDR_PYTHON=${PYTHON:-python3}
for arg in "$@"; do
    case "$arg" in
        --plan|--help|-h)
            exec "$NDR_PYTHON" "$NDR_ROOT/tests/ll_rdma/one_click.py" "$@"
            ;;
    esac
done
# Vendor environment scripts may reference unset variables, so do not use -u.
source "$NDR_ROOT/tests/ll_rdma/env.sh"
exec "$NDR_PYTHON" "$NDR_ROOT/tests/ll_rdma/one_click.py" "$@"
