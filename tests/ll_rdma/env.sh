#!/usr/bin/env bash
# Source before importing vendor libraries or creating a runtime.
LL_RDMA_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
source "$LL_RDMA_ROOT/scripts/ll_timer_env.sh"
if [[ -n "${DEEPEP_RDMA_PROVIDER_PATH:-}" ]]; then
    ll_rdma_provider="$DEEPEP_RDMA_PROVIDER_PATH"
    if [[ "$ll_rdma_provider" != /* ]]; then ll_rdma_provider="$LL_RDMA_ROOT/$ll_rdma_provider"; fi
    if [[ ! -d "$ll_rdma_provider" ]]; then echo 'Invalid DEEPEP_RDMA_PROVIDER_PATH' >&2; return 1; fi
    export LD_LIBRARY_PATH="$ll_rdma_provider:${LD_LIBRARY_PATH:-}"
fi
# Device selection belongs to the launcher. Do not impose a two-card target default.
