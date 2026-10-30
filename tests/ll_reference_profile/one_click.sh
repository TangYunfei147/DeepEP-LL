#!/usr/bin/env bash
# Invoke with bash; the caller's interactive shell remains open on failure.
set -e -o pipefail
REFERENCE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
REFERENCE_PYTHON=${PYTHON:-python3}
if [[ "${1:-}" == "--analyze-only" ]]; then
  shift
  exec "$REFERENCE_PYTHON" "$REFERENCE_ROOT/tests/ll_reference_profile/analyze.py" "$@"
fi
for arg in "$@"; do
  case "$arg" in --plan|--help|-h) exec "$REFERENCE_PYTHON" "$REFERENCE_ROOT/tests/ll_reference_profile/one_click.py" "$@" ;; esac
done
if [[ -n "${CANN_ENV:-}" ]]; then
  source "$CANN_ENV"
fi
exec "$REFERENCE_PYTHON" "$REFERENCE_ROOT/tests/ll_reference_profile/one_click.py" "$@"
