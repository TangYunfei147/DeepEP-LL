#!/usr/bin/env bash
# CPU --plan is safe on A5; only an explicit user --target-run expands devices.
set -eo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$HERE/../../scripts/project_env.sh"
set -u
mode=${1:---plan}
if (($#)); then shift; fi
case "$mode" in
  --plan)
    exec python3 "$REPO_ROOT/tests/elastic/test_ep_low_latency.py" --plan "$@"
    ;;
  --target-run) ;;
  *) echo 'Usage: run_ll16.sh --plan [LL args] | --target-run --check|--profile [LL args]' >&2; exit 2 ;;
esac
phase=${1:---check}
if (($#)); then shift; fi
case "$phase" in
  --check) phase_args=(--skip-perf-test --num-tokens 1 8 33 --hidden 256 --num-topk 4 --num-experts 128 --dtype bf16 fp16 fp8 --repeat-combine 3) ;;
  --profile) phase_args=() ;;
  *) echo 'Target phase must be --check or --profile' >&2; exit 2 ;;
esac
# This expansion is for the user target machine only, never A5 validation.
export PROBE_VISIBLE_DEVICES=${LL_TARGET_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
python3 - "$PROBE_VISIBLE_DEVICES" <<'PY'
import sys
values=sys.argv[1].split(',')
if len(values)!=16 or len(set(values))!=16 or any(not v.isdigit() for v in values):
    raise SystemExit('LL_TARGET_DEVICES must contain 16 distinct numeric device IDs')
PY
export ASCEND_RT_VISIBLE_DEVICES="$PROBE_VISIBLE_DEVICES"
export ASCEND_VISIBLE_DEVICES="$PROBE_VISIBLE_DEVICES"
seconds=${LL_TIMEOUT_SECONDS:-1800}
[[ "$seconds" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid LL_TIMEOUT_SECONDS' >&2; exit 2; }
cd "$REPO_ROOT"
# GNU timeout supervises the entire spawned process group, including workers.
timeout --kill-after=15s "${seconds}s" python3 "$REPO_ROOT/tests/elastic/test_ep_low_latency.py" \
  --num-processes 16 --num-nodes 1 "${phase_args[@]}" "$@"
