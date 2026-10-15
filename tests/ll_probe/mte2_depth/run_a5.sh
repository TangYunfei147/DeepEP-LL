#!/usr/bin/env bash
# A5 专用：仅真实设备0/1；每次一个进程、一个AIV，不创建通信域。
set -eo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$HERE/../../../scripts/project_env.sh"
export PROBE_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1
export ASCEND_RT_VISIBLE_DEVICES=0,1 ASCEND_VISIBLE_DEVICES=0,1
exec python3 "$HERE/run.py" "$@"
