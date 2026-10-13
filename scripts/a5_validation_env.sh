#!/usr/bin/env bash
# Source only for A5 validation. Never use as target-machine setup.
export PROBE_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1
export ASCEND_RT_VISIBLE_DEVICES=0,1 ASCEND_VISIBLE_DEVICES=0,1
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/project_env.sh"
# Reassert the A5 boundary after loading any external CANN environment.
export PROBE_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1
export ASCEND_RT_VISIBLE_DEVICES=0,1 ASCEND_VISIBLE_DEVICES=0,1
