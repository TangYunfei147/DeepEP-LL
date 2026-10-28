#!/usr/bin/env bash
# 一键：编译（仅依赖 CANN/bisheng，不需要 shmem、catlass）→ 单卡跑 → 出图。
#   bash tests/ll_route_q_bench/run.sh
# 计时口径：每核每轮用 GetSystemCycle 采样，路由表每轮以 CACHE_MODE_DISABLE 从 HBM 重读；
# 默认 100 轮，丢弃前 50 轮，后 50 轮按“最慢核”给出 avg / p50 / p99。
# 被测内容只有一项：生产同款 q 构建（LlBuildRouteQ 语义），按专家区间分核；
# 1 核即等价于 dispatch 里每核全表那一趟，多核则均分专家区间。
# 可透传参数，例如：
#   bash tests/ll_route_q_bench/run.sh --tokens 256 --topk 8 --experts 384 --reps 200 --cores 1,2,4,8,16,32,64
# 环境变量：DEVICE_ID（默认 0）、CANN_ENV（未设置 ASCEND_HOME_PATH 时指向 set_env.sh）。
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
ROOT=$(cd -- "$HERE/../.." && pwd -P)
# 只借用它配置的 ASCEND_HOME_PATH/缓存目录；不链接 shmem。
source "$ROOT/scripts/ll_timer_env.sh"

ARCH=$(uname -m)
OUT="$ROOT/build/ll_route_q_bench"
mkdir -p "$OUT"

INC=(-I"$ROOT/csrc"
     -I"$ASCEND_HOME_PATH/include"
     -I"$ASCEND_HOME_PATH/pkg_inc"
     -I"$ASCEND_HOME_PATH/asc/include"
     -I"$ASCEND_HOME_PATH/include/experiment/runtime"
     -I"$ASCEND_HOME_PATH/pkg_inc/runtime"
     -I"$ASCEND_HOME_PATH/${ARCH}-linux/include"
     -I"$ASCEND_HOME_PATH/compiler/tikcpp"
     -I"$ASCEND_HOME_PATH/compiler/tikcpp/tikcfw"
     -I"$ASCEND_HOME_PATH/compiler/tikcpp/tikcfw/impl"
     -I"$ASCEND_HOME_PATH/compiler/tikcpp/tikcfw/interface")
# --cce-auto-sync=off 与 megamoe 的 VF 编译保持一致；kernel 内的同步全部显式给出。
# 不定义 CATLASS_ARCH、不 include 任何 catlass/shmem 头，也不链接 shmem。
DEVICE_FLAGS=(-O2 -std=c++17 -xasc --npu-arch=dav-3510 --cce-auto-infer-kernel-type=false --cce-auto-sync=off)

bisheng "${DEVICE_FLAGS[@]}" "${INC[@]}" -c "$HERE/route_q_bench_kernel.cpp" -o "$OUT/route_q_bench_kernel.o"
bisheng -O2 -std=c++17 "${INC[@]}" "$HERE/route_q_bench_host.cpp" "$OUT/route_q_bench_kernel.o" --cce-fatobj-link \
    -L"$ASCEND_HOME_PATH/lib64" -lascendcl -lruntime -lm -o "$OUT/route_q_bench"

"$OUT/route_q_bench" --device "${DEVICE_ID:-0}" --out "$OUT/route_q_bench.csv" "$@"
python3 "$HERE/plot_route_q.py" "$OUT/route_q_bench.csv" --out "$OUT/route_q_bench.png"
printf 'artifacts: %s\n' "$OUT"
