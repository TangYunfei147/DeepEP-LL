# kernel/ll-all-send-recv 分段打点

Dispatch固定56核发送 → 业务SyncAll → 56核前缀/接收；combine仍按编译期S/R，默认28/28。S+R必须56，仅控制combine，不要设置56/56。核心标识11000+S、schema8、每核4608B；保留参考64阶段、前两次实际区间及全量metrics。

## 更新与安装

在本分支工程根目录执行；先处理已有未提交修改，多机每台使用相同提交和绝对工程路径：

```bash
git status --short
git switch kernel/ll-all-send-recv
git fetch origin kernel/ll-all-send-recv
git merge --ff-only FETCH_HEAD

# 已加载有效CANN环境时可省略CANN_ENV
export CANN_ENV=/实际的CANN安装目录/set_env.sh
source scripts/ll_timer_env.sh
if [ ! -d .venv-ll-timer ]; then
  python3 -m venv --system-site-packages .venv-ll-timer
fi
source .venv-ll-timer/bin/activate
python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__)"
git submodule update --init --recursive --depth 1
unset SHMEM_ROOT CATLASS_ROOT
export DEEPEP_NPU_ARCH=Ascend950
export DEEPEP_LL_SEND_CORES=28
export DEEPEP_LL_RECV_CORES=28
export DEEPEP_LL_STAGE_TIMER=ON
python scripts/build_shmem.py
python -m pip install -r requirements-build.txt
python -m pip install -v --force-reinstall --no-build-isolation --no-deps .
```

SDK匹配时校验复用，不强制重建。所有rank重装完整wheel；日志/缓存/临时文件位于工程内。首次clone使用`git clone --single-branch --branch kernel/ll-all-send-recv https://github.com/TangYunfei147/DeepEP-LL.git DeepEP-LL-all-send-recv`。

安装检查（不启动通信域）：

```bash
python - <<'PY'
import os, runpy
runpy.run_path('tests/elastic/test_dispatch.py')['_import_installed_deep_ep']()
from deep_ep._native import require_native
n=require_native('all-core timer installation check')
assert n.low_latency_implementation=='all-send-recv'
assert n.low_latency_stage_timer_enabled
assert n.low_latency_core_protocol==11000+int(os.environ['DEEPEP_LL_SEND_CORES'])
assert n.low_latency_stage_timer_layout()['schema']==8
print('PASS protocol',n.low_latency_core_protocol,'schema 8')
PY
```

## 目标机采集与解析

以下为单机EP16示例，设备必须实际获分配；B8/M256/H3584/K8/E384，BF16与FP8配置串行运行，dispatch+combine均采集。FP8配置的combine仍为BF16。每机16连续rank的入口约束保持，不放宽原生EP准入。

```bash
source .venv-ll-timer/bin/activate
source scripts/ll_timer_env.sh
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE
DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
NNODES=1
NODE_RANK=0
MASTER_ADDR=127.0.0.1
RUN_NAME=all-send-recv-segment-001
LL_ARGS=(
  --devices "$DEVICES"
  --num-nodes "$NNODES" --node-rank "$NODE_RANK"
  --master-addr "$MASTER_ADDR" --master-port 29500 --shmem-port 29501
  --num-tokens 8 --num-max-tokens-per-rank 256
  --hidden 3584 --num-topk 8 --num-experts 384 --dtype bf16 fp8 --do-cpu-sync 0
  --warmup 10 --num-iters 100
  --ll-stage-timer --ll-timer-iters 5 --ll-timer-mode observe
  --skip-perf-test --output-dir results/ll-timer --run-name "$RUN_NAME"
)
python tests/ll_stage_timer/run.py "${LL_ARGS[@]}" --plan
python tests/ll_stage_timer/run.py "${LL_ARGS[@]}"
python tests/ll_stage_timer/analyze.py "results/ll-timer/$RUN_NAME" --trace-iteration 4
```

--plan是CPU参数检查，不启动NPU。--skip-perf-test仍执行10预热、100稳定调用后采5轮打点，每轮Exp+scalar all_reduce，默认Exp约13.53GiB/卡。只测dispatch添加--dispatch-only；重跑改RUN_NAME。禁止drain与旧逐专家GM打点。

多机统一NNODES/MASTER_ADDR/端口/RUN_NAME，设置各自NODE_RANK，重新构造LL_ARGS后在各节点执行。非共享目录先按原结构汇集同一run全部rank文件再解析；缺rank或配置不一致会拒绝。

## 报告解释

- metrics.csv：所有调用累计cycles/calls；轮询计数不算耗时。
- intervals.csv / trace.json：实际采到的前两次区间。
- send_phase.csv：区分sender_lane_drain与send_phase_drain_sync，保留是否完整采样。跳过Sender的空分片只有后者，前者留空；不完整采样不推算。
- cores.csv：全部dispatch核为send_and_recv，combine保留S/R角色。
- summary.json / stages.csv：跨轮统计；报告在run下stage-analysis，多个case分目录。

SEND_DRAIN累计已包含拆分项，不能再次相加；阶段同步包括原有PipeBarrier和SyncAll等待，不代表纯网络传输。INIT在KERNEL之外；KERNEL之后的诊断SyncAll/Flush不混入内部计时，但仍进入Host/profiler算子耗时。正式性能另用OFF wheel。

业务流程及验证边界见[ALL_SEND_RECV.md](ALL_SEND_RECV.md)。本次不改收发算法，CANN编译和NPU正确性/性能仍需目标环境验证。

只分析一个算子（统计与trace同时筛选），默认输出stage-analysis-dispatch或stage-analysis-combine：

```bash
python tests/ll_stage_timer/analyze.py "results/ll-timer/$RUN_NAME" --operation dispatch --trace-iteration 4
python tests/ll_stage_timer/analyze.py "results/ll-timer/$RUN_NAME" --operation combine --trace-iteration 4
```

不传--operation仍分析全部算子；所选算子仍检查全部rank和采样轮次，不会跳过缺失rank。此功能纯离线，更新代码即可，无需重装wheel或重采。
