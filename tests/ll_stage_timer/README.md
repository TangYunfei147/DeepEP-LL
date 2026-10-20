> 当前新增输入/量化等待使用 **schema9、66阶段**。原64个ID保持不变；旧schema7可离线读取。新采样不再导出仅支持64阶段的参考format5二进制，使用JSON/CSV/Chrome trace。瓶颈报告与重采要求见 [INPUT_WAIT.md](INPUT_WAIT.md)。

# 最新共享接收分支：分段打点全流程

当前源端算法基线为`a192113`：S核跨组初发，R核本组初发后进入原前缀与接收流程。分段打点使用schema9、66阶段、前两次实际区间+全量累计；本次补齐本组准备/排空和非去重远端索引筛选。二进制头保留初始角色，分析器依据实际SEND_TOKENS把参与本组发送的R核显示为send_and_recv，并保留initial_role=cumsum；无本组发送任务的核不补造区间。

下例为**目标机单机16rank**、B8/M256/H3584/K8/E384、BF16和FP8两组配置，dispatch+combine均采集；FP8配置的combine仍是BF16。默认G4/S28/R28。多机入口要求每机16个连续rank，不改变已有硬件/EP准入。

## 1. 获取或更新代码

首次部署，在选定工程父目录执行：

```bash
git clone --single-branch --branch kernel/ll-g4-shared-recv \
  https://github.com/TangYunfei147/DeepEP-LL.git DeepEP-LL-g4-shared-recv
cd DeepEP-LL-g4-shared-recv
```

已有checkout，则进入其根目录执行下面这组，先确认没有未提交修改；不重置、不覆盖本地改动：

```bash
git status --short
git switch kernel/ll-g4-shared-recv
git fetch origin kernel/ll-g4-shared-recv
git merge --ff-only FETCH_HEAD
git rev-parse HEAD
```

以下所有命令都在该工程根目录执行。多机每台同提交、同工程绝对路径、同编译配置。

## 2. 加载环境并完整安装ON wheel

CANN_ENV填写目标机真实环境脚本；已经加载CANN且ASCEND_HOME_PATH有效时可省略该export。使用工程内venv复用目标机已有torch/torch_npu：

```bash
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

环境脚本已将日志、缓存、临时文件定向工程内并加入工程SDK库路径，不默认选卡。SDK身份匹配时复用，不删除构建缓存或强制重建SDK。全rank重装完整wheel；当前核心标识10000+S、schema9。

## 3. 检查实际安装包

这一步核对安装包的能力，不初始化通信域、不替代设备正确性验收：

```bash
python - <<'PY'
import os, runpy
runpy.run_path('tests/elastic/test_dispatch.py')['_import_installed_deep_ep']()
import deep_ep
from deep_ep._native import require_native
from deep_ep.utils.ll_stage_timer import SCHEMA, SLOTS, FIELDS, STAGES
n = require_native('segment timer installation check')
s = int(os.environ['DEEPEP_LL_SEND_CORES'])
r = int(os.environ['DEEPEP_LL_RECV_CORES'])
assert s > 0 and r > 0 and s + r == 56
assert n.low_latency_implementation == 'g4-shared-recv'
assert n.low_latency_stage_timer_enabled
assert n.low_latency_core_protocol == 10000 + s
assert n.low_latency_ranks_per_group == 4
layout = n.low_latency_stage_timer_layout()
assert layout == dict(schema=SCHEMA, slots=SLOTS, fields=FIELDS, stages=list(STAGES))
assert (SCHEMA, SLOTS, FIELDS) == (9, 72, 8)
print('installed:', deep_ep.__file__)
print('protocol:', n.low_latency_core_protocol, 'layout:', (SCHEMA, SLOTS, FIELDS))
PY
```

## 4. 检查计划并采集

新终端需重新激活venv和source环境脚本。下面设备必须是实际获分配的目标设备。正式入口不接受drain或逐专家GM打点；重跑必须修改RUN_NAME。

```bash
source .venv-ll-timer/bin/activate
source scripts/ll_timer_env.sh
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE
DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
NNODES=1
NODE_RANK=0
MASTER_ADDR=127.0.0.1
RUN_NAME=segment-local-b8-001
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
```

--plan仅检查CPU参数规划，不能作为NPU验收。--skip-perf-test关闭正式profiler轮次，但仍执行10预热+100稳定调用，再采5轮独立打点；每轮Exp+scalar all_reduce，保持计时外校验。默认Exp约13.53GiB/卡，并保留诊断输出。只测dispatch时在LL_ARGS中加--dispatch-only；只测BF16时把--dtype改为bf16。

双机EP32时，各节点先完成安装检查；统一NNODES=2、MASTER_ADDR=首机实际可达IP、RUN_NAME和端口，NODE_RANK分别0/1，然后在每台重新定义LL_ARGS并执行。四机EP64同理NNODES=4、NODE_RANK=0..3。入口不通过SSH替用户启动其他节点，跨机MTE须有真实可用peer映射。

## 5. 解析全部rank

单机可直接解析；多机先把同一run各节点的rank文件按原相对目录合并到一个run，保留公共cases.json，不覆盖不同实验文件。缺rank或重复/错配数据会拒绝。

```bash
python tests/ll_stage_timer/analyze.py "results/ll-timer/$RUN_NAME" --trace-iteration 4
```

这会汇总全部5轮数据，并展示第5轮（编号4）的采样时间线。结果位于run下stage-analysis，多个case会分子目录：

- summary.json / stages.csv：跨轮统计。
- metrics.csv：全量累计cycles/calls；轮询次数等行time_measured=false。
- intervals.csv：各阶段前两次真实区间。
- cores.csv：实际角色、initial_role、记录数、省略区间数。
- trace.json：Chrome/Perfetto时间线。
- 原始JSON旁的dispatch_prof_rankN_sample_*.bin / combine同类文件：参考format5二进制。

重点看R核SEND_TOKENS内的PREPARE_INPUT、TOKEN_*、SEND_DRAIN，以及之后的CAL_CUMSUM/WAIT_CUMSUM/LOCAL_WINDOW_COPY。R核本组q已融合COUNT_COMPUTE，不应期待逐包CAL_TOKEN。SEND_TOKENS是父阶段，不与其子阶段相加。

正式性能另装OFF wheel、不传--ll-stage-timer，沿用10/100/每完整task最后50条。ON profiler包含诊断同步与写出，不能替代OFF基线。更多语义见[SEGMENT_TIMER.md](SEGMENT_TIMER.md)与[GROUP_LOCAL_SEND.md](GROUP_LOCAL_SEND.md)。

跨轮汇总按initial_role保持同核同阶段唯一统计项；某轮有本组发送、其他轮为空时，不拆分KERNEL统计。汇总角色表示该采样集合中实际参与过发送，cores.csv仍逐轮保留角色。

## 按算子筛选离线分析

```bash
python tests/ll_stage_timer/analyze.py <run目录> --op dispatch --trace-iteration 4
python tests/ll_stage_timer/analyze.py <run目录> --op combine --trace-iteration 4
```

`--operation`是`--op`的同义写法；不传则保留全部算子。筛选覆盖summary、CSV、trace及本分支附加报告，所选算子的全部rank/轮次仍必须完整。默认分别输出stage-analysis-dispatch、stage-analysis-combine；可用--output-dir指定新目录，已有输出不覆盖。仅离线脚本更新，无需重编wheel或重新采样。
