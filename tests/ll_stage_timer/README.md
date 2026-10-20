# LL阶段与专家诊断（泛化分支）

> 2026-10-19：本分支现在按编译配置分配固定56个AIV，默认28发+28收，支持不对称比例；以[CORE_SPLIT.md](CORE_SPLIT.md)为准。旧1:1和动态核数说明仅描述此前版本。全rank重建匹配wheel。

维护分支kernel/ll-configurable-groups。实验、采样、解析与NDR分支共用同一套代码，
推荐直接按[PARITY.md](PARITY.md)启动和对比。

本分支保留MTE传输、原始静态分组默认G=4；不包含RDMA data plane，也不采用NDR的G4/16 tiling策略。
G=16与G=4可以直接作为比较变量，工具记录实际G而不要求相同。

## 从克隆仓库到执行

以下步骤在**用户目标机**执行。每台参与节点都需准备同一分支、同一提交和匹配wheel；单机EP16，两机EP32，每机16个连续rank。原生LL不支持EP2/4/8。
前置环境：Linux、Python≥3.11、C++17编译器、CMake≥3.20、Make或Ninja，以及目标机器配套CANN、驱动、Torch/torch_npu。CANN 9.3.0按已确认支持准备，不猜测框架版本；requirements-build.txt只安装构建依赖。

### 1. 拉取指定分支

在计划存放工程的父目录执行；两分支建议使用独立目录：

```bash
git clone --branch kernel/ll-configurable-groups --single-branch \
  https://github.com/TangYunfei147/DeepEP-LL.git DeepEP-LL-groups
cd DeepEP-LL-groups
git submodule update --init --recursive --depth 1
git rev-parse HEAD
```

多机时各节点使用相同工程路径，并核对输出提交一致；后续全部命令在该仓库根目录执行。

### 2. 准备环境并构建timer wheel

先将下面CANN_ENV替换为本机真实工具链脚本路径；已加载CANN环境时可省略该export。
以下venv复用基础Python环境中的框架包；若导入检查失败，先在此工程内环境安装目标机配套Torch/torch_npu，再继续。

```bash
export CANN_ENV=/实际的CANN安装目录/set_env.sh
source scripts/ll_timer_env.sh
python3 -m venv --system-site-packages .venv-ll-timer
source .venv-ll-timer/bin/activate
python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__)"

unset SHMEM_ROOT CATLASS_ROOT
export DEEPEP_NPU_ARCH=Ascend950
python scripts/build_shmem.py
export DEEPEP_LL_STAGE_TIMER=ON
python -m pip install -r requirements-build.txt
python -m pip install -v --force-reinstall --no-build-isolation --no-deps .
```

环境脚本将日志/缓存/临时文件放在工程内，并添加仓内SHMEM库路径；不自动选卡。SDK匹配时自动校验复用，身份不匹配时按报错处理，停止使用者后才显式--rebuild。每次换提交或修改源码后都需重装wheel；普通增量构建不删除build目录。

### 3. 单机EP16计划检查与执行

新终端先重新激活环境、加载CANN。devices必须按本机已分配设备明确选择；下面列出目标机16卡，runner统一设置四个设备可见性变量，无需重复export。
本分支默认MTE、逻辑组G=16。多机运行需要目标环境确实支持跨机peer映射；不能把机内MTE测试当成跨机通信验收。

```bash
source .venv-ll-timer/bin/activate
source scripts/ll_timer_env.sh
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE
DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
NNODES=1
NODE_RANK=0
MASTER_ADDR=127.0.0.1
RUN_NAME=timer-001

LL_ARGS=(
  --devices "$DEVICES"
  --num-nodes "$NNODES" --node-rank "$NODE_RANK"
  --master-addr "$MASTER_ADDR" --master-port 29500 --shmem-port 29501
  --num-tokens 32 --num-max-tokens-per-rank 256
  --hidden 7168 --num-topk 6 --num-experts 256 --dtype bf16 --do-cpu-sync 0
  --warmup 10 --num-iters 100
  --ll-stage-timer --ll-timer-iters 5 --ll-timer-mode observe
  --skip-perf-test --output-dir results/ll-timer --run-name "$RUN_NAME"
)
python tests/ll_stage_timer/run.py "${LL_ARGS[@]}" --plan
python tests/ll_stage_timer/run.py "${LL_ARGS[@]}"
```

--plan只检查CPU计划，不验证设备或互联。实际运行仍执行10轮预热、100轮稳定调用，再采5轮打点；每轮Exp+scalar all_reduce，默认Exp约13.53GiB/卡，保留打点输出也占显存。输出目录不能已存在，重复实验请改RUN_NAME。
需要专家统计时，把数组中的--ll-stage-timer改为--ll-expert-timer，阶段capture会自动开启。只测dispatch加--dispatch-only。

### 4. 双机EP32

每台节点完成步骤1、2，确认匹配的提交/wheel、可达地址与端口。将步骤3的NNODES改为2，两台NODE_RANK分别为0、1，MASTER_ADDR都改为首节点可达IP，RUN_NAME和其他参数完全相同；然后**重新定义LL_ARGS数组**并在两台同时执行实际命令。入口不通过SSH自动启动另一台。
本分支没有RDMA后端，不能传--backend rdma。MTE跨机不可用时记录为不可用基线，不计算虚构加速比。

### 5. 分析结果

单机完成后直接执行：

```bash
python tests/ll_stage_timer/analyze.py results/ll-timer/timer-001 --trace-iteration 4
```

若改了RUN_NAME，分析路径也相应修改。多机先将同一run各节点的rank数据按原相对目录合并到一个run目录，保留公共manifest；不要覆盖不同rank文件或混入其他实验。缺rank或配置不匹配时分析器会拒绝处理。
默认生成stage-analysis下的summary.json、stages.csv和trace.json；多case分子目录。开启专家统计还会生成experts.csv、expert_cores.csv。trace.json可导入Chrome tracing/Perfetto；跨rank时间戳不对齐。

### 6. 后续Git更新与正式性能

```bash
git status --short
git pull --ff-only
git submodule update --init --recursive --depth 1
source .venv-ll-timer/bin/activate
source scripts/ll_timer_env.sh
unset SHMEM_ROOT CATLASS_ROOT
export DEEPEP_NPU_ARCH=Ascend950
python scripts/build_shmem.py
export DEEPEP_LL_STAGE_TIMER=ON
python -m pip install -v --force-reinstall --no-build-isolation --no-deps .
```

先确认或妥善保留本地修改，不使用reset/clean覆盖它们。所有节点更新到同一提交；构建依赖变更时重跑requirements-build.txt安装。
正式未插桩基线改为DEEPEP_LL_STAGE_TIMER=OFF重装wheel，并从执行数组中移除timer/专家参数及--skip-perf-test，保留10/100采样（perf默认最后50条）。ON打点耗时不能充当OFF基线。

## 格式与构建

核心协议标识7000+发送核数、阶段schema5、专家schema1；全rank重建同一分支的匹配wheel。
每核阶段缓冲2048B、512B对齐；专家统计有独立可选缓冲。
设置DEEPEP_LL_STAGE_TIMER=ON启用；仅阶段schema1旧数据也可离线解析。
泛化分支默认MTE构建；python scripts/build_shmem.py自动校验并复用已有SDK，包括身份匹配的HNS能力超集。
无需重复--if-needed；不匹配时保留显式--rebuild保护。详见[CONFIGURATION_NOTES.md](CONFIGURATION_NOTES.md)。

## 采样

有perf时先10次预热、100次Exp+scalar all_reduce性能采集，统计最后50条，
关闭profiler后再追加ll-timer-iters次内部打点；无perf时先同样的预热和稳定调用。
默认5轮打点，最后一轮编号4；全部完成后才复制/校验/导出。
默认Exp约13.53GiB/卡，保留多轮输出会增加显存。

observe不新增流水屏障；drain在阶段边界排空，改变重叠与等待成本。
目标计数器按cycles/1000 us换算，原始ticks保留。父子/跨核累计不可当作端到端时延相加。

## 子阶段

recv_consume细分recv_lane_wait、recv_gm_to_ub、recv_load_wait、recv_metadata、
recv_output_wait、recv_output_write、recv_forward。
send_pack细分send_input_gm_to_ub、send_input_wait、send_tail、send_lane_wait；
send_issue下的send_mte_write记录MTE写入下发。
共享schema还保留send_rdma_*名称，但本分支不执行这些阶段，缺失数据不补0。

## 专家统计

增加--ll-expert-timer（自动启用阶段capture），记录执行rank/core/专家的packet计数、
metadata、output_write及按目标专家记录的FIRST_HOP forward。
共享首跳准备、relayReadIndex和批量等待不分摊给各专家；不表示FFN/GEMM耗时。
专家缓冲每轮字节数为实际AIV×(8+全局专家数×4)×8×8。

## 分析

```bash
python tests/ll_stage_timer/analyze.py results/ll-stage-001 --latest --trace-iteration 4
```

也可在测试时--run-name固定目录。输出summary.json、stages.csv、trace.json；
启用专家统计时增加experts.csv、expert_cores.csv。
默认每AIV一轨道，各rank/调用独立归零；没有跨rank时钟对齐。
六类重复发送包络默认隐藏，累计统计保留，--show-repeated可恢复。
新公共compare.py会记录不同G，不强制分组一致；步骤见PARITY.md。

本次同步只完成本地CPU回归，尚未做CANN编译或目标设备验收。

## G4基线与错峰候选（2026-10-19）

本分支默认逻辑组已改为4，保持MTE、原发送顺序、阶段schema5/专家schema1。所有受支持EP均采用同一个编译期G，不引入NDR的按world选G策略。物理每机16rank的约定及原生EP准入保持不变。

更新后全rank重装匹配wheel；仅改变逻辑G不要求重建已通过校验的SDK。若此前显式定义了DEEPEP_LL_RANKS_PER_GROUP，该值会覆盖新默认，比较前核对运行记录中的logical_group。不要在同一作业内混用G4和G16 wheel。

```bash
git pull --ff-only
source scripts/ll_timer_env.sh
export DEEPEP_NPU_ARCH=Ascend950
export DEEPEP_LL_STAGE_TIMER=OFF
python -m pip install -v --force-reinstall --no-build-isolation --no-deps .
```

错峰候选独立维护在kernel/ll-g4-stagger，基于本分支G4提交，仅移植NDR发送顺序轮转，不移植RDMA、缓存策略、分组tiling或timer schema3。基线与候选用独立checkout和匹配环境，在相同shape/seed/采样协议下比较timer OFF的完整task性能；ON仅供诊断。默认值修改本地83项CPU通过，2项C++执行因无编译器跳过，1项既有Bash环境检查未选；未做设备验收。
