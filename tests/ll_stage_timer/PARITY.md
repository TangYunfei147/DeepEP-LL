# 两个 kernel 分支的共同实验与对比

本指南同时适用于kernel/ll-configurable-groups与kernel/ll-ndr-hns1825。
统一启动入口tests/ll_stage_timer/run.py，统一ST、采样、timer/专家格式和分析脚本。
两边核心启动ABI5、阶段schema2、专家schema1；implementation字段会拒绝混用不同分支wheel。
同一ABI号不意味着可以在一个作业内混用两个分支。

## 必须匹配的条件

- 相同目标设备、每机16个连续rank、相同EP、实际AIV数、B/M/H/topK/专家数/dtype/路由/seed。
- 相同编译timer开关、observe/drain、专家打点开关、预热/轮数/Exp尺寸及page policy。
- G是用户选择的比较变量：泛化分支保留默认16，NDR EP16为4、EP32为16；允许直接比较不同G，不要求调整一致。
- perf仍为10预热、100采集，每次Exp+scalar all_reduce，最后50条独立统计；timer是之后追加的诊断尾段。
- 正式性能对照用两边timer OFF；阶段对照用两边timer ON。不要交叉比较。

泛化分支保留MTE与原静态分组，不包含RDMA实现。多机MTE是否可用取决于目标环境的真实peer映射；
若EP32的MTE无法运行，它是不可用基线，不能填0或声称NDR加速。

## 各分支构建

建议各自使用独立clone、工程内虚拟环境和工程内SDK。每个分支在所有节点的绝对目录相同，
不同分支之间目录可以不同。先加载当前机器CANN或显式设置CANN_ENV，然后在相应仓库根目录执行。

```bash
source scripts/ll_timer_env.sh
git submodule update --init --recursive --depth 1
unset SHMEM_ROOT CATLASS_ROOT
export DEEPEP_NPU_ARCH=Ascend950
export DEEPEP_LL_STAGE_TIMER=ON    # 正式未插桩性能改为OFF，两边一致
```

泛化分支（保留原默认G=16，需要其他G时按原分支方法自行配置）：

```bash
unset DEEPEP_LL_RDMA
python scripts/build_shmem.py --if-needed
```

NDR分支：

```bash
export DEEPEP_LL_RDMA=ON
python scripts/build_shmem.py --if-needed --rdma-backend HNS_1825
```

两边随后均执行：

```bash
python -m pip install -r requirements-build.txt
python -m pip install -v --force-reinstall --no-build-isolation --no-deps .
```

泛化分支仍通过原DEEPEP_LL_RANKS_PER_GROUP宏配置G，本同步不改其默认值；改变G后自行重建Host和device。
SDK身份变化须停止SDK用户后显式--rebuild；不得跨分支混用不匹配SDK。

## 同一条目标机启动命令

在对应仓库/虚拟环境中执行。仅BACKEND和结果名称按分支改变。
单机NNODES=1、NODE_RANK=0、MASTER_ADDR=127.0.0.1；双机NNODES=2，两节点分别0/1并同时执行。

```bash
source scripts/ll_timer_env.sh
BACKEND=mte               # 泛化分支mte；NDR分支rdma，也可显式mte做NDR自身对照
NNODES=1
NODE_RANK=0
MASTER_ADDR=127.0.0.1      # 多机改为首节点可达地址
python tests/ll_stage_timer/run.py \
  --backend "$BACKEND" --devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --num-processes 16 --num-nodes "$NNODES" --node-rank "$NODE_RANK" \
  --master-addr "$MASTER_ADDR" --master-port 29500 --shmem-port 29501 \
  --use-1g-huge-pages 0 \
  --num-tokens 32 --num-max-tokens-per-rank 256 --hidden 7168 \
  --num-topk 6 --num-experts 256 --dtype bf16 --do-cpu-sync 0 \
  --warmup 10 --num-iters 100 --perf-tail 50 \
  --ll-stage-timer --ll-expert-timer --ll-timer-mode observe --ll-timer-iters 5 \
  --output-dir results/compare --run-name run-001
```

只做阶段诊断时，两边都增加--skip-perf-test；也会执行10次预热和100次稳定调用后再采集5轮。
正式性能则两边重建timer OFF，删除所有ll-stage-timer/ll-expert-timer参数，保留性能轮数设置。
统一入口默认page policy为0，记录实际G但不要求跨分支一致；不会把泛化分支的rdma请求静默替换为mte。
A5或纯CPU检查仅用`--validation --plan --num-processes 16`，四个可见性变量固定0,1，绝不启动EP16设备。

## 统一输出和对比

原run目录的summary.csv/ranks.csv增加implementation、transport、logical_group、active_aiv、core_protocol及source_revision。
每case及阶段summary.json记录完整comparison身份。分别收齐所有rank后运行：

```bash
python tests/ll_stage_timer/analyze.py results/compare/run-001 --trace-iteration 4
```

两边都输出stages.csv、trace.json、experts.csv、expert_cores.csv（专家文件仅开启该项时产生）。
六类重复发送包络默认隐藏；--show-repeated可恢复，累计统计不删。
RDMA专属阶段在MTE结果中缺失是正常现象，不补成0。

将两边结果保存在同一分析工程下的不同目录，例如results/mte与results/ndr：

```bash
# 完整perf的run summary
python tests/ll_stage_timer/compare.py results/mte/summary.json results/ndr/summary.json \
  --kind perf --output results/perf-comparison.csv
# analyze产生的stage summary；专家统计则把kind改成expert
python tests/ll_stage_timer/compare.py results/mte/stage-analysis/summary.json results/ndr/stage-analysis/summary.json \
  --kind stage --output results/stage-comparison.csv
```

只有配置匹配的同一key才计算candidate_over_baseline（NDR/MTE，越小表示该指标越低）。
实际G分别写入baseline_group/candidate_group，不做一致性检查。其他核数、形状、seed、采样协议不一致会报错；缺失阶段保留空值，无数据或缺身份的旧报告拒绝对比。
多case分析输出位于stage-analysis下各case/路由相对目录，逐case选择对应summary.json。
专家跨核累计是工作量，不是端到端时延；跨rank时间戳不对齐，不能相减。

可使用`python tests/ll_stage_timer/verify_shared.py ../另一个checkout`核验共享文件一致性。
当前只在本地做CPU/路径/配置回归，未进行本次CANN编译或目标机设备验收。
