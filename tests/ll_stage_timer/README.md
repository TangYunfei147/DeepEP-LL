# LL dispatch/combine阶段打点

维护分支：`kernel/ll-ndr-hns1825`。本功能保持纯AIV、实际核数查询和1:1收发分工，不改变逻辑组配置及EP准入条件。

## 构建与接口

打点默认编译关闭，使用`DEEPEP_LL_STAGE_TIMER=ON`开启。打点最初引入核心协议版本2；当前混合通信/tiling key语义已将协议升为4。所有rank必须重建并安装同一版本完整wheel，不能混搭旧设备库。RDMA协议见[审计说明](../ll_rdma/AUDIT.md)。

在仓库根目录执行，外部CANN位置由当前环境提供：

```bash
# 若尚未加载CANN，显式设置CANN_ENV为当前机器set_env.sh路径。
source tests/ll_rdma/env.sh
git submodule update --init --recursive --depth 1
unset SHMEM_ROOT CATLASS_ROOT
python scripts/build_shmem.py --if-needed --rdma-backend HNS_1825
export DEEPEP_LL_RDMA=ON
export DEEPEP_NPU_ARCH=Ascend950
export DEEPEP_LL_STAGE_TIMER=ON
export DEEPEP_LL_BACKEND=rdma
export LD_LIBRARY_PATH="$PWD/third-party/shmem/install/shmem/lib:${LD_LIBRARY_PATH:-}"
python -m pip install -r requirements-build.txt
python -m pip install -v --force-reinstall --no-build-isolation --no-deps .
```

使用工程内Python虚拟环境；脚本将缓存、临时文件和日志定向工程内，不设置设备可见性。此分支在保留9.2版本声明的同时接受用户确认的CANN 9.3.0，SDK身份中继续记录实际版本。

已有业务代码可以对单次LL调用使用以下独立诊断上下文，公开dispatch/combine参数不变：

```python
from deep_ep.utils.ll_stage_timer import capture

with capture(buffer, "dispatch", mode="observe") as trace:
    result = buffer.dispatch(...)  # 正常LL参数，实际EP必须合法
trace.save(output_dir / "dispatch-stage.json", metadata={"rank": rank})
```

`output_dir`由业务入口按自身脚本位置解析成工程内绝对路径，避免依赖当前工作目录。

每个上下文只允许同一buffer的一次指定操作，不允许嵌套/复用；timer张量会随LL内部stream保活，离开上下文后复制到CPU并检查完整性。业务示例的自定义metadata不直接用于下述全rank汇总；完整测试入口会写入所需身份及正确性证据。

## 目标机运行

下例为每机16卡。所有节点配置相同参数和可达地址，`NODE_RANK`从0开始。单机EP16设置`NNODES=1`；两机EP32设置`NNODES=2`。本分支按通信域自动选择tiling key：EP16为G=4，EP32及以上为G=16；物理每机16卡独立于逻辑组。

```bash
# 在目标机仓库根目录执行。先激活匹配wheel的Python环境。
source tests/ll_rdma/env.sh
export DEEPEP_LL_BACKEND=rdma
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export ASCEND_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES"
export CUDA_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES"
export PROBE_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES"
NNODES=1
NODE_RANK=0
MASTER_ADDR=127.0.0.1       # 多机时改成首节点可达IP
MASTER_PORT=29500
SHMEM_PORT=29501
unset RANK WORLD_SIZE LOCAL_RANK

python tests/elastic/test_ep_low_latency.py \
  --num-processes 16 --num-nodes "$NNODES" --node-rank "$NODE_RANK" \
  --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" --shmem-port "$SHMEM_PORT" \
  --num-tokens 32 --num-max-tokens-per-rank 256 \
  --hidden 7168 --num-topk 6 --num-experts 256 --dtype bf16 --do-cpu-sync 0 \
  --use-1g-huge-pages 0 --ll-stage-timer --ll-timer-iters 5 --ll-timer-mode observe \
  --skip-perf-test --output-dir "$PWD/results/ll-stage-001"
```

这是稳定运行后的独立诊断尾段。跳过perf时，先执行warmup次预热、num-iters次未打点稳定调用，再执行ll-timer-iters次打点；稳定和打点调用前均执行Exp+scalar all_reduce。所有打点轮次结束后才复制、校验和保存，不重试失败实验。`--plan`仅打印CPU计划；A5不可执行上面的16卡命令。

去掉`--skip-perf-test`后，每个操作先完成原有profiler采样（10预热、100次采集、每次Exp+scalar all_reduce、统计最后50条），关闭profiler后复用同一Exp缓冲追加打点轮次，再解析性能数据。profiler结束可能包含导出开销，不保证诊断与最后一条性能样本无间隙。timer诊断和profiler样本不混合。旧delay-exp-before/after参数仅接受1/0；不要沿用首调用后1000次Exp的旧配置。`--dispatch-only`只诊断/测量dispatch；正常默认同时覆盖dispatch和combine。

M决定去重：`M>32`启用，`M<=32`关闭。若对比两条路径，在满足`B<=M`的前提下分别测量并保留独立输出目录。

## 两种边界口径

- `observe`：不增加流水屏障，记录现有函数/代码段边界。异步MTE的下发和完成可能跨越阶段；`send_issue`表示下发区间，不是网络传输完成时间。
- `drain`：每次Start/End显式加入`PipeBarrier<PIPE_ALL>()`，便于观察阶段执行与排空成本，但会破坏重叠，改变整体性能。必须单独运行，不能与observe或未打点基线混算。

计时本身也有读取计数器、GM写回、分支及寄存器资源成本。ON构建且不传timer时不记录数据，但二进制资源占用仍可能与OFF不同；严格性能基线使用`DEEPEP_LL_STAGE_TIMER=OFF`重新构建的wheel。

## 阶段和源码

| 阶段 | 含义 | 主要源码 |
|---|---|---|
| `kernel` | 本核主体区间；不含timer头初始化和最终完成标志写回，也不等于profiler完整task区间 | `csrc/kernels/{dispatch,combine}/low_latency.cpp` |
| `magic`、`worker_init`、`count_init` | 窗口轮转、worker初始化与计数初始化；dispatch存在嵌套 | `dispatch/low_latency/dispatch_server_dedup_kernel.h`、combine入口 |
| `count_send` | 路由计数及向专家所属rank发布，observe模式不额外等待DMA | `dispatch/low_latency/device/dispatch_server_dedup_counts.h` |
| `count_receive`、`count_prefix`、`count_publish` | 计数ready等待、局部前缀、前缀发布及原有排空 | 同上及dispatch总调度 |
| `sender` | 发送核整个Process父区间 | `dispatch/low_latency/device/dispatch_server_dedup_send.h` |
| `send_prepare`、`send_payload`、`send_metadata` | UB/路由准备，发送主循环，lane收尾及源掩码/权重写回 | 同上 |
| `send_classify` | 去重路由分类累计，不含外部q计算 | 同上`Classify`调用点 |
| `send_pack` | 输入搬入、可选量化、包尾/标志打包及lane等待；不解释为纯计算 | 同上`SendPacket` |
| `send_issue` | 机内为MTE下发；跨机包含暂存和RDMA put/quiet，不能解读为纯网络时间 | 同上 |
| `receiver`、`recv_prepare` | 接收父区间；前缀/任务表和接收UB准备 | `dispatch/low_latency/device/dispatch_server_dedup_recv.h` |
| `recv_wait`、`recv_consume`、`recv_cleanup` | 探测直到出现ready批次、加载/消费及relay、最后lane/标志清理；重复区间累计 | 同上 |
| `count_finish` | 原计数收尾及控制状态恢复 | dispatch总调度 |
| `combine_prepare`、`combine_notify`、`combine_wait_ready` | 本地准备、通知（含等待本卡准备完成）、组内ready等待 | `combine/low_latency/combine_server_dedup_kernel.h` |
| `combine_direct`、`combine_partial` | DIRECT回传及原排空、partial归约回传 | 同上及`combine_server_dedup_partial.h` |
| `combine_aggregate`、`combine_drain`、`combine_final_drain` | M5/M6源token归约、共享输出收尾、最终流水排空 | combine总调度及`combine_server_dedup_source_aggregate.h` |

上表kernel路径相对于`csrc/kernels/`（第一行除外）。公共格式/设备实现位于`dispatch/low_latency/common/ll_stage_timer_{layout,device}.h`；Host校验在`csrc/ops/low_latency.cpp`；Python上下文及解析器在`deep_ep/utils/ll_stage_timer.py`。

## 输出与统计

每个case分别输出：

```text
<run>/case_00000/normal/ll_stage_timer/
  dispatch/rank_000/sample_0000.json
  combine/rank_000/sample_0000.json
  ...
```

每文件包含原始`[实际AIV数,32,8]` int64数组和每核/阶段解析值。第0槽记录magic/schema/操作/核数/核号/模式/完成标志；阶段槽记录first/last/累计ticks/次数及活动状态。分配器创建512B对齐视图，Host拒绝未对齐的直接传入指针；每核独占2048B，避免跨核cache-line写回冲突，当前支持16..72偶数核。每轮新建零缓冲，不覆盖前一轮；未执行阶段保持零值，不冒充零成本样本。

**目标计数器换算已由用户确认：`time_us = ticks / 1000`。** 测试及命令行分析默认1GHz，原始ticks仍保留；可用frequency参数显式覆盖其他已确认环境。

收齐所有rank的同一case目录后，在仓库根目录执行：

```bash
python tests/ll_stage_timer/analyze.py \
  results/ll-stage-001/<run>/case_00000/normal/ll_stage_timer \
  --output-dir results/ll-stage-summary-001
```

输出`summary.json`与`stages.csv`，按rank/core/stage给出mean/min/max/P50/P90/P99；JSON另有各rank逐次取core最大阶段累计值的统计。后者不是跨核全局跨度。重复阶段用sum而不是first到last的跨度。嵌套父子阶段、不同核并行阶段不能直接相加；不跨rank相减时间戳。缺rank、缺轮次、损坏/未闭合记录及混配置会报错。

## 验证范围

- CPU：格式/ABI映射、损坏数据、重复与缺rank/轮次、上下文隔离及基线采样时序。
- `bash tests/ll_stage_timer/validate_device.sh`：专用A5入口，强制卡0/1，仅在卡0启动独立timer验证核；验证开/关、两种模式、每核记录及guard，不建立伪造LL域。
- `tests/ll_stage_timer/check_installed.py`：A5卡0验证完整wheel的ABI、查询核数和timer分配，不执行LL通信。
- 完整dispatch/combine的精度、阶段曲线、跨机同步及实际性能必须由用户在合法目标EP上验证。

## Chrome tracing / Perfetto 时间线（NDR，2026-10-16）

参考 Mega_Moe 的 Chrome Trace Event JSON 导出格式，支持将内部阶段导入
`chrome://tracing`（Load）或 Perfetto（Open trace file）查看。无需修改设备kernel、SDK或ABI。

### 新采集自动输出

构建仍使用 `DEEPEP_LL_STAGE_TIMER=ON`，测试保留
`--ll-stage-timer --ll-timer-mode drain`，默认使用用户已确认的1GHz计数器（ticks/1000 us）。
每份 `sample_0000.json` 旁自动生成 `trace_sample_0000.json`，可直接加载。
原始ticks始终保留。Python导出代码位于wheel中，更新后重新安装wheel以启用自动导出；
离线转换不需重编。

### 已有数据离线转换（不需重新采集）

在仓库根目录执行，替换实际run路径；所有相对输入输出均按脚本定位的工程根解析：

```bash
python tests/ll_stage_timer/export_trace.py \
  results/ndr-drain-ep32/<run>/case_00000/normal/ll_stage_timer \
  --frequency-hz 1000000000 \
  --rank 0 --operation dispatch --iteration 0 \
  --output results/trace-dispatch-rank0-sample0.json
```

上例频率可省略，默认1GHz。也可传单个raw sample文件。
去掉rank/operation/iteration筛选可导出多份采集，允许单机的部分rank数据；此转换器
用于浏览，不替代analyze.py对完整rank/轮次的验收。输出文件已存在会报错，避免覆盖。

### 时间线口径

- 每个rank/operation/sample为独立进程轨道，分别以本次最早内部时间戳归零。
  核间保留本设备计数器相对时间；不同rank/不同调用不对齐，不与外部profiler合并。
- 默认每个AIV core只有一条具名轨道，所有阶段使用同一tid，父子区间显示嵌套。重叠或重复包络可能仍有嵌套层级；不改变真实起止点。加`--lane-layout stage`可恢复分阶段轨道排查重叠。
- 单次阶段显示首次到末次真实区间；重复阶段标为`[envelope xN]`，宽度是首尾包络，
  **包括中间空隙，不是累计执行时间**。点击事件查看`total_us/total_ticks`累计耗时、
  `average_us`、`calls`、`gap_ticks`及原始起止ticks。
- schema 1没有保存每次重复调用的起止点，无法从现有数据还原Mega_Moe式逐次动态事件；
  不伪造连续片段。父子阶段/并行阶段不可直接相加，drain数据也不是未插桩性能基线。

## 推荐：一个命令完成统计和tracing导出

运行用例时可以提前指定目录名（两台节点参数一致）：

```bash
# 加到原有测试命令末尾；新run名不得与已有目录重名。
--output-dir results/ll-stage-001 --run-name drain-001
```

收齐同一run全部rank后执行：

```bash
python tests/ll_stage_timer/analyze.py results/ll-stage-001/drain-001
```

不指定run-name、仍使用自动随机目录时，无需查目录名：

```bash
python tests/ll_stage_timer/analyze.py results/ll-stage-001 --latest
```

`--latest`根据子目录的cases.json修改时间选最近创建的run，打印实际选择路径；
复制多机数据时保留该manifest时间（或使用固定run-name路径）。不会因最新run缺rank而退回旧run。
多机仍须先合并所有rank的原始数据；analyze会拒绝不完整或混配置样本。

默认在选中目录的stage-analysis下生成summary.json、stages.csv、trace.json。
多个case/路由各自输出到stage-analysis下对应相对目录，不混算配置。
统计覆盖全部轮次；trace.json默认展示第0轮全部rank的dispatch/combine，
可加`--trace-iteration 3`查看第3轮。各rank/操作独立归零，不代表真实跨rank同步。
默认换算ticks/1000 us，原始ticks仍在样本和trace事件参数中保留。

已有报告不覆盖；再次分析使用`--output-dir results/ll-stage-summary-002`。
无需另跑export_trace.py；该入口仅保留给单rank/部分数据浏览及细粒度筛选。

## 稳定尾段采样参数（2026-10-16）

以下两种模式采用相同前处理；参数加到原目标机测试命令中：

```bash
# 无外部profiler：10次预热 + 100次稳定调用 + 5次内部打点
--skip-perf-test --ll-stage-timer --warmup 10 --num-iters 100 \
  --ll-timer-iters 5 --ll-timer-mode observe

# 外部profiler：10次预热 + 100次性能采集 + 5次内部打点
--ll-stage-timer --warmup 10 --num-iters 100 --perf-tail 50 \
  --ll-timer-iters 5 --ll-timer-mode observe
```

需要drain时将observe改成drain。内部打点仍编号0..4，最后一轮分析使用
`analyze.py ... --trace-iteration 4`。perf-tail只控制性能统计，不改变打点数量。

timer缓冲在采样前分配，采集循环只执行前处理、LL调用及设备流依赖；
不逐轮复制到CPU、执行正确性collective或写文件。所有轮次的输出和handle保持存活，
结束后统一完成边界，再逐轮校验、导出。失败不自动重试；analyze会校验新metadata声明的
完整诊断轮数，拒绝中断后只留下部分轮次的结果。

显存注意：非perf打点现在同样使用默认118260×30720 FP32 Exp缓冲（约13.53GiB/卡），
并同时保留N轮输出/handle及timer（每轮每实际AIV 2048B加对齐空间）。N增大会增加峰值显存；
建议先保持默认5轮。A5仅双卡原语或CPU验证，不运行原生LL；小Exp不能代表目标参考负载。

新流程修改了Python capture接口，所有节点须安装匹配版本wheel：更新Git后保持
DEEPEP_LL_RDMA=ON、DEEPEP_LL_STAGE_TIMER=ON，执行既有pip重装命令。
SDK、C++ kernel和ABI4未变，不需仅为本次更新重建SHMEM。
默认单AIV轨道适用于自动trace、analyze.py及export_trace.py。更新Git后对已有raw sample重新分析即可，无需重新采集；核间起始偏移保持原样，不做逐核归零。
