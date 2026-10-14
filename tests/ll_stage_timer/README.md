# LL dispatch/combine阶段打点

维护分支：`kernel/ll-configurable-groups`。本功能保持纯AIV、实际核数查询和1:1收发分工，不改变逻辑组配置及EP准入条件。

## 构建与接口

打点默认编译关闭，使用`DEEPEP_LL_STAGE_TIMER=ON`开启。Host/Device启动参数ABI更新，核心协议版本升为2；所有rank必须重建并安装同一版本完整wheel，不能混搭旧设备库。

在仓库根目录执行，外部CANN位置由当前环境提供：

```bash
# 若尚未加载CANN，显式设置CANN_ENV为当前机器set_env.sh路径。
source scripts/ll_timer_env.sh
git submodule update --init --recursive --depth 1
unset SHMEM_ROOT CATLASS_ROOT
python scripts/build_shmem.py --if-needed
export DEEPEP_NPU_ARCH=Ascend950
export DEEPEP_LL_STAGE_TIMER=ON
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

下例为每机16卡。所有节点配置相同参数和可达地址，`NODE_RANK`从0开始。单机EP16设置`NNODES=1`；两机EP32设置`NNODES=2`。默认逻辑组G=16独立于物理卡数，修改G需全量重建。

```bash
# 在目标机仓库根目录执行。先激活匹配wheel的Python环境。
source scripts/ll_timer_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export ASCEND_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES"
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
  --ll-stage-timer --ll-timer-iters 5 --ll-timer-mode observe \
  --skip-perf-test --output-dir "$PWD/results/ll-stage-001"
```

这是独立诊断调用，每次先执行完成边界，再逐次校验输出后保存timer，不重试失败实验；这里不执行大Exp扰动，warmup/num-iters/perf-tail只控制另行运行的profiler采样。`--plan`仅打印CPU计划；A5不可执行上面的16卡命令。

去掉`--skip-perf-test`会另外进行原有profiler采样：10预热、100次采集，每次Exp+scalar all_reduce，统计最后50条。timer诊断和profiler样本不混合。旧delay-exp-before/after参数仅接受1/0；不要沿用首调用后1000次Exp的旧配置。`--dispatch-only`只诊断/测量dispatch；正常默认同时覆盖dispatch和combine。

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
| `send_issue` | 对端地址计算及MTE下发；真正完成依赖原lane收尾 | 同上 |
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

**默认单位是GetSystemCycle ticks，不猜测频率。** 仓内MegaMoE解析器采用`cycles/1000`的换算，相当于1GHz；确认目标机同一计数器采用该频率后，可显式加`--ll-timer-frequency-hz 1000000000`，或在离线解析时加`--frequency-hz`。不能用AIV计算频率替代系统计数器频率。

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
