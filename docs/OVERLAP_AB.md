# 四阶段 dispatch A/B 对照

本分支从 kernel/ll-g4-overlap-recv@bf1c3a3 派生，保留七模式拆分编译、
AICore 回调修复、schema 入口校验和编译防回退要求。原分支不修改。

## 为什么不能仅凭 SEND_TOKENS 判断收益

旧改动改变了 SEND_TOKENS/CAL_CUMSUM 的范围；取消发送尾部等待还可能将同一
通信成本推迟到前缀等待、接收 lane 复用或接收最终排空。整体 kernel 时间由
关键路径决定，发送核局部变快不保证整体变快。独立收发 UB 可能减少 lane 深度，
每八任务 flush 输入预取及轮询也可能抵消重叠收益。没有实测数据不能判定某一步优劣。

## 四个版本

所有版本仅通过 common/ll_overlap_stage.h 的提交常量选择，无运行时启发式。
从 stage1 前进到4，或反向 checkout，对应唯一功能增量：

| Stage | 功能 | 与前一版对比 |
|---|---|---|
| 1 | 串行收发，原 UB 复用、保留排空 | 公共打点/编译框架下的基线 |
| 2 | 独立 UB、事件规划、共享前缀；保留原顺序和排空 | 分区/准备成本与 lane 深度影响 |
| 3 | 取消中途排空；R 在本组发送之前准备前缀 | 延后退休与前缀前移的组合收益 |
| 4 | R 每8个任务穿插 count/接收尝试 | 批次预取中断、轮询与早接收的净影响 |

Stage3有两个绑定变化，若3→2差异显著再拆分二者，不把结果归因于其中单项。
Stage1是现代框架下的串行消融基线，不是历史781b1aa的逐字恢复。
Stage2..4沿用同一 UB 规划器和1..6 lane选择；不足时显式SERIAL_FALLBACK。
OVERLAP_PATH仅表示启用独立分区，Stage2仍保留排空，不能仅凭此计数认定已重叠。
Stage1也记SERIAL_FALLBACK，表示此核走串行路径；结合overlap_stage区分主动基线与UB不足。

Host/device校验stage一致；ST检查安装wheel与当前checkout相符，并在创建SHMEM前
核对各rank一致。结果comparison记录overlap_stage。全rank必须重装同一版本wheel。

## 统一打点（schema12，全部四版一致）

- SEND_TOKENS：仅发送循环/输入预取推进，不含准备、元数据收尾、排空及穿插接收。
- CAL_CUMSUM：仅CalcCumsum，BUILD_RECV_CUMSUM是其中的嵌套阶段；不含RecvCnt/CompleteCumsum。
- WAIT_RECV_COUNTS：各次count检查累计，不包含两次检查之间的发送。
- PREFIX_COMPLETE：完整前缀准备，包括其内部WAIT_CUMSUM、PREFIX_LOAD及必要阶段排空。
- PREFIX_LOAD：完整前缀搬入及原MTE2_S；RECV_COMPLETE_WAIT：接收末尾原PIPE_ALL。
- LOCAL_WINDOW_COPY仍是接收尝试累计。SEND_SUBMIT_END与KERNEL均预留trace记录；
  其余阶段仍最多采样前两次。只统计、不增加同步。

12与11的阶段语义不同，不对两者做诊断耗时比。比较四版时以同rank/轮次/核的
记录为单位；父子不能相加，不相减不同核的最大值或p50。总体以timer OFF profiler为准。

## 测试顺序与回退

先固定shape、dtype、S/R、seed、route、correctness-repeats与采样参数。
建议先测1和4；确有变化后补2和3定位，或四版全部测齐。每版先正确性再OFF性能，
需要归因时重建ON采集observe；不要使用relay-complete作性能对照。

下面以28/28为例；若当前实验使用其他配比，四版均沿用相同实际值。

```bash
git fetch origin experiment/ll-g4-overlap-ab
# 工作区有改动时先保存；不用reset --hard。
git switch --detach <该stage的提交号>
source scripts/ll_timer_env.sh
export DEEPEP_LL_SEND_CORES=28 DEEPEP_LL_RECV_CORES=28
export DEEPEP_NPU_ARCH=Ascend950 DEEPEP_LL_STAGE_TIMER=OFF
export DEEPEP_LL_EXPECT_IMPLEMENTATION=g4-overlap-ab
python -m pip install -v --force-reinstall --no-build-isolation --no-deps .
```

运行原测试命令，去掉 --ll-stage-timer 等打点参数；保持默认10预热/100采样/尾50，
正确性检查开启。每版使用不同RUN_NAME，例如ab1_fp8_off_001；同一轮各节点同名。
诊断时ON重建，添加 --ll-stage-timer --ll-timer-iters 5 --ll-timer-mode observe，
运行新版analyze.py。保留combine检查下游元数据与窗口复用。SDK无需仅因此重建。

设四版完整dispatch耗时为T1..T4：T2-T1衡量分区代价，T3-T2衡量取消排空/前移前缀，
T4-T3衡量协作调度净效应。阶段耗时用于解释，不代替T值。若T3最优，可以停在stage3；
若T2显著退化且3/4补不回来，先回stage1，再单独优化分区/lane预算。

## 精确提交

| Stage | Commit |
|---|---|
| 1 | `93e33f0` |
| 2 | `21f5e0e` |
| 3 | `ac876ba` |
| 4 | `fbd1509` |

2、3、4各提交仅改变stage常量；公共打点/构建/身份框架均在1中。
Stage4仍保留当前CalcCumsum/CompleteCumsum的阻塞式前缀等待，本次不混入
“所有前缀等待都可让出发送”的额外优化。

建议记录一张表：stage、commit、OFF dispatch尾50均值、最慢rank、正确性、
ON诊断kernel跨度、发送/接收lane数、SEND_COMPLETE_WAIT、WAIT_CUMSUM、
PREFIX_LOAD、RECV_BUFFER_WAIT、RECV_COMPLETE_WAIT和FINAL_DRAIN。ON与OFF结果分列，不能混算。
对于接近噪声的差异，按1→2→3→4→3→2→1顺序复测，保持同一设备/路由/shape，
先判断重复运行的波动，再判断优劣。不要根据单个阶段的缩短直接选择版本。

## 编译验证记录（2026-10-21）

A3_docker / CANN9.3.0 / O2 / Ascend950 / S28R28 / jobs4。
四个精确提交各自完成311项CPU/C++检查、完整ON wheel及dispatch全部模式ON/OFF编译。
未安装wheel或运行NPU；这些结果不能替代目标机正确性与性能测试。

| Stage | ON冷编译秒 | OFF冷编译秒 | 完整ON wheel |
|---|---:|---:|---|
| 1 | 48.756 | 40.618 | 通过 |
| 2 | 49.338 | 41.113 | 通过 |
| 3 | 51.572 | 42.139 | 通过 |
| 4 | 79.302 | 53.695 | 通过 |

上表是dispatch编译墙钟时间，不是算子运行耗时T1..T4。所有产物使用独立目录。
远端原工程 /home/tyf/Q4_remote_test/DeepEP-LL-g4-overlap-recv 下：
`.logs/overlap-ab-001/stageN-cpu.log`、`stageN-wheel.log`、`stageN-on/timing.json`、
`stageN-off/timing.json`；wheel在`.artifacts/overlap-ab-001/stageN/`。
验证结束已恢复原远端kernel/ll-g4-overlap-recv分支。
