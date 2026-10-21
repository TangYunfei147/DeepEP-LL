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

```bash
git fetch origin experiment/ll-g4-overlap-ab
# 工作区有改动时先保存；不用reset --hard。
git switch --detach <该stage的提交号>
source scripts/ll_timer_env.sh
export DEEPEP_LL_SEND_CORES=28 DEEPEP_LL_RECV_CORES=28
export DEEPEP_NPU_ARCH=Ascend950 DEEPEP_LL_STAGE_TIMER=OFF
python -m pip install -v --force-reinstall --no-build-isolation --no-deps .
```

运行原测试命令，去掉 --ll-stage-timer 等打点参数；保持默认10预热/100采样/尾50，
正确性检查开启。每版使用不同RUN_NAME，例如ab1_fp8_off_001；同一轮各节点同名。
诊断时ON重建，添加 --ll-stage-timer --ll-timer-iters 5 --ll-timer-mode observe，
运行新版analyze.py。保留combine检查下游元数据与窗口复用。SDK无需仅因此重建。

设四版完整dispatch耗时为T1..T4：T2-T1衡量分区代价，T3-T2衡量取消排空/前移前缀，
T4-T3衡量协作调度净效应。阶段耗时用于解释，不代替T值。若T3最优，可以停在stage3；
若T2显著退化且3/4补不回来，先回stage1，再单独优化分区/lane预算。

各版本提交号和编译验证记录在完成后补充。
