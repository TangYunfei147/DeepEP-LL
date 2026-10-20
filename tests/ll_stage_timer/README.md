# NDR 逐次事件打点

分支：`kernel/ll-event-timer`，由 `29401f6` 创建。阶段格式为schema4，通信ABI保持5。设备只保存每次任务执行/等待片段的开始、结束、task ID及独立字节计数；所有累计时间、平均值和调用次数由离线分析计算。当前分支只接受该事件格式。

## 目标机更新与采集

在独立checkout使用本分支，不覆盖其他任务的工作区。各rank必须安装同一分支、同一版本的完整wheel。

```bash
git clone --branch kernel/ll-event-timer --single-branch \
  https://github.com/TangYunfei147/DeepEP-LL.git DeepEP-LL-events
cd DeepEP-LL-events
# 已有checkout则先检查git status，再git pull --ff-only。
source /实际的CANN路径/set_env.sh
source scripts/ll_timer_env.sh
git submodule update --init --recursive --depth 1
unset SHMEM_ROOT CATLASS_ROOT
python scripts/build_shmem.py
export DEEPEP_NPU_ARCH=Ascend950
export DEEPEP_LL_STAGE_TIMER=ON
python -m pip install -r requirements-build.txt
python -m pip install -v --force-reinstall --no-build-isolation --no-deps .
```

保留锁定SDK原生限制，不应用QP扩容。匹配SDK可校验复用，不为此打点格式自动重建SDK。环境路径由目标机提供，设备由用户明确分配；本指南不改变设备可见性。

在用户EP16单机环境中，沿用原测试参数，把打点模式设为mega。先使用1轮确认缓冲容量和事件数，再按需要增加轮数；run-name需使用新名称。

```bash
python tests/elastic/test_ep_low_latency.py \
  --num-processes 16 --num-nodes "$NNODES" --node-rank "$NODE_RANK" \
  --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" --shmem-port "$SHMEM_PORT" \
  --num-tokens 256 --num-max-tokens-per-rank 256 \
  --hidden 3584 --num-topk 8 --num-experts 384 --dtype fp8 --do-cpu-sync 0 \
  --ll-stage-timer --ll-timer-iters 1 --ll-timer-mode mega \
  --output-dir "$PWD/results/$CASE_NAME" --run-name events-001
```

原生LL仍只准入支持的EP；EP16的逻辑G4、每机16连续rank及跨机RDMA协议不变。正式性能遵循原10次预热、100次采集、各task最后50条；事件诊断在profiler关闭后的独立尾段执行。稳定与诊断每轮仍有Exp+scalar all_reduce，预分配并延迟复制/校验/导出。

## 同步策略

| 模式 | 边界行为 |
|---|---|
| mega（默认） | 普通阶段Start/End使用PIPE_ALL；依赖等待阶段不额外排空 |
| observe | 所有打点边界不新增PIPE_ALL |
| drain | 所有打点边界使用PIPE_ALL |

mega参考MegaMoE的普通span及依赖wait区分。无额外屏障的等待阶段为recv_wait、recv_lane_wait、recv_load_wait、recv_output_wait、send_input_wait、send_lane_wait、send_rdma_body_wait、send_rdma_flag_wait、combine_wait_ready、combine_data_wait、combine_wait_result。已有业务同步不删除。PIPE_ALL会串行化异步搬运，将等待成本转移到前面的阶段；不同模式不能直接当作同等扰动的性能对照。

所有模式均保存逐次时间戳。相同核的metadata和output_write按实际区间显示；任务执行与依赖等待互斥；等待切断执行，结束后用同一task ID续记。kernel固定总区间放在独立概览轨道。每次实测区间的位置保留，跨rank仍独立时间原点，不伪造时钟对齐。

首跳包尾部编码需要为额外成员计算q：先结束当前send_tail，再记录send_route，最后开始下一段send_tail，普通任务之间不嵌套。send_tail的calls因此表示尾部写入任务段数，不能当作包数。combine_return_task、combine_reduce_task和combine_output_task内部的依赖等待仍按同task ID暂停、续记。事件错误flags按位组合：1为容量溢出，2为重复/意外嵌套Start，4为不匹配End，16为退出时未闭合；flags=6属于配对问题，增加容量不能解决。

## 布局、内存与溢出

每核独占512B对齐区域：8-word header、每阶段实际事件计数、5类通信字节计数、错误标志，然后按事件类型分区存放[start,end,task_id]。无设备端累计耗时、首尾包络或旧格式适配。

固定初始化/计数/收尾阶段容量1，recv_prepare容量4；执行任务及依赖等待各4096槽。计数保存在每核私有数组，Finish统一写回实际片段数。容量由native/Python布局握手校验。重复超限仍保留真实计数并置错误；未闭合、重复Start、缺失End、时间戳异常及溢出都会拒绝导出，不输出截断trace。热点或大EP可能超过4096，需要先用较小诊断输入，不能把截断数据作为完整测量。

每核2,557,440B（约2.44MiB）；56核单次约136.6MiB；5轮dispatch+combine预分配约1.33GiB，不含输出、工作区及默认约13.53GiB Exp。增加轮数继续增加显存。导出只复制已写入事件到JSON，避免序列化全部空槽。

## 分析与结果

```bash
python tests/ll_stage_timer/analyze.py "results/$CASE_NAME/events-001" \
  --output-dir "results/$CASE_NAME/analyze-events"
```

单case输出bottleneck_summary.md、summary.json、stages.csv、events.csv、recv_core_samples.csv、bandwidth.csv及trace.json；多个case分别分析并生成REPORT_INDEX.md。所有shape从采样配置读取，不固定hidden或包长。默认使用sample记录的频率，目标已确认1GHz（ticks/1000 us）。输出目录不可已存在。

trace.json默认每AIV一个轨道，每条事件都有真实起止时间、task ID和片段index，不隐藏重复事件。export_trace.py支持筛选rank/operation/iteration及按stage分轨。

stages.csv的累计耗时是逐次(end-start)之和；每次成本为该阶段总时间/calls。recv报告覆盖慢rank全部接收核，详细CSV覆盖所有rank。同核任务/等待片段互斥，可以统计覆盖时间；未打点空隙不能解释成固定开销。calls为逻辑task数，segments为等待切分后的片段数。

字节计数和EP16有效带宽定义见[BANDWIDTH.md](BANDWIDTH.md)。独立专家归属统计仍可通过--ll-expert-timer采集，不用其累计数据绘制逐次事件。跨分支比较见[PARITY.md](PARITY.md)。

## 验证边界

本分支需要全rank重建和重新采集；现有其他格式的数据请使用对应原分支工具处理。正式性能基线使用timer OFF。事件记录、PIPE_ALL和大缓冲都会扰动执行，不把诊断时间当作未插桩性能。

CPU检查覆盖真实区间绘图、计数/容量/边界错误、紧凑导出、逐轮带宽及多shape离线报告。C++ host模型测试执行实际DeviceTimer，另有只编译检查；具体本轮验证结果见[VALIDATION.md](VALIDATION.md)。
## 任务组织

任务轨道没有sender/receiver、send_payload/send_pack、recv_consume等父层插桩；按路由处理、输入搬入、metadata、输出写回、逐包转发及独立等待记录。combine_return_task逐路由返回，combine_reduce_task逐partial归约回传，combine_output_task逐输出token；输入/输出依赖等待切开执行区间，同一任务各片段保留相同task ID。

例如一个归约任务先执行10tick、等待20tick、再执行30tick，保存执行[0,10,id]、等待[10,30,id]、执行[30,60,id]。该执行类型的统计为1个task、2个片段、40tick；等待类型为1个task、1个片段、20tick。固定kernel时长单独记录，不参与互斥任务的求和。

## 按算子筛选离线分析

```bash
python tests/ll_stage_timer/analyze.py <run目录> --op dispatch --trace-iteration 4
python tests/ll_stage_timer/analyze.py <run目录> --op combine --trace-iteration 4
```

`--operation`是`--op`的同义写法；不传则保留全部算子。筛选覆盖summary、CSV、trace及本分支附加报告，所选算子的全部rank/轮次仍必须完整。默认分别输出stage-analysis-dispatch、stage-analysis-combine；可用--output-dir指定新目录，已有输出不覆盖。仅离线脚本更新，无需重编wheel或重新采样。
