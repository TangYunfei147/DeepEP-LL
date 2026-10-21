# Relay下发与完成时延诊断

分支kernel/ll-g4-relay-priority，算法基线35ae147。
新增阶段只包围原relay循环，不改地址、包数、payload、lane归属或输出顺序。

| ID | 名称 | 模式 | 含义 |
|---|---|---|---|
| 66 | RELAY_ISSUE | observe及relay-complete | 一组FIRST_HOP的relay循环内地址计算和下发，可能含发射背压 |
| 67 | RELAY_PRE_DRAIN | 仅relay-complete | 排空此前本核MTE3工作的等待；不计入relay完成区间 |
| 68 | RELAY_COMPLETE | 仅relay-complete | 前置排空后，从下发开始到本组MTE3写完成 |
| 69 | RELAY_PACKETS | 两种模式 | 实际relay写调用数，只计数，没有耗时 |

组数是RELAY_ISSUE的calls，一组可能有多条relay写。RELAY_COMPLETE包含
RELAY_ISSUE及诊断边界开销，两者不能相加，也不将相减结果称为纯链路时延。
完成指生产者MTE3完成，不是对端消费确认；其他核/卡仍可产生竞争流量。
relay循环的目标可能包含本rank；本次不拆分本地/远端目标，不把统计全部称为跨卡流量。

## 同步与开关

默认observe不执行新同步，只增加私有计时和计数，仍在kernel结束后统一导出。
显式relay-complete仅允许dispatch诊断，且要求：

```
--ll-stage-timer --dispatch-only --skip-perf-test --ll-timer-mode relay-complete
```

该模式在每个FIRST_HOP relay循环之前、之后分别执行一次本核MTE3_S：
前置排空消除之前其他lane写请求的尾部，结束等待确保本组写退休。
沿用现有S_MTE3/MTE2_MTE3准备依赖；没有全核或跨卡屏障，也不在每个
转发目标之间插同步。使用即时FetchEventID配对Set/Wait，两个等待不重叠持有事件。
只有诊断tensor存在且模式为relay-complete才启用；OFF编译不执行这些等待。

完成模式会改变正常重叠与下发背压，不能当正式性能基线。输入等待瓶颈自动
判断在该模式下不适用，标为数据不足，不从受扰动的KERNEL占比推断正常瓶颈。
Host、Python与设备均拒绝combine的完成模式。CLI也拒绝未开启打点、
未限制dispatch或同时请求正式profiler的完成模式。

## 格式

新schema10，保留原0..65 ID；metric容量70，trace容量仍192，
每核缓冲仍72×8 int64（4608B）。原始header编码observe=1、relay-complete=3。
完整解析器支持旧schema7/9；旧数据的relay时间显示unavailable，不补零。
同一报告内混模式/混schema拒绝，跨诊断模式或schema不输出加速比。
核心通信协议10000+S不变，需全rank重装相同提交的完整ON wheel并重采，
无需因此强制重建已校验SHMEM。新格式继续用JSON/CSV/Chrome trace。

## 目标机采集

沿用README的设备、拓扑和CANN环境设置；以下仅在目标机执行。
先构建安装timer ON wheel，然后分别运行正常下发诊断和完成诊断，
使用不同run-name，不覆盖现有结果：

```bash
export DEEPEP_LL_STAGE_TIMER=ON
python -m pip install -v --force-reinstall --no-build-isolation --no-deps .

# DEVICES/NNODES/NODE_RANK/MASTER_ADDR使用现有目标作业的实际配置。
# 以下shape只是示例，可替换成当前性能case；所有节点参数保持一致。
BASE_ARGS=(
  --devices "$DEVICES" --num-nodes "$NNODES" --node-rank "$NODE_RANK"
  --master-addr "$MASTER_ADDR" --master-port 29500 --shmem-port 29501
  --num-tokens 256 --num-max-tokens-per-rank 256
  --hidden 3584 --num-topk 8 --num-experts 384 --dtype fp8 --do-cpu-sync 0
  --warmup 10 --num-iters 100
)
python tests/ll_stage_timer/run.py "${BASE_ARGS[@]}" \
  --dispatch-only --skip-perf-test --ll-stage-timer --ll-timer-iters 5 \
  --ll-timer-mode observe --output-dir results/relay-timing --run-name relay-issue-001

python tests/ll_stage_timer/run.py "${BASE_ARGS[@]}" \
  --dispatch-only --skip-perf-test --ll-stage-timer --ll-timer-iters 5 \
  --ll-timer-mode relay-complete --output-dir results/relay-timing --run-name relay-complete-001

python tests/ll_stage_timer/analyze.py results/relay-timing/relay-issue-001 --op dispatch
python tests/ll_stage_timer/analyze.py results/relay-timing/relay-complete-001 --op dispatch
```

可先对采集命令追加`--plan`做CPU规划。BASE_ARGS必须含现有实际设备列表；
多机同run的全部rank文件按原目录合并后解析，不拼接两个模式的数据。
稳定调用/Exp/正确性校验协议继续沿用原入口；--skip-perf-test不跳过稳定阶段。

每个报告目录新增：

- relay_timing.md：模式、组数、relay写次数、下发/完成/前置排空的每组加权均值。
- relay_samples.csv：逐rank/轮/核的原始ticks及计数。
- relay_timing.csv：逐核汇总；保留ticks，已知频率时另输出us/组。
- summary.json的relay_timing：上述结构化结果及解释。

累计时间与比例（2026-10-21离线更新）：

- 每个rank/core使用同一组采样轮次同时累加RELAY_COMPLETE和KERNEL，
  包括新格式中有测量但没有relay的轮次，不用累计完成时间除以单轮均值。
- `complete_ticks_total`/`complete_us_total`为累计转发完成时间；
  `kernel_ticks_total`/`kernel_us_total`为对应累计KERNEL时间。
- `complete_kernel_ratio`为前者除以后者，Markdown显示百分比；
  `relay_writes_per_group`为总relay写次数/总首跳组数。
- observe及旧schema的完成时间和占比为未测，不补零；缺KERNEL或分母为零
  时不输出占比；零首跳组时不输出平均每组转发次数。
- 分析器拒绝重复rank/core/iteration。累计值直接由原始ticks计算，
  不通过Markdown中已经舍入的每组均值反推。

占比只表示完成诊断模式下该核的时间构成。前后排空改变正常重叠，
因此高占比不等于正常模式已确定为转发瓶颈，更不等于可获得同等加速；
还需结合慢核/下游等待和timer OFF A/B。完成累计不含前置排空。
本次仅更新离线分析，不改kernel/schema，无需重装wheel或重采已有schema10数据；
使用新的`--output-dir`重跑分析，保留原报告。

observe的完成列为未测，不能当零时延；新schema没有FIRST_HOP的核标为no_relay，
不产生虚构的每组均值。按写次数摊销不是单条写延迟，跨核总工作量不是kernel时延。
无NPU测量前不宣称实际转发开销或性能收益。

schema10诊断首次交付的完整本地回归：526 passed、48 skipped、9 deselected，包含带空格目录迁移。
跳过项为缺C++编译器或POSIX环境。提供了原生诊断guard的OFF/ON mock，
验证observe不发同步、完成模式前后各一次同步；本机缺编译器，该mock未执行。
尚未CANN编译或NPU验证，Python协议/报表测试不能替代设备验收。

2026-10-21累计比例离线更新：相关回归46 passed、2项原生mock缺编译器跳过，
覆盖不等时长轮次、零转发轮次、旧格式/observe未测、缺失/零分母、重复轮次及目录迁移。
