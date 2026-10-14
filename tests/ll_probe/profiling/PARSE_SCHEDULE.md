# 解析时序与整条命令耗时诊断

本诊断从 `tests/elastic/test_ep_low_latency.py` 的显式 `--parse-schedule-probe` 分支进入，
只允许A5设备0/1、两个真实rank。使用现有 `ll_transfer_probe` 本地MTE拷贝作为临时算子，
HCCL进程组贯穿所有case。不创建原生LL通信域，不绕过原生LL EP16+限制，不代表目标机性能。
BF16/FP8仅用于选择拷贝字节数；没有真实FP8算术、专家路由、SHMEM工作区或dispatch/combine协议。

## 三种模式

| schedule | 每组采样结束 | 所有case采集后 |
|---|---|---|
| immediate | TorchNPU同步解析原始数据；调用原ST的read_measurement读取CSV并跨rank汇总 | 进程退出 |
| deferred | TorchNPU仍同步解析原始数据；保留读取任务 | 顺序执行相同read_measurement、汇总与结果保存，然后退出 |
| deferred_all | `tensorboard_trace_handler(analyse_flag=False)`，保存原始profiling数据 | 每rank逐组调用`profiler.analyse(max_process_number=1)`，再读取CSV、汇总与保存，全部完成后退出 |

原ST的 `measure()` 默认行为不变；只将原有解析逻辑提取成 `read_measurement()`。
每个对照完成相同数量的输出校验、10次预热、100次采集和最后50条各task统计；
每次采集前仍为Exp + fill + scalar all_reduce。A5使用64x64 Exp，不能推算参考13.53GiB缓冲下的性能。

默认13个B（1/2/4/8/16/24/32/48/64/96/128/192/256）×2个dtype，共26case、52组采样/rank。
诊断按原ST结构执行1次dispatch检查、2次combine检查及combine前刷新；两种操作都使用临时拷贝核。
默认以 immediate、deferred、deferred_all、deferred_all、deferred、immediate 顺序执行六个全新进程命令。
先使用pilot验证首次初始化与离线解析兼容，不将pilot的冷启动差异作为加速结论。

## 执行

所有路径相对工程根；已有CANN环境及仓内SHMEM SDK按项目标准流程准备。
如果临时核库不存在，先构建 `bash tests/ll_probe/profiling/build.sh`。

```bash
source scripts/a5_validation_env.sh
python3 tests/ll_probe/profiling/compare_parse_schedule.py \
  --pilot --output results/parse-schedule-pilot-new
python3 tests/ll_probe/profiling/compare_parse_schedule.py \
  --output results/parse-schedule-comparison-new
```

单次命令也可直接执行：

```bash
python3 tests/elastic/test_ep_low_latency.py --parse-schedule-probe \
  --schedule deferred_all --output results/parse-schedule-one-new
```

不要把普通LL的 `--num-processes 2` 当成此诊断入口。普通LL约束保持原样。
比较程序不设置大于两卡的可见性，四个设备可见性变量必须明确为0,1，否则拒绝运行。
失败不重试；超时只终止本次创建的进程组。生成数据保留在results，不提交Git。

## 计时与验收

- `command_wall_seconds`：父进程从启动Python命令前到该进程退出后的monotonic时间，包含
  Python/Torch初始化、两个worker创建、全部case、全部延后解析/汇总、退出清理。不包含预先构建。
- `capture_including_export_seconds`：run_profiled完整耗时；immediate/deferred包含TorchNPU解析，
  deferred_all只包含采集结束、同步与原始数据落盘。
- `close_sync_export_seconds`：Capture.close耗时，是上一项的子集，不能相加。
- `offline_analysis_seconds`：只在deferred_all模式记录TorchNPU后置解析时间。
- `parse_gather_seconds`：相同read_measurement函数的CSV读取、统计、跨rank汇总和samples.json写入，包含等待对端。
- 各rank的阶段时间相互重叠，不可相加当整条命令耗时。

各rank输出 `rank_0.json/rank_1.json`，每组原始trace和case结果均保留。
要求所有rank接收/本地拷贝校验通过、每组解析出完整100条目标kernel样本、全部case解析完成。
版本与源文件hash见identity.json；命令、每轮wall time见wall_times.json；差值汇总见summary.json。
两次重复只提供诊断范围，不构成严格统计显著性证明。若后移后的差异与运行波动同量级，应报告没有稳定收益。

注意：TorchNPU可能报告临时核的部分memory记录不完整。本诊断验收目标kernel记录数和拷贝正确性，
不使用该memory timeline推导显存峰值或分配器根因。

## 2026-10-14 A5验证结果

实际顺序为A/B/C/C/B/A；每种2次，26case×2操作，每rank每组10/100/50，Exp64x64。
计时从启动原测试脚本Python进程到全部解析和进程退出，单位秒：

| 策略 | 第一次 | 第二次 | 均值 | 相对当前方式 |
|---|---:|---:|---:|---:|
| A 当前方式 | 275.436 | 273.234 | 274.335 | 基线 |
| B 只后移CSV读取/跨rank汇总 | 264.870 | 275.328 | 270.099 | 平均少4.236秒，1.54% |
| C 原始profiling解析及CSV汇总全部后移 | 274.568 | 274.678 | 274.623 | 平均多0.288秒，0.11% |

B的两次波动为10.458秒，大于平均减少的4.236秒；与邻近A配对比较，一次更快、一次更慢。
本次没有观察到稳定提速。C基本只改变时间分布：采集/原始落盘阶段每rank合计约32–35秒，
之后仍需约208–217秒的TorchNPU解析，以及CSV读取、汇总和进程启动/退出等开销。
这里没有减少profiler会话数/解析次数，也没有增加解析并行度；不代表其他批量或并行解析方案无效。

所有624组rank级采样（6命令×26case×2操作×2rank）均通过目标kernel100条记录检查，
每个case本地拷贝正确性检查通过。测试替身、rank数和Exp缓冲均与目标原生LL不同，不能外推16卡LL的节省比例。
原始数据未提交，保留在验证工程 `results/parse-schedule-full-001`；summary.json与wall_times.json记录完整命令和分阶段时间，identity.json记录实测源码hash。
