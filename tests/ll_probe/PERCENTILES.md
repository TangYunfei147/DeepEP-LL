# P50 / P90 / P99 输出审计（2026-10-15）

本次目标：实验分支的`tests/elastic/test_ep_low_latency.py`。只读参考独立
`DeepEP-LL-kernel` checkout的`kernel/ll-configurable-groups`（检查时cb59bff），
未修改该clone。其阶段timer分析已输出P50/P90/P99；其原生LL测试仍只有median/P95/P99。
实验分支原生LL的每rank统计本来已有P50/P90/P99，本次主要补齐汇总CSV和明确的汇总口径。

## 本次已补齐

| 入口/产物 | 修改前 | 当前 |
|---|---|---|
| `tests/elastic/test_ep_low_latency.py` 的每rank JSON / ranks.csv | 已有P50/P90/P99 | 保留每rank分位数与原始kernel_us |
| 同脚本summary.json / summary.csv / 控制台 | 汇总CSV只有rank均值，缺分位数 | 每kernel/task独立增加P50/P90/P99，明确为所选窗口的逐轮rank-MAX |
| `communication/tests/analyze_comm.py` | Event/API/job_join有P50/P95/P99，缺P90 | 三种时间分别补P90 |
| `bandwidth/tests/scan.py` | Event均值/中位数、Host均值 | 分别增加event_p50/p90/p99_us和host_p50/p90/p99_us |
| `bandwidth/tests/scan_shmem.py` | 同上 | 同上；只统计实际发送rank已有样本 |
| `bandwidth/tests/merge_shmem_results.py` | 继承发送rank字段 | 自动保留新增字段，不伪造接收rank时长 |

采集协议没有改变：默认预热10、采集100、最后50条，每次Exp+scalar all_reduce；
不同kernel/task分开统计。LL汇总新增字段：

- `p50_us`、`p90_us`、`p99_us`：所选tail窗口中，先逐轮取各rank时长最大值，再计算分位数。
- `iteration_rank_max`：该分布的完整统计；`rank_max_samples_us`保留对应样本。
- `percentile_scope`明确口径；不是全局时间戳对齐的makespan，也不是所有rank样本混成一个池。
- `ranks[].p50_us/p90_us/p99_us`仍是各rank自己的分布，不能与上面的汇总字段混淆。
- `min/median/max_rank_mean_kernel_us`旧指标保留；其值不是时间分位数。

线性插值位置为`(样本数-1)*q`。不要先对每rank求P99再取最大值代替逐轮rank-MAX分布。
`summary.csv`按operation/kernel/task逐行输出，`ranks.csv`按rank逐行输出，单位均为us。

## 已有完整分位数，无需补算

- `profiling/profile_common.py`及`profiling/analyze_profile.py`：每task、每rank、逐轮rank-MAX已有P50/P90/P99。
- `profiling/bandwidth_model.py`：延迟、有效带宽和参考利用率分位数已有；不同分布不互相冒充。
- `native_ll/api_adapter.txt`生成的独立API采样：调用原生`sample_stats`，每rank和rank-MAX都有三项。
  本次已重新生成快照并核对适配器。API时间继续单独写入api_samples.json/full_api，不放进kernel分位数。
- kernel分支`tests/ll_stage_timer/analyze.py`：每rank/core/stage已有P50/P90/P99，保留原始ticks；
  转us仍须显式计数器频率，不当作未插桩性能。

## 发现的其他缺口（本次未改动独立HT/通用benchmark框架）

| 位置 | 目前输出 | 缺口/说明 |
|---|---|---|
| `tests/elastic/test_dispatch.py::_statistics` | avg/median/min/max | HT路径缺显式P50/P90/P99；median等价P50值但字段不统一 |
| `tests/elastic/test_notify.py::_statistics` | avg/median/min/max | Notify及复用此解析器的HT Combine同样缺三项字段 |
| `tests/elastic/test_combine.py`的api-timings.json | 原始durations_us、kernel_durations_us | 该独立stream/API区间未输出分位汇总；不能混入kernel统计 |
| `tests/elastic/test_combine.py`诊断clean_profile | 只摘录kernel/samples/avg_us | 精简诊断摘要未带分位数 |
| `benchmarks/_utils.py::aggregate_samples`及CSV_FIELDS | P50/P95（nearest-rank算法） | 通用benchmark缺P90/P99；与本LL线性插值协议不同，需按其schema独立处理 |
| `profiling/compare_parse_schedule.py` | 整次命令walltime均值与降幅 | 未输出walltime分位数；这是解析调度诊断，不是kernel时延 |

本清单覆盖实验checkout的LL探针、elastic统计入口、通用benchmark以及只读参考分支timer；
不宣称审计了所有第三方依赖。`notify_profile.py`等单次阶段打点的跨核mean/max、拟合截距、
网络路由/正确性工具不是重复kernel时延分布，不为凑字段而增加伪分位数。

## 验证

CPU覆盖分位数线性插值、tail排除、跨rank慢者交替、样本长度不齐拒绝、JSON/CSV字段与Event/Host分离。
只改统计/报告；没有新增设备采样或把A5结果解释成目标性能。
