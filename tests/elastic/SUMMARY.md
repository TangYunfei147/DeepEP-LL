# summary.csv 性能统计

每一行对应独立case、route和dispatch/combine操作。新增字段：

| 字段 | 含义 |
|---|---|
| mean_kernel_us | kernel时长的算术平均值，微秒 |
| p99_kernel_us / p90_kernel_us / p50_kernel_us | kernel时长P99/P90/P50，微秒 |
| kernel_samples | 参与汇总的所有rank样本总数 |
| perf_tail | 每rank选取末尾样本数；0表示全部 |
| summary_statistics_scope | 固定为pooled_rank_tail_kernel_samples |

先按现有规则对各rank独立选取perf-tail，再合并该case/route/操作的kernel样本计算统计。分位数将排序后的样本按位置(n-1)*p线性插值，不平均各rank分位数，不取逐轮跨rank最大值，也不作为集体操作端到端延迟。原有rank均值的最小/中位/最大值及slowest_rank保留；ranks.csv保留逐rank统计，并增加p50_us和p90_us。原始样本不裁剪。

统计仅使用profiler的kernel/task时长，不混入Event、API或阶段timer数据。预热、采集轮数和默认尾段规则不变；未采集性能的case不会伪造性能行。正常执行tests/elastic/test_ep_low_latency.py后，在run目录查看summary.csv；无需新增开关或重编wheel。
