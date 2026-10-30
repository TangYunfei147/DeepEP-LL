# 当前 LL 采样协议

统一入口 `tests/ll_reference_profile/one_click.sh`，底层ST为
`tests/elastic/test_ep_low_latency.py`，协议标识 `initial-reduce-exp100-batch-v3`。

每个操作批次只在开始all_reduce一次，10次预热后同步；active profiler.step前执行100次Exp，
随后连续100次算子调用，循环内不插入Exp或采样all_reduce，按完整kernel/task取最后50条。
不在100次Exp之后额外同步。Event、API和device task分开，不相互替代。

独立reference capture同样使用初始all_reduce、预热同步、100次Exp和连续稳定/capture。
perf后的追加诊断另排100次Exp，共用perf初始all_reduce。恢复请求seed和相应handle，
采样记录实际profile_seed与完整路由指纹。

reference只支持dispatch；其最长单核w21-w0是内部阶段跨度，不是OFF完整task。
新旧逐轮Exp+all_reduce结果不可直接作同条件A/B。
解析和单轮/多轮Excel规则见 [reference指南](../ll_reference_profile/README.md)。
