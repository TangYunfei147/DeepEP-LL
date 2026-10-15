# 生产 LL 性能协议

此维护分支的完整 LL 采样实现在 `tests/elastic/test_ep_low_latency.py`。HNS_1825 显式部署入口为 `tests/ll_rdma/run_ll.py`，完整步骤见 [RDMA 验收指南](../ll_rdma/README.md)。它不依赖实验 checkout 的生成 snapshot 或仓外 example 模块。

保持已确认的 profiling.py 协议：默认预热 10 次、采集 100 次；每次采集前执行 Exp 与 scalar all_reduce，按 dispatch/combine 的 kernel/task 分别统计最后 50 条。Exp 默认 118260×30720 FP32，约 13.53GiB/卡。保留所有原始 task 和 trace，Event/API 时间与 kernel 时间分开。Stage timer 诊断调用另行执行，不纳入未插桩性能基线。

A5 只验证脚本/构建/允许的双卡原语；不伪造 LL EP16，不作目标性能结论。完整 EP16/32 数值、多机和性能由用户在目标机执行。没有匹配的目标机实测 MTE 基线时，不报告加速比。
