# 泛化分支生产LL路径

该分支保留MTE传输和DEEPEP_LL_RANKS_PER_GROUP静态可配置分组，默认16。
不要复制NDR的Session、ll_transport或tiling-key策略来替代这里的实现。

| 目的 | 源码 |
|---|---|
| Host接口、可选诊断缓冲 | csrc/ops/low_latency.cpp |
| 分组默认值 | csrc/kernels/dispatch/low_latency/common/ep_memory_server_dedup_protocol.h |
| dispatch主入口 | csrc/kernels/dispatch/low_latency.cpp、low_latency/dispatch_server_dedup_kernel.h |
| 发送/接收/计数 | low_latency/device/dispatch_server_dedup_send.h、dispatch_server_dedup_recv.h、dispatch_server_dedup_counts.h |
| combine | csrc/kernels/combine/low_latency.cpp及low_latency目录 |
| 阶段/专家打点 | low_latency/common/ll_stage_timer_*、ll_expert_timer_* |
| 共同实验入口 | tests/ll_stage_timer/run.py、tests/elastic/test_ep_low_latency.py |
| 共同分析/对比 | tests/ll_stage_timer/analyze.py、compare.py、PARITY.md |

无生成snapshot需要手动编辑；保持生产算法与诊断的边界。
