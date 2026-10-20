# 泛化分支生产LL路径

该分支保留MTE传输和DEEPEP_LL_RANKS_PER_GROUP静态可配置分组，默认4。
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

本错峰候选只额外使用common/ll_send_order.h，由SendWithDedup调用；未引入NDR传输层。

收发编译配置：common/ll_core_config.h、cmake/CoreSplit.cmake及build_support.py；固定总数56，默认28+28，详见../ll_stage_timer/CORE_SPLIT.md。

共享接收分支：dispatch前S核完成发送后加入接收，全部56核按物理核编号静态分片；R核仍负责前缀生产。入口、counts前缀发布矩阵、receiver任务划分、Host UB预算必须一起维护。见../ll_stage_timer/SHARED_RECEIVE.md。

当前诊断为graph_dc_debug_seg语义的schema7（64阶段，前两次区间+全量metrics）；ll_stage_timer_device.h只在KERNEL结束且全核同步后Flush，旧ll_expert_timer仅用于历史记录接口，新采样明确拒绝。新增探针不得改业务等待/协议；静态摘要基线见tests/ll_stage_timer/segment_baseline.json。

本组初发：common/ll_group_local_send.h统一专家所有者；Counts在本组计数循环生成q/紧凑任务，Sender::ProcessGroupLocal由R核调用。Receiver和接收侧计数链保持不变；UB预算与源端分工一起维护，见../ll_stage_timer/GROUP_LOCAL_SEND.md。

本组初发打点适配：ProcessGroupLocal的准备和退休分别PREPARE_INPUT/SEND_DRAIN；非去重远端压紧记PREPARE_INPUT。分析按实际SEND_TOKENS提升显示角色并保留initial_role；schema7/参考64阶段字典不变。

远端纯计数直方图：device/dispatch_frequency_histogram.h 使用 asc_frequency_histogram；
CountByVectorRange 调用单段/有界多段模板，复用原输入和直方图 UB。
本组稳定 q 不使用总频数替代。见 ../ll_stage_timer/FREQUENCY_HISTOGRAM.md。
