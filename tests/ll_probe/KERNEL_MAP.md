# 泛化分支生产LL路径

当前业务恢复用户指定的9027275基线，失败Vector-tail候选仅保留历史固定源码入口。
当前schema17/profile8在counts.h、send.h、recv.h的真实DataCopy前统计六类发送，
由ll_stage_timer_device.h统一保存peer/self字节和显式完成区间。
observe不加等待；bandwidth-complete仅stage3 capture增加阶段MTE3_S，
首跳→DIRECT时flush最后预取包只存在于该诊断模式。
详细范围、统计口径和一键命令见[SEND_BANDWIDTH.md](../ll_control_opt/SEND_BANDWIDTH.md)。
旧[SENDER_PREFETCH.md](../ll_control_opt/SENDER_PREFETCH.md)和
[MTE3_BACKPRESSURE.md](../ll_control_opt/MTE3_BACKPRESSURE.md)仍使用各自固定历史提交。
下文更早打点为历史；业务分工、wire和lane预算不变。

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

输入等待打点：SendPacket原有MTE2_S、量化末尾V_S分别记TOKEN_INPUT_WAIT、
TOKEN_QUANT_WAIT；schema9追加ID64/65，不新增同步。分析入口额外输出
input_wait_bottleneck.md/csv，阈值与局限见 ../ll_stage_timer/INPUT_WAIT.md。

发送一包前瞻：common/ll_input_prefetch.h保持任务顺序，Sender拆分IssueInput与
ConsumePacket；两份输入及输出lane分配不变，MTE2_S使用独立分配事件。
测试/A-B基线与打点边界变化见 ../ll_stage_timer/INPUT_PREFETCH.md。

独立relay-priority实验：Receiver先发relay再输出本地结果；
common/ll_receive_priority.h对同批就绪FIRST_HOP优先排序，预取/消费顺序一致，
pending仍按原下标倒序删除。见 ../ll_stage_timer/RELAY_PRIORITY.md。

Relay诊断：schema10追加RELAY_ISSUE/PRE_DRAIN/COMPLETE/PACKETS。
普通observe无新同步；显式relay-complete在单组relay前后做本核MTE3_S，
只允许dispatch独立诊断。分析区分下发、完成、前置排空，见 ../ll_stage_timer/RELAY_TIMING.md。

纯CPU路由/分核报告：tests/elastic/inspect_dispatch_routes.py 复用ST路由生成，
按静态接收和批内relay优先规则输出数量及因果顺序，不运行NPU。
命令与统计口径见 ../elastic/CPU_DISPATCH_ROUTES.md。

Overlap candidate: common/ll_overlap_layout.h plans disjoint persistent UB and
event domains. Counts PrepareRecvCnt/TryRecvCnt and Receiver Prepare/TryBatch/Run
allow R-local send quanta to yield; S enters receive without retiring remote DMA.
Schema11, protocol14000+S, implementation g4-overlap-recv. Read
../ll_stage_timer/OVERLAP_RECEIVE.md before modifying this path. The old dispatch
non-probe token digest is intentionally no longer an algorithm-equivalence claim.

Dispatch compilation now uses low_latency.cpp as Host selector,
low_latency/variants/*.cpp as seven device entries, and
low_latency/dispatch_variant_kernel.h for identical entry/timer wiring.
Algorithms remain in dispatch_server_dedup_kernel.h and device/*.h.
See docs/BUILD_DISPATCH_SPLIT.md; keep profiler name matching and compile_off.sh
in sync when adding modes.

当前R0候选：counts.h删除片段总数/逐核前缀广播，ll_group_local_send.h仅给R核分配计数。
receive_tasks.h按记录段生成tokenPos、一次生成pendingIdx；Host和overlap核算R0峰值。
详见../ll_stage_timer/R0_PREFIX.md。发送算法恢复S3，compact共享任务表已删除。

非去重模式在主Process前置独立分支，Recv编号core-S且接收核数R；去重保持56核共享接收。

本核q表：common/ll_local_q.h负责分片与播种；sender在PREPARE_INPUT构建qWork表。
CalTokenSendExpertCnt只查表；Host/overlap核算qSub直方图最小容量。收发顺序和wire不变。

q表VF播种：device/dispatch_frequency_histogram.h复用既有实现，common/ll_local_q.h仅本片更新。

R0 uint32前缀VF：device/dispatch_prefix_scan.h；counts CalRecvAndSetFlag仅调用VF并同步V_MTE3。
count_layout DispatchSinglePrefixStageUbBytes包含256B对齐读缓冲，协议标识19000+S。

schema13打点与communication_report补齐发送初始化、q准备、首发、探测/依赖/输出/尾部。
源码业务token对照fb9263d存communication_baseline.json，计数/包格式/顺序不改。

2026-10-25 delivery树采用c1e4faa成对flag预取：common/ll_receive_probe_pipeline.h
编排两批稳定快照，recv.h在首批消费间提交第二批探测；64B额外快照只用原lane预算余量。
SHMEM更新基线c705381不改变该LL算法。独立Buffered/Immediate候选以
tests/ll_control_opt/commits-topology.json选择；其新增common/ll_early_receive.h
和S侧有界回调不默认进入delivery源码。修改前查../ll_stage_timer/BUFFERED_RECEIVE.md
及../ll_control_opt/TOPOLOGY_EXPERIMENTS.md；CPU检查不能视为CANN/设备验证。

当前主瓶颈诊断参考small-b-prefix@39eadb0的schema15 sparse：
common/ll_stage_timer_{layout,device,profile}.h只保留14个首次时间戳，512B/核，导出192B；
dispatch成对和串行探测都标记首次Probe/ready，combine仅入口/初始化/结束三点。
dispatch/combine内部launch符号带_sparse15拒绝旧对象混装；7个profiler入口名称不变。
原业务同步和c1算法不变，删除的是旧timer专用SyncAll。说明及OFF/active扰动验收见
../ll_control_opt/BASELINE_DIAGNOSIS.md，源码证据见../ll_stage_timer/sparse_control_business_baseline.json。
