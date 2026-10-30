# 泛化分支生产LL路径

当前打点为reference schema7，参考metadata-cores@c3d8640的记录和解析方式，并适配本分支实际兼任S/R。旧stage/expert/sparse代码和解析器已删除；combine无打点。入口tests/ll_reference_profile/one_click.sh，说明见[reference指南](../ll_reference_profile/README.md)及[阶段映射](../ll_reference_profile/STAGE_TIMELINE.md)。业务与原SDK保持。

LL打点源码剥离实验由`scripts/prepare_ll_no_timer.py`和`ll_no_timer_python.py`在独立clone生成，不直接编辑本表的生产源码或cb59生成目录。剥离范围覆盖五selector、combine、Host launch/pybind、Elastic attach及ST诊断入口；业务token与Scalar/DMA/同步序列须保持。目标编排`tests/ll_stage_timer/no_timer_ab.py`，三wheel先建后测，详见`../ll_stage_timer/NO_TIMER_AB.md`。

最新两项实验：`cb59-pair`走`low_latency/cb59_pair/`，`pair_handoff.h`共享相邻两count-owner的区间，`receive_tasks.h`将dense半区映射回原record/q；预算与规划在`common/ll_pair_receive.h`。`cb59-small`走独立生成目录`low_latency/cb59_small/`，只能由`scripts/generate_cb59_small.py`生成，强制1<=B<=M<=32并在编译期消除dedup/大B计数选择，保留七dtype。当前使用reference schema7；旧schema16/15只在原Git提交中解析。指南`../ll_stage_timer/CB59_PAIR.md`与`CB59_SMALL.md`。

分片接收候选见`../ll_stage_timer/CB59_STREAM.md`：`cb59-stream`在EP16小direct走`low_latency/cb59_stream/kernel.h`，使用原cb59的source counts/Sender；`stream_counts.h`实现上三角sum、前序偏移和专家计数，`receive_tasks.h`按record分片，`receiver.h`复用原packet消费逻辑。`common/ll_stream_receive.h`共用ownership/UB公式。其他shape走原cb59接收协议。新目录为独立手工实现，不直接修改cb59生成目录。

当前新增cb59算法对照：DEEPEP_LL_DISPATCH_IMPL选择current或cb59，公共七模式入口dispatch_variant_kernel.h只包含所选算法。cb59六算法在low_latency/cb59/，由scripts/generate_cb59_replay.py固定提取；count_layout.h/host_layout.h保留旧私有布局和190KiB预算。公共Host、packet格式、combine及当前SDK共用。配置与测试见../ll_stage_timer/CB59_REPLAY.md。

该分支保留MTE传输和DEEPEP_LL_RANKS_PER_GROUP静态可配置分组，默认4。
不要复制NDR的Session、ll_transport或tiling-key策略来替代这里的实现。

| 目的 | 源码 |
|---|---|
| Host接口、可选诊断缓冲 | csrc/ops/low_latency.cpp |
| 分组默认值 | csrc/kernels/dispatch/low_latency/common/ep_memory_server_dedup_protocol.h |
| dispatch主入口 | csrc/kernels/dispatch/low_latency.cpp、low_latency/dispatch_server_dedup_kernel.h |
| 发送/接收/计数 | low_latency/device/dispatch_server_dedup_send.h、dispatch_server_dedup_recv.h、dispatch_server_dedup_counts.h |
| combine | csrc/kernels/combine/low_latency.cpp及low_latency目录 |
| dispatch reference记录 | low_latency/common/ll_reference_profile_*、ll_reference_receiver_log.h |
| 共同实验入口 | tests/ll_stage_timer/run.py、tests/elastic/test_ep_low_latency.py |
| 当前分析/表格 | tests/ll_reference_profile/analyze.py、export_xlsx.py、report_tables.py |

无生成snapshot需要手动编辑；保持生产算法与诊断的边界。

## 以下为历史算法维护记录（旧打点说明已被上述schema7替换）

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

小B并行前缀候选：device/dispatch_parallel_count_prefix.h、common/ll_parallel_prefix.h；
仅M<=32/B<=32，其他路径保持R0。Host/device预算同步于count_layout和ll_overlap_layout。
固定magic槽+64/+96记录本地总数/代次与完成代次，+32继续维护旧R0 epoch。
算法、额外开销和验证边界见../ll_stage_timer/PARALLEL_PREFIX.md。

2026-10-22 schema14分阶段诊断：`common/ll_stage_timer_profile.h`控制编译期探针组和紧凑累加槽。
新入口`tests/ll_stage_timer/diagnose.py`先采OFF正式性能，再分别采overview/prepare/prefix/send/recv；
ST `--ll-profile-capture`将实际capture调用放进独立profiler窗口，包含INIT/最终同步/Flush。
`analyze.py`调用`transfer_report.py`生成≤60000字符/字节的可恢复分块。
完整用法与扰动判断见`../ll_stage_timer/PHASED_DIAGNOSIS.md`；通信协议、收发顺序及业务UB不改。

编译限定：ll_stage_timer_device.h在包含profile helper时显式设置DEEPEP_LL_TIMER_DEVICE_CONTEXT，
使非模板DeviceTimer/RelayCompletionScope在Bisheng Host预处理阶段也引用__aicore__函数。
Host绑定直接包含profile头保持普通constexpr；不要只依赖__CCE_IS_AICORE__，也不要仅修补某个调用点。
回归tests/ll_stage_timer/test_profile_device_context.py包含真实Bisheng OFF/六组ON编译探针（不运行NPU）。

EP16小direct/P1计数接收候选：device/dispatch_count_ready_extract.h把ready判断和uint32计数提取合并为一个VF，由dispatch_parallel_count_prefix.h调用；原快照MTE2和V_S依赖、clear/前缀发布顺序不变。其他P/域保持旧路径。基线与测试见../ll_stage_timer/P1_COUNT_VF.md。
