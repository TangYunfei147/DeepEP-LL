> 当前逻辑组大小与每机物理卡数已解耦，配置及分支关系见 [LL_KERNEL.md](../../LL_KERNEL.md)。本文原有每机16卡描述仅对应旧部署/专用 run_ll16.sh。

# 测试脚本与 kernel 源码审计索引

生产LL每机16卡分组、同步协议和目标机验收见[LL16.md](LL16.md)，实际AIV核数查询及分工见[CORE_CONFIG.md](CORE_CONFIG.md)，剩余固定值见[HARDCODE_AUDIT.md](HARDCODE_AUDIT.md)。

本文件路径为 `tests/ll_probe/KERNEL_MAP.md`，以下链接均指向仓库内源码。先确定测量入口，再沿 Host 包装追到设备函数；`*_do` 是 Host 启动包装，不是 profiler 的设备 task 名。运行命令见 [PROFILING.md](PROFILING.md)。

## 正式采样入口与设备函数

| 测试入口 | Host / 绑定入口 | 应修改的 kernel 源码与函数 | 测量范围 |
|---|---|---|---|
| [profile_probe.py](profiling/profile_probe.py) `--kind comm --layer 0` | [comm_host.cpp](communication/host/comm_host.cpp) → `four_comm_do` | [comm_kernel.cpp](communication/kernel/comm_kernel.cpp) → `four_empty` | 同样的三个GM入参，完全空返回，不读tiling |
| 同上，`--layer 1` | 同一Host，scenario选择key | 同一文件 → `four_comm_1..4` → `Run<Scenario>` | 仅读取32B tiling；task名与传输模式相同，须根据报告layer区分 |
| 同上，`--layer 3 --scenario 1/2/3/4` | 同一Host准备SHMEM peer表和每发送rank独立slot | 同一文件 → `HasEdge`、`Run<Scenario>`、`four_comm_1..4` | 最简GM→UB→peer GM通信；场景3的fanout限制接收rank为1..N；修改通信图、分片、lane和搬运逻辑从这里开始 |
| [profile_probe.py](profiling/profile_probe.py) `--kind metadata` | [metadata_host.cpp](metadata/host/metadata_host.cpp) → `four_metadata_do` / `four_metadata_check_do` | [metadata_kernel.cpp](metadata/kernel/metadata_kernel.cpp) → `four_metadata` / `four_metadata_check`；Sender来自生成快照，见下节 | 正确性核逐项检查q/掩码/权重；正式报告只匹配计时核`four_metadata` |
| [profile_probe.py](profiling/profile_probe.py) `--kind peer` | [peer_probe.cpp](bandwidth/host/peer_probe.cpp)建立卡0/1 IPC映射 → `ll_transfer_probe_do` | [transfer_probe.cpp](bandwidth/kernel/transfer_probe.cpp) → `ll_transfer_probe` | 单进程双卡的旧版四模式搬运核 |
| [profile_probe.py](profiling/profile_probe.py) `--kind shmem` | [shmem_probe.cpp](bandwidth/host/shmem_probe.cpp)建立两rank对称映射 → `ll_transfer_probe_do` | 同一 [transfer_probe.cpp](bandwidth/kernel/transfer_probe.cpp) | rank0发送，rank1仅接收校验，不伪造发送task样本 |
| [test_ep_low_latency.py](../elastic/test_ep_low_latency.py)，dispatch | [elastic.py](../../deep_ep/buffers/elastic.py) `ElasticBuffer._dispatch_low_latency` → [csrc/ops/low_latency.cpp](../../csrc/ops/low_latency.cpp) 的`dispatch_low_latency`绑定 → `dispatch_server_dedup_kernel_do` | [dispatch/low_latency.cpp](../../csrc/kernels/dispatch/low_latency.cpp) → `dispatch_server_dedup_kernel`，内部实现见下表 | 完整生产LL dispatch |
| 同一脚本，combine | 同一Python文件 `_combine_low_latency` → 同一C++绑定的`combine_low_latency` → `combine_server_dedup_kernel_do` | [combine/low_latency.cpp](../../csrc/kernels/combine/low_latency.cpp) → `combine_server_dedup_kernel` | 完整生产LL combine，需要匹配的dispatch handle |
| 生成的 `native_ll/snapshot/test_ll_four_layers.py` | 原生LL调用链不变；[api_adapter.txt](native_ll/api_adapter.txt)增加独立API循环 | 上面两套生产kernel，不是另外复制的一套LL设备实现 | 同时保留profiler task与独立无profiler API时间 |

`comm` 的内部 `layer=0/1/3` 并不是四层实验的层号：空载和tiling都属于第一层，metadata是第二层，通信是第三层，原生LL是第四层。

## 元数据实验：原始源码、生成规则、编译副本

1. 生产源码是 [dispatch_server_dedup_send.h](../../csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h)。`ServerDedupSender::Process`负责UB/路由准备与调度；`CalTokenSendExpertCnt`计算q；`SendPacket`负责包构造与发送。
2. [prepare_snapshot.py](metadata/prepare_snapshot.py)将 `csrc/kernels` 复制到 `metadata/snapshot/kernels`，再给副本的`SendPacket`插入提前返回和诊断输出。**修改实验提取边界要改这个生成脚本；修改生产算法要改原始头文件。**
3. 实际编译包含 `metadata/snapshot/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h`。该目录被Git忽略，下次生成会覆盖，不应直接编辑或提交。`snapshot/identity.json`记录原Sender SHA256。
4. [metadata_kernel.cpp](metadata/kernel/metadata_kernel.cpp)使用`PROBE_METADATA_DIAGNOSTIC`编译两次：诊断核写逐路由q，计时核只写每核checksum，类名也分开以避免模板实例混用。

该实验保留路由读取、q前缀、去重分类、DIRECT任务及sourceMask/sourceWeights，但不含payload打包/量化/发送、跨rank计数交换、packet tail及receiver元数据。checksum写回和同步仍有观测成本，不能用其耗时直接相减得到“纯通信时间”。

## 完整 LL 内部审计位置

这些是生产kernel内部函数/模板，不是各自独立的profiler task。审计固定开销时应区分计数、控制、准备与数据路径，不直接把重叠阶段耗时相加。

| 阶段 | 源码 | 审计重点 |
|---|---|---|
| dispatch总调度 | [dispatch_server_dedup_kernel.h](../../csrc/kernels/dispatch/low_latency/dispatch_server_dedup_kernel.h) | `Init/Process`；实际N核入口中的1:1发送/接收分工，计数前缀与接收流水的顺序 |
| 本轮窗口状态 | [dispatch_server_dedup_magic.h](../../csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_magic.h) | `RotateDispatchServerDedupMagic`；每核本地槽、双窗口轮转；dispatch/combine共用 |
| dispatch计数 | [dispatch_server_dedup_counts.h](../../csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_counts.h) | `SendCnt/RecvCnt/CalcCumsum/CompleteCumsum/Finish`；跨rank交换和CTRL状态约定 |
| dispatch发送 | [dispatch_server_dedup_send.h](../../csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h) | `Process/Classify/CalTokenSendExpertCnt/FillTail/SendPacket`；路由、包尾、量化及MTE发送 |
| dispatch接收 | [dispatch_server_dedup_recv.h](../../csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_recv.h) | `Process/IssueLoad/ConsumeLoaded/ClearFlags`；等待、消费与输出元数据 |
| 接收任务和包定义 | [receive_tasks.h](../../csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_receive_tasks.h)、[receive_packet.h](../../csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_receive_packet.h) | 接收任务构造和包解释；修改包格式需联查发送端 |
| combine总调度及DIRECT回传 | [combine_server_dedup_kernel.h](../../csrc/kernels/combine/low_latency/combine_server_dedup_kernel.h) | `Process/CopyLocalPendingTokens/ReturnDirectTokens/NotifyServerCard/WaitServerReady`；准备与ready协议 |
| combine partial阶段M4 | [combine_server_dedup_partial.h](../../csrc/kernels/combine/low_latency/combine_server_dedup_partial.h) | `Process/SelectServer/Aggregate/Send`；partial归约、回传及完成事件 |
| combine最终聚合M5/M6 | [combine_server_dedup_source_aggregate.h](../../csrc/kernels/combine/low_latency/combine_server_dedup_source_aggregate.h) | `Process/InitDirectShared/DrainM6`；权重、源token输出及与M4交接的UB/事件生命周期 |

调整参数或容量前还应查看 [comm tiling](communication/tiling.h)、[dispatch tiling](../../csrc/kernels/dispatch/low_latency/common/ep_memory_server_dedup_tiling.h)、[dispatch data layout](../../csrc/kernels/dispatch/low_latency/common/ep_memory_server_dedup_data_layout.h)、[combine tiling](../../csrc/kernels/combine/low_latency/ep_memory_server_dedup_combine_tiling.h)及[combine buffer layout](../../csrc/kernels/combine/low_latency/ep_memory_server_dedup_buffer_layout.h)。探针参数与生产参数布局不同，不能互换。

## 其他脚本：构建、启动、校验、统计

下表覆盖新增实验目录中的辅助脚本；“不启动kernel”表示脚本本身只做CPU工作，不应把它当作设备实现。

| 脚本 | 对应源码 / 职责 |
|---|---|
| [build_profile_bridge.sh](profiling/build.sh) | 编译`bandwidth/kernel/transfer_probe.cpp`、`communication/kernel/comm_kernel.cpp`、两份`metadata_kernel.cpp`及对应Host，生成四个`lib*_profile.so` |
| [profile_bridge.cpp](profiling/host/profile_bridge.cpp)、[profile_bridge.h](profiling/host/profile_bridge.h) | `ll_probe_run`与`ProfilePhase(0/1/2/3)`桥接预热开始、进入采集、样本准备、结束采集；没有新增设备kernel |
| [profile_common.py](profiling/profile_common.py) | 控制Exp/all_reduce/目标调用次序与命名task解析；Exp/HCCL是外部torch_npu实现，不是本仓库通信kernel |
| [build_probe.sh](bandwidth/build.sh) | 编译`bandwidth/kernel/transfer_probe.cpp`，链接`bandwidth/host/local_smoke.cpp`与`bandwidth/host/peer_probe.cpp` |
| [build_shmem_probe.sh](bandwidth/build_sdk.sh) | 构建锁定SHMEM依赖库；不编译本仓库探针kernel |
| [build_shmem_driver.sh](bandwidth/build_shmem_driver.sh) | 链接`bandwidth/host/shmem_probe.cpp`与已有`bandwidth/build/transfer_probe.o`；先运行`bandwidth/build.sh` |
| [four_layers/build.sh](four_layers/build.sh) | 编译comm/metadata设备源码并链接对应独立Host可执行文件 |
| [run_local_smoke.sh](bandwidth/tests/run_local_smoke.sh) | `bandwidth/host/local_smoke.cpp` → `ll_transfer_probe`，本卡内容/边界校验 |
| [run_peer_validation.sh](bandwidth/tests/run_peer_validation.sh) | `bandwidth/host/peer_probe.cpp` → `ll_transfer_probe`，IPC双卡接收校验 |
| [run_scan.sh](bandwidth/tests/run_scan.sh)、[scan.py](bandwidth/tests/scan.py) | `bandwidth/host/peer_probe.cpp` → `ll_transfer_probe`，旧Event诊断扫描 |
| [run_shmem_scan.sh](bandwidth/tests/run_shmem_scan.sh)、[scan_shmem.py](bandwidth/tests/scan_shmem.py) | `bandwidth/host/shmem_probe.cpp` → `ll_transfer_probe`，旧SHMEM/Event扫描 |
| [validate_shmem_launch.py](bandwidth/tests/validate_shmem_launch.py) | 检查上述两rank分开启动和结果合并；实际kernel仍是`ll_transfer_probe` |
| [four_layers/run.sh](four_layers/run.sh) | 设置SDK、日志、可见卡并执行传入命令；具体kernel由所传Host/脚本决定 |
| [four_layers/validate.py](four_layers/validate.py) | `metadata_probe` + `comm_probe`；对应metadata/comm源码，使用Event诊断入口 |
| [validate_profile.py](profiling/validate_profile.py) | 调用`profiling/profile_probe.py`的metadata、empty、tiling、comm、peer、shmem校验项，覆盖上表探针 |
| [profile_probe.py](profiling/profile_probe.py) `--plan` | CPU检查指定通信域/场景/字节配置，不启动kernel |
| [four_layers/verify_adapter.py](native_ll/tests/verify_adapter.py) | CPU mock检查API计时边界和通信图，不执行真实LL |
| [test_profile_contract.py](profiling/tests/test_profile_contract.py) | CPU mock检查10/100/50时序、task匹配和统计窗口，不启动kernel |
| [analyze_profile.py](profiling/analyze_profile.py) | 汇总命名task报告与全rank校验；不启动kernel |
| [four_layers/analyze_comm.py](communication/tests/analyze_comm.py) | 汇总comm Host的Event诊断CSV；不是profiler task统计 |
| [merge_shmem_results.py](bandwidth/tests/merge_shmem_results.py) | 合并旧SHMEM双rank结果，不启动kernel |
| [analyze_ll_overhead.py](native_ll/analyze_ll_overhead.py)、[verify_analysis.py](native_ll/tests/verify_analysis.py) | 完整LL原始样本拟合及CPU校验，不启动kernel；截距不是空kernel实测耗时 |
| [network_probe/run.py](network_probe/run.py) | 编译复用生产Session，查询0与指定peer的UDMA已选网络层；[probe.cpp](network_probe/host/probe.cpp)与[copy.cpp](network_probe/kernel/copy.cpp)另做双向MTE小包校验，不测性能、不推断MTE物理选路 |
| [paths.py](paths.py)、[project_env.sh](../../scripts/project_env.sh)、[project_paths.py](../../scripts/project_paths.py) | 工程相对路径、CANN环境、仓内SDK、日志和缓存配置，见[PATHS.md](PATHS.md) |

## 修改后的最小检查

- 只改注释/文档：检查非注释代码是否变化、链接是否有效，并验证快照生成锚点仍匹配。
- 改探针kernel或Host：运行对应构建脚本，先校验输出/guard，再采样。通过profiler路径运行时使用`profiling/build.sh`。
- 改元数据提取：修改原始Sender或`prepare_snapshot.py`后重新生成，分别检查`four_metadata_check`与计时checksum。
- 改生产LL：重建匹配源码的生产kernel库/wheel；重建探针库不会更新已安装LL wheel。实际EP16/32验收由用户目标机执行。

所有目标机所需修改通过本分支Git提交获取。A5只使用物理卡0/1；生成文件和结果不提交。不要将旧IPC模式号、comm内部layer编号或metadata逻辑专家分区数当作生产LL的设备rank/tiling格式。

## 解析调度诊断

[PARSE_SCHEDULE.md](profiling/PARSE_SCHEDULE.md)：原ST的opt-in双卡临时搬运核模式，比较整条命令的即时解析、CSV后移、全部解析后移。设备函数仍是bandwidth/kernel/transfer_probe.cpp中的ll_transfer_probe，不执行原生LL协议。普通LL默认逐组解析不变。
