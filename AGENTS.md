# LL 编译维护约束

## 结果回传文件限制（2026-10-25，持续有效）

- 用户只能发送单个不超过65535、总数不超过10个的文件；导出时按每个文件实际UTF-8字节数不超过65535字节保守控制，总文件数不超过10。
- 优先提供少量可读的紧凑分析文件和导出命令，交付前核对数量和大小；不再要求上传全部summary、原始采样、日志或完整profiling目录。
- 原始数据留在用户机器，紧凑报告注明统计口径和省略内容；后续深入分析仍遵守同一回传限制。

## 最新工作方式（2026-10-25，优先于下方历史要求）

- 后续不再使用27（177.22.27.1）验证编译，不为此连接27、准备SDK、构建OFF/ON wheel或执行相关Host导入检查。此前隔离编译授权已结束，历史记录不授权后续操作。
- 完成修改及必要本地检查后，直接提交、推送，并提供对应提交和可执行的测试命令，由用户在目标机编译和测试；远端编译不作为推送前置条件。
- 如实标注未执行的CANN编译、设备正确性和性能验证；不擅自改用其他远端主机，也不反复询问27编译授权，除非用户明确更新本规则。

修改 LL kernel、模板、打点或构建配置前，先阅读：

- `tests/ll_probe/KERNEL_MAP.md`
- `docs/BUILD_DISPATCH_SPLIT.md`（两次同类编译劣化的处理经验与验收要求）
- 编译环境见 `docs/A3_COMPILE_VALIDATION.md`；远端操作仍遵循上级 AGENTS.md 的授权范围。

## 必须保持的结构

- `csrc/kernels/dispatch/low_latency.cpp` 只做 Host 选择；不得重新包含完整 kernel 实现或在一个 global 中展开全部模式。
- 每种输入/量化模式使用独立 `variants/*.cpp`，公共设备入口保持在 `dispatch_variant_kernel.h`。禁止 unity/jumbo 合并这些编译单元。
- 新增模式或模板维度时，先评估实例化数量和最大函数规模；同步修改入口清单、Host/Device 选择校验、CMake、OFF 编译覆盖和 profiler 精确匹配。
- 保持全部支持模式、生产 O2、目标架构、打点语义及同步顺序。不得用裁剪功能、关闭打点、降低优化级别来冒充编译问题的修复。
- Host/Device 启动接口变化必须拒绝混装；CPU 编译不能替代 Bisheng 的 Host/AICore 限定检查。设备回调须显式设备限定，避免普通 lambda 隐式成为 Host 函数。

## 验证与记录

- 运行 `python -m pytest tests/ll_stage_timer/test_dispatch_variants.py tests/ll_stage_timer/test_segment_source.py tests/elastic/test_ll_st_contract.py -q`。
- 影响模板展开、打点规模或构建图的变更，交付用户目标机的 ON/OFF 独立空对象目录测量与完整 ON wheel 链接验证命令；助手不再使用27执行，也不等待远端编译才推送。只改文档或 CPU 检查不必重复设备编译。
- 对照必须固定工具链、SDK、模式集合、优化级别、核数和并行度；分开记录冷编译、暖构建、SDK准备和打包时间。失败、跳过、审批未执行均须如实记录。
- 修改缓存逻辑时另查无改动重建与头文件修改后的依赖重编；不得只凭源码摘要跳过真实构建系统的依赖检查。
- 交付前更新 `docs/BUILD_DISPATCH_SPLIT.md` 的已完成检查和未执行验收项，不将本地检查写成设备验证。向其他分支移植时检查其入口是否已拆分，不得因 NDR 分支已有修复而假定其他分支也已修复。

## 当前 A/B 分支
四版策略、统一schema12口径和切换方式见docs/OVERLAP_AB.md。
仅通过ll_overlap_stage.h的提交常量选择阶段；不得让四个版本各自修改打点范围。
保留Host/device stage校验、各rank一致性及ST对安装wheel/checkout的校验。

## 当前R0前缀候选

kernel/ll-g4-group-compact现基于S3/ac876ba，撤销两轮compact改动。
身份g4-r0-prefix、协议17000+S。SEND_COUNTS仅R核，前缀仅R0生产，所有核读同一表。
固定epoch槽为R0 magic槽+32，先完成表写入再发布；消费者不清共享flag。
dispatch UB预算230KiB、combine独立预算不变。接收表向量化须处理32B对齐及V_S依赖。
当前指南tests/ll_stage_timer/R0_PREFIX.md。

非去重M<=32专用路径：S发所有route后退出，R仅计数/前缀/接收，不生成本组描述符。
去重路径不变，协议18000+S；Host/Device按M选择实际接收核数，不能仅按B判断。

## 本核q表候选

当前kernel/ll-g4-local-q-table从f7c7848派生；协议18000+S、收发调度不变。
qWork持久保存本片q，qSub播种私有专家直方图，内部量化只能复用qDst。
去重按token分片，非去重按route分片，R本组q不变。指南tests/ll_stage_timer/LOCAL_Q_TABLE.md。

q前缀播种现复用CountFrequency VF，V_S后只做本片Scalar更新；空片段跳过。
保持18000+S及收发协议，候选ST检查low_latency_q_seed_vf。A/B基线af46122。

R0前缀生成改为device/dispatch_prefix_scan.h的uint32 VF扫描：64项/段、6级gather/add、寄存器carry。
输入缓冲按256B补齐，Host与Device共用预算。配置标识19000+S，wire/epoch和schema12不变。
无逐项Scalar累加，不使用float转换；A3不支持该A5 VF，不能作为运行性能验收。

当前q分支叠加9e24a79 R0前缀VF，构建标识19000+S，q表VF播种保持。
PREFIX_COMPLETE本地专家计数提取仍Scalar；没有改变wire/epoch或发送顺序。

当前schema13通信诊断：108metrics/192采样事件/每核5120B。SEND_SETUP与首包前区间分开。
First区间从KERNEL起点到第一次DataCopy/Probe之前，空任务不补零；保留无新排空的observe语义。
新报告tests/ll_stage_timer/communication_report.py，参考COMMUNICATION_DIAGNOSTICS.md。

## Historical cumulative control-path candidates

kernel/ll-g4-control-opt starts at197052c and retains production protocol19000+S,
schema13, S+R56 and seven dispatch TUs. Four cumulative commits are pinned in
tests/ll_control_opt/commits.json. Preserve exact pending permutations; only a
fully ready suffix may be dropped without swaps. Fused flag masks apply to eight
blocks only and retain every flag word and original readiness equality. Metadata
staging uses spare conservative UB budget without reducing lanes; fallback keeps
the original writes. Scalar role/header snapshots precede FIRST_HOP rewrites.
Final metadata DMA retires at the original receive completion barrier.

The one-command performance suite uses the unchanged production ST entry with
OFF wheels, dispatch+combine and summary.csv mean_kernel_us (pooled per-rank last
50of100). Do not substitute max-rank mean or replay durations. Keep all artifacts,
loaded native/device/SDK identity checks, per-node readiness and idle checks.
See tests/ll_control_opt/README.md and docs/BUILD_DISPATCH_SPLIT.md for validation.
The core-only suite disables MegaMoE/fused components: do not require CATLASS or
its test/DSL dependencies. Reused SHMEM SDKs still need full standard validation;
only cold SDK builds need the JSON source asset. Preserve failed build caches.

## Independent receive candidates (2026-10-23)

Default suite baseline is868ddf8 (batch16). Independent direct children are
85d05f4 source-mask revert,1c29599 batch8 with unchanged16-slot allocations,
and c1e4faa paired flag prefetch. Metadata batching is absent. The delivery
branch keeps the baseline kernel tree and retains candidate commits as merge
parents; suite manifests always select an exact revision. Do not accumulate
these candidates or interpret adjacent rows as parent-to-child gains.

Prefetch keeps both pending snapshots immutable until consumed, removes B
before A, and checks spare64B only after choosing the original lanes. No
asynchronous probe may survive TryBatch or a cooperative callback. Preserve
MTE2_V6/V_S6 ownership, original payload event domains and serial fallback.
Pending traversal can differ from baseline. Timer RECV_POLL/RX_FLAG_REDUCE
submission/join fragments have different calls semantics; do not infer old
per-probe averages or DMA duration. See tests/ll_control_opt/RECEIVE_CANDIDATES.md.

The user explicitly deferred27connection and requested local checks only.
CPU checks do not constitute new CANN/NPU validation. Do not reuse historical
d00e87f build results as evidence for these commits. S+R56/G4/O2/protocol19000+S
and the original ST10/100/last50 protocol remain unchanged.

## Single-machine result continuation (2026-10-23)

User requested reuse after an interrupted suite and explicitly excluded new
multi-machine resume work. `--resume-results` is limited to NNODES1/node-rank0.
Keep the original output-dir/run-prefix and validate saved experiment identity.
Reuse each complete candidate/round only after checking both summaries, actual
all-rank raw100 samples and the tail50 mean; a done marker alone is insufficient.
Missing done markers do not invalidate complete summaries. Preserve partial
directories and rerun the whole item in a new attempt, never append samples.

New session plans/comparisons are separate from old files and record reused/new
provenance. No wheel/device preparation is needed if every item is reusable.
Explicit continuation may recover only a confirmed dead local suite-lock owner;
never kill processes or delete an active/unknown lock or its persistent OS guard.
Kernel/SDK sources and the ST sampling protocol remain unchanged.

## Topology trials and official SHMEM mainline (2026-10-25)

User confirmed16NPUs per host and contiguous groups of4. Per NPU,3ports connect
the other group members through fullmesh,4ports serve intra-host Clos and1port
serves inter-host traffic. QP IDs are not physical port IDs; LL here still uses
MTE. Endpoint packet accounting does not measure physical link utilization.

The delivery tree now adopts the measured c1e4faa flag-prefetch kernel.
3141120 retains its old SDK; c705381 changes SDK/compatibility only and is the
new experiment reference. Official SHMEM master was verified on2026-10-25 at
8e3fc610f91ae9e5e8cf13748cfb74826b272c2a (VERSION1.7.0). The submodule is
unmodified. SDK identity is checked fully and mismatches fail closed; explicit
rebuild archives project-owned old build/install. Never silently replace a live
SDK or copy an old SDK into a new dependency cache. See docs/SHMEM_MAINLINE_20261025.md.

The new pin removes the old UDMA peer-QP-mask setter and600000ms timeout override.
HT sparse requests fail explicitly; no dense fallback is selected automatically.
Its UDMA channel poll uses fixed120000ms. Single-host LL MTE is separate from
that transport path. CPU API/WQE tests do not establish CANN/device compatibility.

Use tests/ll_control_opt/commits-topology.json explicitly for the new experiment;
the old default manifest stays stable for result continuation. Match every
candidate/reference by round and requested seed; measured ST route is seed+2
because the original3correctness repeats remain. Default10/100/last50 and real
uniform top-k are unchanged. No fixed expert/rank/seed specialization.

Early receive candidates retain the original TX/RX lanes and230KiB software
budget. After every2actually submitted DIRECTs, a bounded16-slot probe selects
at most2ready FIRST_HOPs. Buffered policy keeps payloads in original RX lanes
until source sending finishes; immediate policy consumes at the callback.
They are separate committed candidates on the same new-SDK baseline. Input
lookahead stays active; control events are TPipe leases acquired after input
queue creation, with every callback's control joins closed. No new drain or
cross-core barrier. Source metadata remains at the original send tail.

Only local CPU work is authorized while27is unavailable. New CANN compilation,
NPU correctness and performance remain target-machine acceptance work.

## Historical LL-only initialization fix and isolated27build (2026-10-25)

The completed one-time authorization (now closed for future work) covered
177.22.27.1, limited to the isolated directory
/home/tyf/Q4_remote_test/DeepEP-LL-g4-small-b-prefix/.tmp/topology-mainline-20261025.
Source, dependencies, SDK/wheel builds, logs and caches stay inside it; preserve
the original deployment. All four device visibility variables stay0,1. Only
Host import/checks are allowed: no NPU initialization/test/benchmark. Report
compiler errors promptly. Inspect existing files/Git/tasks before writing.

User explicitly requested excluding HT from this performance experiment rather
than expanding HT compatibility. The suite sets DEEPEP_BUILD_HT=OFF; only the
LL Host and nine LL device translation units are built. Keep all seven dispatch
modes, combine, O2 and timer OFF/ON support. Build identity includes build_ht.
The LL-only native capability selects MTE initialization explicitly before
SHMEM; do not enable dense QPs to bypass the removed sparse API. Page selection,
SHMEM symmetric mapping and the8GiB+internal2MiB layout remain unchanged.
The old Elastic constructor always enabledUDMA even for LL, which caused the
reported missing-mask failure. New LL-only sessions reject HT fallback.

New commits-topology-mte.json uses a new run prefix and three same-SDK LL-only
versions. Historical four-entry manifests/results remain separate; an old
mixed-runtime result is not a new MTE-only measurement. No SDK source patch or
forced SDK rebuild is needed when the installed new SDK passes identity checks.

Isolated27validation completed: fresh unmodified8e3fc610 SDK,111CPU checks,
and complete OFF/ON wheels for ed1c553/4d37eb6/c7d70a4 all passed. Six Host-only
imports confirmed HT disabled, LL factory, timer/protocol and loaded SDK hash;
no native Runtime/device execution. Logs and wheels remain in the authorized
directory's ll-only-fix-001 child; see tests/ll_control_opt/LL_ONLY_FIX.md.
