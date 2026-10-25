# LL 编译维护约束

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
- 影响模板展开、打点规模或构建图的变更，须在获授权编译环境测量 ON/OFF 的独立空对象目录，并验证完整 ON wheel 的链接。只改文档或 CPU 检查不必重复设备编译。
- 对照必须固定工具链、SDK、模式集合、优化级别、核数和并行度；分开记录冷编译、暖构建、SDK准备和打包时间。失败、跳过、审批未执行均须如实记录。
- 修改缓存逻辑时另查无改动重建与头文件修改后的依赖重编；不得只凭源码摘要跳过真实构建系统的依赖检查。
- 交付前更新 `docs/BUILD_DISPATCH_SPLIT.md` 的实测证据。向其他分支移植时检查其入口是否已拆分，不得因 NDR 分支已有修复而假定其他分支也已修复。

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
