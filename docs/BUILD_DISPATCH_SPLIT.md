# LL dispatch compile partition

## HNS PI SDK metadata compatibility (2026-10-26)

The MTE consumer accepts the known `HNS_1825` capability with an optional,
strict integer `hns_1825_pi_on_chip` value 0 or 1. Python reuse, SDK validation
and CMake admission agree. CANN version/timestamp, source revision/patches,
public headers and complete artifact checksums remain enforced; no RDMA/PI
definitions are added to DeepEP's MTE targets. Existing SDKs are not rewritten.
Bandwidth source `0ebb65b` differs from `e297d1b` only in those three admission
files and their CPU test; all kernel and runtime Python files are unchanged.
SDK regression: 69 passed; 17 real CMake configure cases skipped because CMake
is unavailable locally. No CANN compilation or device execution was performed.

## Sending-stage bandwidth diagnostic (2026-10-26, current)

The user selected business baseline `9027275`; the failed Vector-tail candidate
has been reverted. Schema17/profile8 (`bandwidth`) records six send categories
in a 1024-byte, 512-byte aligned carrier per core. Observe adds no synchronization.
Explicit `bandwidth-complete` stage3 dispatch captures isolate local phase
intervals with MTE3_S, including a first-hop prefetch flush at the DIRECT boundary.
They change scheduling and are diagnostic, not optimizations.

Seven translation units, all modes, O2/Ascend950, 152-byte launch arguments,
wire19000+S, SDK identity and production UB/lane budgets remain unchanged.
Internal `bw17` bridge symbols and Host schema checks reject mixed native objects.
CPU mocks and the immutable 53-file business-token guard are checked locally;
the guard is not a CANN code-generation proof. CANN compilation, target NPU
correctness and performance have not been executed by the assistant.

Validation covers 879 passing CPU checks and two skips (the historical schema13
source snapshot and POSIX Bash semantics on Windows), with affected compatibility
fixtures rerun after correction. The current native recorder is exercised OFF/ON,
including null-buffer argument suppression, exact counters, phase lifecycle,
completion/empty event pairs, prefetch split gating and all seven selectors.
The report's byte/operation formula agrees with the production C++ layout and
count ownership. A complete three-seed, five-capture synthetic report is one
14,957-byte UTF-8 file; this is workflow validation, not measured NPU performance.

[SEND_BANDWIDTH.md](../tests/ll_control_opt/SEND_BANDWIDTH.md) provides complete
isolated OFF/ON wheel builds, Host checks and target measurements. The separate
cold-build procedure later in this document remains available. Earlier sections
below describe historical revisions, not the current diagnostic identity.

## Prequantized sender Vector tail candidate (2026-10-25)

The sender candidate moves metadata-tail stores onto the Vector pipeline only
for asynchronous deduplicated S-core prequantized FP8 sending. It removes that
path's explicit input MTE2_S and packet S_V while retaining TQue dependencies,
two input buffers, output lanes and all Copy/publication ordering. One small
portable tail emitter is instantiated with device-callable adapters; there is
no new launch mode or translation-unit dimension. Seven dispatch modes, O2,
Ascend950, Host budgets, communication protocol and SDK remain unchanged.

The explicit PIPE_V between Vector tail generation and packing is part of the
candidate, not a diagnostic full-pipeline drain. Extra Vector instruction issue
can offset the removed Scalar waits; no performance gain is assumed.

Local regression: 514 passed, 2 skipped (historical schema13 source guard and
Windows/POSIX path semantics), including the required selector/source/ST checks.
The actual tail helper covered 104,544 valid-field/padding cases and 243 delayed
stream schedules. A negative control detected corruption from removing MTE2_S
while retaining Scalar tail writes. No packet UB or event IDs were added.
The unchanged business files retain their ed1 token checks; the sender is an
intentional dependency change and is checked separately for preserved routing,
q, ordering, source metadata publication, R-local and quantization algorithms.
The pinned A/B plan and existing suite/export checks passed 69 cases. The actual
CPU plan resolves two OFF builds and 12 complete ST invocations with identical
SDK dependency identity; a 24-row synthetic comparison exports one 4,597-byte
file. This validates the workflow, not measured candidate performance.

No CANN compilation or target device validation was performed. Use the pinned
OFF A/B in [SENDER_PREFETCH.md](../tests/ll_control_opt/SENDER_PREFETCH.md), which
also provides separate cold ON/OFF object and complete ON wheel checks. CPU
stream models are dependency tests, not timing or Bisheng code-generation proof.

## MTE3 dependency diagnostics (2026-10-25)

Current control-opt diagnostics use schema16/profile7 (`mte3`), with 18 first-hit
milestones and eight scoped wait statistics in the same 512-byte carrier. Seven
dispatch translation units, all input modes, production O2/Ascend950, the 152-byte
launch arguments, wire19000+S and UB/lane budgets are unchanged. Internal Host/
device bridge symbols now contain `mte3_16` to reject old-object mixing.

Observe adds no synchronization. Explicit dispatch TX/RX captures on S cores add
MTE3_S at selected lane-reuse/end boundaries, immediately followed by an empty
control pair. Statistics also measure existing input MTE2_S (the tested FP8 input
is prequantized), receive-table V_S, receive-payload MTE2_S and final PIPE_ALL.
OFF and ON calls without a capture buffer add neither waits nor clock reads.
These are purposeful diagnostic interventions, not production speedups.

Local checks completed: 532 timer/ST/runner regressions passed, 2 skipped
(historical schema13 source snapshot and Windows/POSIX path semantics); the
final focused capture/runner/new-and-historical-report batch passed 174 checks,
including the complete 720-record fixture and an unmodified-input, single-file
report of about 24KB. These batches overlap and their counts are not additive. Native CPU mocks
cover OFF/ON, real mode/core gating, event/control pairs, actual overlap metadata,
all seven selectors and the existing 53-file business-token reference. Removing
only diagnostic plumbing still matches ed1c553; this is not a code-generation
proof. Source-guard tests were updated for the new probe placement/schema; the
production waits and algorithm were preserved.

No CANN compilation, target NPU correctness or performance was executed for this
revision. Follow [MTE3_BACKPRESSURE.md](../tests/ll_control_opt/MTE3_BACKPRESSURE.md)
for two isolated full OFF/ON builds, Host identity checks and actual active-task
profiling. SDK identity is still validated by the standard builder. The separate
cold-compile commands below remain available; CPU mocks do not replace them.

The previous low_latency.cpp instantiated seven input/quantization modes inside
one global function. With schema11 timer ON, one Bisheng process handled the
expanded algorithm and all timers for every mode. This change compiles each mode
in its own translation unit, allowing the existing four compiler workers to run
in parallel and reducing the size of each optimized function.

All seven modes remain: prequantized E4M3, FP16 plain/E4M3/E5M2, BF16
plain/E4M3/E5M2. Logical group selection remains the existing compile-time
configuration. Production O2 and Ascend950/dav-3510 stay unchanged.

Host selection uses the same format builder as tiling generation, without D2H
or additional launches. Device entry validates its tiling against the selected
mode. The 152-byte launch arguments and communication protocol14000+S are
unchanged. Internal launch symbol is now dispatch_server_dedup_kernel_do_v2:
install matching Host/device libraries together; old mixed libraries fail symbol
resolution. Timer schema11, initialization/kernel/SyncAll/Flush order and object
lifetimes are preserved; send/count/receive/combine algorithms are unchanged.

Profiler names now end with one of the seven mode names. The ST parser accepts
old and new exact names but rejects mixed variants or task names in one sample
set. It retains the expected sample count and tail selection. No timers have
been removed and no output diagnostics disabled.

## Reproduce cold compilation

After loading CANN and preparing the locked SDK, from the checkout root:

```bash
python tests/ll_stage_timer/benchmark_compile.py --output .logs/cold-on-001 --timer ON --jobs 4
python tests/ll_stage_timer/benchmark_compile.py --output .logs/cold-off-001 --timer OFF --jobs 4
```

Output directories must be new. The timer reports wall time of all dispatch
translation units, including the Host launch bridge, plus per-unit durations.
It excludes SDK preparation, unrelated HT/combine compilation and wheel packaging.
Per-unit times must not be summed and called wall time when compiling in parallel.
A full wheel build is additionally required to validate linkage and Host binding.
No NPU execution is performed by these commands.

## 防止再次劣化

这类问题先在 NDR 分支处理过，随后在独立 G4 分支再次出现。修复不会自动跨分支传播；
移植功能或新建分支时必须检查实际入口结构，不能仅凭历史记录判断已修复。
项目根 `AGENTS.md` 已将入口拆分及验收要求设为维护约束。

- **结构检查**：`test_dispatch_variants.py` 检查 Host 桥不再实例化设备算法、每个
  TU 只有一个模式编号和公共入口 include、七种模式都进入 CMake，并保留计时顺序。
- **防止重新合并**：设备 target 显式设置 `UNITY_BUILD OFF`，即使调用方启用全局
  unity build，也不把这些模式重新合并成一个编译单元。
- **新增类型/打点时**：检查模板维度乘积和单函数规模，测量 ON/OFF；不要只测 OFF
  或单个模式。新增模式同时更新选择器、非法输入校验、构建清单和 profiler 测试。
- **排查顺序**：先区分依赖下载、SDK、Host、Device 和打包；记录慢 TU、编译器 CPU
  活动、参数与缓存状态。单个进程长期占满 CPU 时，优先检查展开规模，而非盲目增大 jobs。
- **对照方法**：同机同配置、独立空对象目录、顺序运行基线与候选。完整 wheel 另验链接。
  比较并行总墙钟时间，不把各 TU 耗时之和称为构建时延。
- **劣化预警**：同环境同期基线下，冷编译墙钟时间增加超过 20% 时应调查并复测一次；
  这是维护预警线，不是硬件无关的性能标准。73.501秒是本次环境证据，不是跨机器硬阈值。
- **缓存检查**：涉及缓存/依赖图改动时验证无改动重建与头文件变更后的必要重编；
  不允许为减少耗时绕过 depfile、SDK身份或 Host/Device 一致性检查。

CPU结构检查不需要 NPU，也不能代替真实 CANN 编译计时。目前未配置自动远端编译 CI；
上述命令由维护流程执行，不能把“存在测试文件”表述为已有持续性能监控。

## A3_docker measured results (2026-10-21)

Same CANN9.3.0, pinned SDK, S28/R28, G4, O2, dav-3510 and jobs4. Separate empty
object output directories; baseline finished before candidate started.

| Dispatch compilation | Wall seconds | Result |
|---|---:|---|
| Original combined entry, timer ON | 784.194 | passed |
| Seven entries + launch bridge, timer ON | 73.501 | passed |
| Seven entries + launch bridge, timer OFF | 55.234 | passed |

Timer ON wall time drops 90.6%, about 10.67x. This combines smaller-function
optimization and translation-unit parallelism; it is not a device performance
result. It excludes SDK/HT/combine/packaging time. Individual ON mode units took
34.78..36.96 seconds; the Host bridge took 2.38 seconds.

Baseline source: d2cc3e3 (9b5f980 only added environment documentation).
Validated candidate: 73b75a7, including concurrent schema11 preflight fix d1d8908.
Remote CPU suite: 305 passed, including actual C++ selector/launch fixtures.
Full timer ON wheel: successfully built and linked. All seven OFF device modes
and the launch bridge: successfully compiled. No NPU execution or wheel install.
Warm unchanged-source rebuild was not run: automatic approval timed out twice.
No claim about measured warm rebuild speed is made.

Remote checkout: /home/tyf/Q4_remote_test/DeepEP-LL-g4-overlap-recv.
Evidence inside that checkout:

- `.logs/compile-split-baseline-002/timing.json` and `low_latency.log`.
- `.logs/compile-split-on-001/timing.json` and per-unit logs.
- `.logs/compile-split-off-001/timing.json` and per-unit logs.
- `.logs/compile-split-wheel.log`; wheel in `.artifacts/compile-split/`.

The first measurement helper failed before compilation because /usr/bin/time
was absent; the checked-in Python runner removes that dependency. The first
candidate CPU pass caught a missing direct ll_core_config.h include in the new
Host bridge; 73b75a7 fixed it before CANN measurements.

## Update on the target

```bash
git pull --ff-only origin kernel/ll-g4-overlap-recv
export DEEPEP_LL_STAGE_TIMER=ON
python -m pip install -v --force-reinstall --no-build-isolation --no-deps .
```

Load the same CANN/environment and core split as before. Reinstall the complete
wheel on all ranks; no SDK rebuild is required solely for this change. Use the
existing ST correctness and observe commands in OVERLAP_RECEIVE.md. The updated
parser recognizes the specialized profiler entry names. Device correctness and
performance remain target-machine validation tasks.

## R0 prefix / vector receive tables (2026-10-22)

One commit restores S3/ac876ba send scheduling, removing both compact revisions,
then adds R-only SEND_COUNTS, a single R0-produced prefix, dispatch230KiB budget,
and vector pending/token-position initialization. Identity g4-r0-prefix,
protocol17000+S, Stage3/schema12; seven dispatch TUs unchanged.
Combine source and dispatch sender are byte-identical to S3.

Validation evidence:
- Full CPU suite first run: 848 passed,108 skipped,1 failed (177.54s).
  The failure required the intentionally removed per-core prefix broadcast.
  Updated to assert single publication/shared reading and retained fallback drain.
- Targeted repaired/source/epoch suites:26 passed,1 compiler skip.
- Native CPU batch with assertions enabled:64 passed (24.53s), including actual
  receive_tasks.h with delayed vector writes/alignment checks, R ownership over
  S1/10/28/55 and G1/4/16, Host dispatch selector and UB matrix.
  CPU stubs do not validate AscendC instruction support or real DMA ordering.
- EP16/B=M256/H3584/K8/E384/FP8 --plan passed, no device launch.
- Native layout for that shape, S28/R28: S105792B, R0 111776B, other R96032B,
  send/receive lanes6/6, no serial fallback. Software budget230*1024B.
- Logs inside candidate: .tmp/r0-full-final.log, .tmp/r0-native-final.log,
  .tmp/r0-ub.csv, .tmp/r0-plan.log. Generated data are not committed.
- Prior session approval review failed from account quota; that final command
  did not execute. Existing earlier native result63pass was recovered; resumed
  native batch64pass supersedes it.
- No CANN ON/OFF cold compile, full wheel linkage, NPU correctness or performance
  measurement for this revision. Historical A3 timings above do not apply.
  Follow the existing cold-build protocol and tests/ll_stage_timer/R0_PREFIX.md
  for complete target correctness and OFF profiling.

## Non-dedup dedicated S/R revision (2026-10-22)

Independent commit on top of4ec82a5, selected strictly by existing M<=32.
S sends every original route and returns without prefix/receive. R publishes
counts and reads the R0 prefix, then receives with core-S and R workers.
No R local payload descriptors are built for this mode. Dedup sender and
ProcessGroupLocal bodies are byte-identical to4ec82a5. No q optimization.
Non-dedup loop restored from local main07af8ea with configured S core count;
its8-rank large-domain ordering threshold is preserved.
Protocol18000+S;230KiB and R0 prefix remain. Full rank wheel rebuild required.

Validation:
- Full suite initial run847pass/108skip/3 old source-policy assertions failed.
  Updated the assertions for the intentionally changed mode split.
- Targeted regression122pass; native mode-isolation/receive-table/Host tests21pass.
- Final actual native layout matrix covers both dedup flags plus ST contracts28pass.
  Counts overlap; do not sum as unique coverage.
- Host non-dedup receiver minimum UB and Device overlap budget both use R workers.
- No CANN compilation, wheel linkage or NPU correctness/performance verification.
- Logs .tmp/nondedup-full-001.log and .tmp/nondedup-native-001.log.
  Existing CPU inspect_dispatch_routes.py retains historical S3 allocation;
  do not use it to assert current non-dedup core ownership. See R0_PREFIX.md.

## Local q table candidate (2026-10-22)

Base f7c7848; branch kernel/ll-g4-local-q-table. Seven dispatch TUs, communication
protocol18000+S and Stage3/schema12 remain unchanged. Only per-core q preprocessing
and lookup, corresponding qSub histogram capacity, and candidate capability guard added.

Required CPU/source checks plus q helper:45 passed,2 unavailable C++ selector skips.
Native q and UB matrix final batch:3 passed (2.18s); overlaps the initial q checks.
Actual production q helper tests both ownership modes and all shard boundaries.
Seven send/order/pack/local methods match base exactly; scheduler/counts/receiver/
core protocol match. No device code timing or performance claim.
Logs in .tmp/q-check-001.log; compiled fixtures in .tmp/q-native-final.
No CANN ON/OFF build or full ON wheel link performed; retain the documented target
cold-build procedure and complete dispatch/combine validation before adoption.

## VF-seeded local q table (2026-10-22)

Independent commit on kernel/ll-g4-local-q-table; A/B baseline af46122.
Reuse existing CountFrequency for [0,begin), then V_S and Scalar updates only
inside the original shard. Wire/protocol18000+S, schedule, original VF source,
UB layout and seven TUs unchanged. New local_q_seed_vf capability prevents
accidental testing with the previous Scalar-seed wheel.

CPU combined required regression/histogram/q:135 passed,3 skipped (4.20s).
Final actual VF mock + q update + UB fixtures:4 passed (3.09s), overlapping prior
checks. Covers masked partial input, zero prefix/empty shard, E384/1024 and
multi-segment65536 hot-bin counts. Mock checks arithmetic, not device scheduling.
Eight sender/order/lookup methods are identical to af46122.
Logs .tmp/q-vf-001.log; fixture output .tmp/q-vf-final.
No CANN compilation, full wheel link, NPU correctness or performance validation.
A3 cannot execute this A5-only VF API. Use target ON/OFF compile procedure above
and LOCAL_Q_TABLE.md for independent OFF A/B, including PREPARE_INPUT overhead.

## R0 uint32 vector prefix (2026-10-22)

One A5 VF replaces the serial per-entry prefix loop, six fixed register scan
steps per64-element block. Seven mode TUs and O2 stay unchanged; no new mode.
256B padded gather buffer is reflected in the common Host/device UB formula.
Config19000+S rejects old layout builds; wire/epoch/schema12 unchanged.
Native CPU semantic mock covers lengths0..1024, zeros, ones, random counts,
values beyond FP32 exact integers, masked tail preservation and gather bounds.
Related CPU/Host/profiler/layout tests51pass (4.97s), log .tmp/prefix-scan-001.log.
These simulate public VF operation semantics, not actual AscendC instructions.
A3_163_docker failed hostname resolution before any remote compile command ran.
No CANN ON/OFF cold build, wheel linkage or NPU performance has been validated.
Retain target cold-build and correctness/OFF profiling requirements.

## Prefix VF integrated with local q VF (2026-10-22)

Port of group-compact@9e24a79 onto local-q-table@b6c4efb as a separate commit.
Conflicts were documentation append conflicts; both q and prefix records retained.
Actual prefix VF, counts integration and Host buffer padding match the reference.
Local q VF seeding, packet schedule and receive transport remain unchanged.
Related production CPU/VF mock, q, Host, layout and ST checks:60 passed (7.98s),
log .tmp/q-prefix-001.log. No CANN compile, full wheel or NPU validation performed.
Build identifier19000+S requires matching full Host/device installation; wire and
epoch are unchanged. Compare against b6c4efb using complete OFF kernel timing.

## Communication-neighborhood schema13 (2026-10-22)

Reference: SimpleBrightMan/DeepEP megamoe-urma-tailCover@34d02e97168f799f9558d8999a22e258edc1b3be,
especially urma/kernel.hpp Put/Quiet/Publish/receive-tail boundaries and timer.hpp.
LL retains its own MTE semantics and delayed GM export. No additional completion
wait, pipeline barrier or transport operation was introduced in observe mode.

108 metrics,192 sampled event records,80x8 int64 per core (5120B). Seven dispatch
translation units remain; no extra kernel mode/template dimension. The extra
local timer state and probes may change ON register/spill/Scalar overhead.
No uninstrumented performance claim is made.
Communication protocol19000+S unchanged; native/Python/ST agree on schema13 and
reject stale ON wheels. All ranks need a matching full ON wheel and new captures.

Verification:
- Native recorder ON/OFF, trace-capacity/milestone preservation, schema parsing,
  report CSV/Markdown/Chrome export, Host selector and ST contracts:86 passed
  (5.03s), .tmp/comm-native-001.log.
- Initial full CPU suite:863 passed,111 skipped,2 obsolete tests failed.
  One compared raw source including new probes, the other treated all historical
  schemas as comparable. Business-token comparison and schema13-only strict
  diagnostic mismatch guard fixed these; targeted71 passed/1 POSIX skip.
- Final report/launcher guards:7 passed (0.21s). Counts overlap the above suites.
- Source business-token hashes for dispatch sender/receiver/counts match fb9263d
  after removing diagnostic macros and braces. See communication_baseline.json.
  This does not prove CANN compilation or instruction-level equivalence.
- EP16 B=M256/H3584/K8/E384/FP8 ON --plan passed.
- A3_163_docker hostname resolution failed before the remote preflight command
  executed. No CANN ON/OFF cold build or full wheel link; no NPU run.
  Do not substitute earlier successful builds for this timer revision.
- Acquisition/analysis and interpretation: tests/ll_stage_timer/COMMUNICATION_DIAGNOSTICS.md.
- Final full CPU suite:867 passed,111 skipped (149.83s), .tmp/comm-full-final.log.
  Additional final entry-to-first-send report tests:7 passed (0.25s).
  Entry-to-first-send uses same-core actual INIT start and first-send endpoint,
  never sums independent phase statistics. All validation batches overlap.

## Metadata A/B candidates (2026-10-22)

Candidate 1 reads receiver role once and outputs non-terminal destination directly
from header[0..2]. Original role snapshot, terminal destination, relay ordering,
all synchronization and lane retirement remain unchanged. No extra UB, launch
mode or template dimension. CANN compilation and NPU performance are unverified.
Candidate 1 local required contracts and relay checks: 70 passed, 3 skipped (no C++ compiler). Initial pytest temp-directory failure resolved by using project-local --basetemp.

## Four production control-path candidates (2026-10-22)

Branch kernel/ll-g4-control-opt starts at197052c, the production algorithm used
by the earlier ablation mode0. Commits0a44d45/4783fed/868ddf8/d00e87f separately
add source-mask accumulation, exact pending-suffix removal, an eight-block flag
VF, and optional metadata staging. There are still7dispatch translation units,
one dtype specialization per unit, O2 and Ascend950. The VF has a non-unrolled
batch loop; other block counts retain Sum. No new launch/template dimension or
wire protocol is introduced. See tests/ll_control_opt/README.md for UB/lifetime
constraints and the per-commit suite.

On authorized27, native source d00e87f compiled and linked full OFF and ON
wheels with CANN9.3.0, S30/R26 and4build jobs, covering all7dispatch entries plus
the existing other library targets. Pinned SDK4dd80e was copied locally inside
the new project and passed scripts/build_shmem.py reuse validation. Existing
Python had pybind11 3.1.0; the locked build wheels were uploaded from the local
machine and installed only into project .tmp/build-venv. No system dependency,
driver or pre-existing Python environment was changed. No NPU kernel was run.

Validation root on27: /home/tyf/Q4_remote_test/DeepEP-LL-control-opt.
Logs: .logs/control-wheel-off-002.log and .logs/control-wheel-on-001.log.
Artifacts: .tmp/wheel-off-002 and .tmp/wheel-on-001. Earlier validation-only
failures (missing include search path, absent pytest temp parent, mismatched
pybind11) remain recorded separately; they were resolved without weakening
the production compiler, architecture or SDK checks.

Real cache validation: unchanged full OFF wheel rebuild1.578s, zero changed
native object mtimes. A comment edit to dispatch_flag_probe.h rebuilt exactly
the7LL dispatch objects and relinked/repacked in38.216s; Host/HT objects stayed
unchanged. The original header was restored in finally. Records are in
.tmp/control-cache-timing.json, .logs/wheel-warm-off-003.log and
.logs/wheel-header-off-004.log. These are build times, not kernel performance.

Final Linux CPU suite:76passed, including actual compiled semantic mocks, all
Host selector tests and loopback readiness success/mismatch/busy propagation.
Local Windows lacked a native C++ compiler and blocked loopback; those checks
were completed on27. Device correctness and performance are left to the user's
EP16 target; the suite runs those gates before accepting summary.csv means.

### Core dependency preparation fix

The control-suite builder formerly copied every top-level Git dependency and
unconditionally required the JSON header used to compile SHMEM. That incorrectly
blocked core-only wheels when optional CATLASS was uninitialized, even with a
reusable SDK. Commit9ae2e5c selects only pinned SHMEM, reuses project-local SDK
caches without modifying their kernel worktrees, and requires json3.11.3 only
for a cold SDK build. The standard complete SDK validation still runs before pip.
Offline bundles may contain unused CATLASS records; they are not required.

Recovery was tested on27:98Linux CPU tests passed, including real Git/cache and
failure-reentry fixtures. A real baseline wheel was built with no CATLASS in its
isolated source and no JSON source asset, using the validated SDK. The original
suite --build-only --resume then reused that baseline and built all four candidate
wheels, publishing five verified manifests under .tmp/ll-control-opt/wheels.
No NPU devices were used. Existing failed attempts and caches were retained;
no cleanup, SDK rebuild, external download or kernel change was needed.

## Independent receive candidates (2026-10-23)

The default control suite now compares868ddf8 with three direct children:
85d05f4 restores source-mask UB writes,1c29599 probes8packets while retaining
16-packet allocation/lane budgeting, and c1e4faa pipelines two adjacent16-packet
flag snapshots. The prefetch uses one additional64B ID snapshot and only the
original spare UB. Seven dispatch TUs, production O2 and all input modes remain;
no new launch mode or public kernel ABI is introduced.

The user stated27was unavailable and requested local checks only. Source-mask
contracts45passed, batch8contracts/layout51passed, prefetch contracts/native
model52passed, and suite/cache/real-loopback58passed. These batches overlap.
Native CPU tests compile actual Host selection, flag VF under a SIMD mock,
pair orchestration, pending-table operations and layout helpers. They do not
compile the AscendC receiver with Bisheng or execute hardware DMA/events.

Local logs in DeepEP-LL-control-opt: .tmp/no-mask-contracts-001.log,
.tmp/batch8-contracts-002.log; prefetch log inside its validation worktree:
.tmp/rx-prefetch/.tmp/prefetch-required002.log. Initial Windows sandbox native
process/cache failures and default-encoding failures were resolved using an
approved local CPU check with project-owned cache/temp directories and UTF-8.
No SSH connection, SDK/system modification, CANN build or NPU execution took
place for these commits. Earlier d00e87f CANN evidence cannot be carried forward.

Target --build-only --resume compiles all four exact OFF wheels, including
regular native dependency checks on sibling checkout switches. New ON/OFF cold
timings and full ON linkage remain unverified while the target compiler is
unavailable. Commands and diagnostic limits:
tests/ll_control_opt/RECEIVE_CANDIDATES.md.

## Topology/SDK experiment validation,2026-10-25

The delivery LL tree adopts c1e4faa at3141120 and keeps it unchanged through
the official SHMEM update c705381. LL dispatch/combine source comparisons
between c1e4faa and c705381 are empty. Buffered8a23867 and Immediate0e9db99
are independent children of c705381; their only production-source difference
is the early-receive policy constant. Both preserve seven dispatch TUs, all
input modes, production target/O2 and the original lane counts/230KiB budget.

Buffered passed69CPU checks, including the real one-input helper, bounded
FIRST_HOP selector and pending table under S1/R55,28/28,30/26 and55/1.
Immediate passed13targeted checks after its policy was selected; the shared
native fixture covers both policies. Packet/consumer body comparisons remain
identical to c1e4faa. These checks do not execute actual AscendC DMA or events.

The final root suite passed155distinct CPU tests:153in the full run and the
remaining2continuation tests after shortening their temporary directory to
avoid Windows MAX_PATH. Earlier sandbox Git IPC/compiler failures were rerun
outside the sandbox with project-owned logs/caches. No production change was
made to bypass the failure. Logs: A5/.tmp/topology-root-final02.log,
topology-root-path03.log and topology-immediate-01.log; Buffered logs are in
the candidate validation worktree's .tmp/buffered-final004.log.

The SDK-specific50checks and39production-Session CPU lifecycle checks are
documented in SHMEM_MAINLINE_20261025.md. None of this is CANN or NPU acceptance.
Current target access remains unavailable. Compile all exact OFF wheels, run
their unchanged correctness checks and10/100/tail50sampling, then use matching
ON wheels only for timing diagnosis. See tests/ll_control_opt/TOPOLOGY_EXPERIMENTS.md.

## LL-only MTE initialization validation,2026-10-25

The earlier Elastic constructor always requested UDMA, even for the LL MTE
kernel. New baseline ed1c553 selects an explicit MTE-only Session and the
performance suite builds with DEEPEP_BUILD_HT=OFF. Buffered4d37eb6 and
Immediatec7d70a4 retain their respective original kernel sources. A new
commits-topology-mte.json selects these three exact revisions, separately from
the historical mixed-runtime results.

Within the user's isolated27directory, the unmodified8e3fc610 SDK was built
successfully and all three revisions compiled complete OFF and ON wheels:
six successful wheel builds plus six successful Host-only imports. Each import
checked HT disabled, explicit LL factory present, HT-only factories absent,
matching timer/core protocol and loaded SHMEM library hash.111CPU checks also
passed there, including the actual CMake source inventory and mocked Session
lifecycle. No NPU Runtime was instantiated or device workload launched.
Exact timings, artifact locations and rerun commands are in
tests/ll_control_opt/LL_ONLY_FIX.md. This validates compilation/linkage, not
target-device correctness or performance.

## Sparse baseline bottleneck capture,2026-10-25

Diagnostic source9c2c333 ports schema15 sparse milestones from
kernel/ll-g4-small-b-prefix@39eadb0557525f98f2528b4814901740366f46aa.
It preserves the ed1c553 business algorithm and all seven independent dispatch
entries, production O2, target architecture, configured S/R and business UB.
53 non-probe token streams match the ed1 snapshot; this is a source guard, not
a CANN code-generation or timing-equivalence proof.

Only14 first timestamps remain per core, with a512B carrier and192B export.
Legacy accumulation macros do no work. The removed SyncAll was diagnostic-only;
existing business barriers and lane ownership remain. Internal sparse15 launch
symbols and schema/profile checks reject mismatched Host/device objects.
PREFIX_LOCAL_DONE is a V_MTE3 submission-side boundary, not Scalar-confirmed
completion of the prefix VF. No extra synchronization was added to its probe.

The new profile_baseline.py builds and Host-checks complete OFF and SPARSE wheels
from the same pinned source before any device run. The actual attached sparse
capture calls are timed in a separate profiler window, including INIT/export.
Unknown or ambiguous task durations remain unavailable; ordinary unattached ON
calls and internal spans cannot substitute for this measurement.

Local CPU validation:443passed/2skipped across the timer suite, ST contract and
paired launcher, plus37passing compact-report checks using complete480-sample
disk fixtures. Skips are Windows/POSIX path semantics and the obsolete historical
schema13 business snapshot; the new ed1 token guard covers the current source.
The pre-existing local-q fixture now includes c1's64B flag snapshot, and the
non-dedup preservation test compares against the accepted ed1 source rather
than the pre-optimization split. No production algorithm was changed for tests.
Main regression log: A5/.tmp/sparse-baseline-final3.log.

No SSH, CANN compilation or NPU run was performed for this change, following the
user's updated local-check/direct-push workflow. The target commands and bounded
single-file handoff are in tests/ll_control_opt/BASELINE_DIAGNOSIS.md.

Current shared-pull uses its independent branch and per-consumer prefix-ready protocol. Fast and incremental manifests pin full OFF builds with SDK PI admission; source-mode/prefix/wire identities are checked against native imports. See tests/ll_control_opt/SHARED_PULL_PROBE.md. No CANN compilation or device run was performed here.
