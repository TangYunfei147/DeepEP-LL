# LL dispatch compile partition

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

## Small non-dedup candidate (2026-10-22)

Branch kernel/ll-g4-small-b, base197052c; only M<=32/B<=32 changes.
Sender20fb05b: empty route shard skip, B1 q=0, B2/4 tiny-prefix q, lane cap.
Count e12e478: direct range histogram on original IDs, no filter/GatherMask or
per-expert reduction for this gate. e3e773e corrects branch wheel/ST identity.
Seven original dispatch TUs, O2/dav-3510, S28/R28 and wire19000+S/schema13 remain.
No new template dimensions, mode pruning or new synchronization. Explicit UB
budgets are unchanged; only small-direct actual send lane use can decrease.

Validation:
- Local complete tests/ll_stage_timer + tests/elastic:870passed/113skipped,89.51s,
  .tmp/full-cpu-001.log; compiler/framework/device skips are not device passes.
- Authorized27 at /home/tyf/Q4_remote_test/DeepEP-LL-g4-small-b:150passed in5.39s,
  including actual G++ tiny-q/range-count fixtures, all zero-count publishers,
  duplicates and cross-token shard cases; .logs/cpu-small-002.log.
- Native CANN source-header cold compile e3e773e: all7modes+Host bridge OFF
  34.102s, ON61.534s, jobs4. .logs/source-small-{off,on}-001/timing.json.
  No baseline timing ratio measured. Source headers use locked SHMEM4dd80e2.
- CPU EP16 plans passed FP8 H3584/K8/E384, M32/B1,2,4,8,16,32 and
  dedup M256/B1,8,32,256; these plans never launch devices.
- Dedup protected source-region hashes match197052c; this is not machine-code
  or target-performance equivalence. Both large/small M>32 need device regression.

Initial full ON wheel preparation was blocked. Recursive submodule preparation attempted the
standard command, then stalled downloading CATLASS LLVM/Triton; only the verified
PID tree belonging to this project's download was terminated. The existing other
DeepEP-LL-ablations worktree was read only and its dirty files were preserved.
Independent SDK build reached CMake invocation but cmake is absent. A project-only
pip target installation of CMake/build requirements timed out after150s at that
point. The subsequent local-upload offline workflow below resolved this. No system dependencies, driver, or shared Python env
were modified, no wheel installed, and no NPU/communication task launched.
Initial missing .tmp test directory was corrected before the passing native run.

Environment source: /home/tyf/0env.sh after existing conda initialization. The
CANN path is named cann-9.3.0, while build_shmem's compiler/version.info reports
Version9.2.0; retain both facts rather than silently labeling this as a complete
9.3.0 SDK validation. Source compile succeeded against this actual environment.
Target instructions: tests/ll_stage_timer/SMALL_B_FAST_PATH.md.

Offline follow-up (user requires0env Python and prefers local upload to27):
- Confirmed Python executable /root/miniconda3/envs/py311_tyf/bin/python after
  initializing existing conda then sourcing /home/tyf/0env.sh. No Python replacement.
- Downloaded CMake3.31.6 aarch64 and all pinned requirements-build wheels locally,
  uploaded28,968,960B archive, SHA256 verified each wheel. Installed with --no-index
  --target .cache/build-python; PYTHONPATH/PATH apply only to this build shell.
- SDK build using existing JSON source completed; .logs/sdk-small-offline-001.log.
- Full ON wheel compile/link/package PASSED at e3e773e (same production C++ as
  the final documented candidate). S28/R28, ll_stage_timer=true, pybind11=3.0.4
  verified in wheel _build_info.json; g4-small-b Host identity found in _C binary.
- Artifact: .tmp/wheels-on/ascend_deepep-0.1.0.dev5-cp311-cp311-linux_aarch64.whl
  SHA2565c4ae93970d7fcbebb0e4f1636090d586c8a7b1a321191b5251b0cfab268eed0.
  Full log .logs/wheel-small-on-offline-001.log, under27 project root above.
- Wheel NOT installed; no NPU/communication execution or target performance claim.
  Cold OFF objects and ON objects plus full ON linkage are now confirmed.
- Python/CANN/offline reproduction: A5_SMALL_B_VALIDATION.md.

## Parallel small-prefix candidate and overhead audit (2026-10-22)

Branch kernel/ll-g4-small-b-prefix, baseline c68d8c6. Production source final
ae69d20; identity g4-small-b-parallel-prefix, wire/core19000+S and schema13.
Only M<=32/B<=32 selects the distributed prefix; dedup retains R0. Seven original
translation units and O2/dav-3510 remain, no new kernel-mode dimension.

Correctness/lifetime audit: unique per-R slice, uint32 inclusive scan, preceding
only total dependence, one shared prefix, publication-before-ready MTE3 fence,
fixed per-core generation across mode/shape changes, old epoch kept current, and
serial fallback retirement before Reset. New controls use existing zeroed magic
slot bytes64..127, not E-dependent banked scratch. Repeated calls require the
existing same-stream serialization and fixed communicator S/R.

Removed extra work: draft all-R total reads became preceding-only reads, count
reduction scratch is reused for snapshots, and prefix workspace is not replicated
per consumer. Final budget correction limits added reservations to R cores,
preserving S plans/lanes. Retained costs (including extra control traffic and
R short VF calls) are quantified in tests/ll_stage_timer/PARALLEL_PREFIX.md.
PREFIX_COMPLETE may shrink while BUILD_RECV_CUMSUM grows; target whole OFF task
time and first Probe are the acceptance measures, not one phase's label.

Validation:
- Full local CPU suite before final R-only budget guard:871passed/114skipped,
  87.91s, .tmp/full-prefix-cpu.log. Skips are compiler/framework/device dependent.
- Authorized27 native GCC tests:65passed in5.85s, .logs/native-prefix-002.log;
  includes actual VF mock, random producer ordering, empty slices, varying E,
  generations/wrap/mode switching and UB/control bounds. Initial fixture heap
  alignment violated the CPU VF mock's256B contract; aligned storage fixed it.
- Final budget/Q/prefix native regression:12passed in3.99s,
  .logs/prefix-final-cpu.log. Final local guard regression5passed/3compiler skips.
- CANN first attempt found two LL_SEG_SCOPE macros on one line (__LINE__ variable
  collision); split lines fixed it. No algorithm change from that repair.
- Final cold compile, validated installed SDK headers, S28/R28/jobs4:
  OFF37.614s, ON71.513s, all7modes+Host bridge passed. Logs/timing.json under
  .logs/prefix-{off,on}-final. These times are compilation, not NPU performance.
- Final full ON wheel compile/link/package PASSED. _build_info confirms timerON
  and S28/R28; Host binary contains g4-small-b-parallel-prefix.
  Artifact .tmp/wheels-on-final/ascend_deepep-0.1.0.dev5-cp311-cp311-linux_aarch64.whl
  SHA256d1f99c4a5a778c672f1ca54ed784a3120ceaf3ee019b6e0ad12ae261fa8c3308.
  Full log .logs/prefix-wheel-on-final.log.

All remote paths are under /home/tyf/Q4_remote_test/DeepEP-LL-g4-small-b-prefix.
Used conda initialization + /home/tyf/0env.sh (py311_tyf), local Git bundle upload,
validated offline SDK copy and project-local build tools from the prior small-B
validation. No external downloads, shared-environment changes, wheel installation
or NPU/communication launch. Target device correctness/performance remains pending.

## 2026-10-22 offline compressed transfer

Only tests/ll_stage_timer Python exporters, restore/render helpers, CPU tests and
usage documents changed. csrc, deep_ep, cmake and build_support are byte-for-byte
unchanged from0f6f02e; no ON/OFF native recompilation or device run was needed.
Local regression:494passed/29skipped in23.51s (tests/ll_stage_timer and the elastic
LL ST contract). Skips require a C++ compiler or POSIX environment. Compression,
missing/corrupt/mixed inputs, CRLF limits, oldv1 imports, multi-case restore,
exact decoded-field roundtrip and actual interval rendering are covered.
The two existing user-file byte benchmarks reduced transmitted text72.81% and
86.04% including headers/CRLF, with identical restored bytes. This is a data
transport result, not a device latency claim. See COMPRESSED_TRANSFER.md.

## 2026-10-22 timer profile Host/AICore qualifier fix

User compilation on72da8fe failed at RelayCompletionScope's initializer:
ProfileEnabled(RELAY_COMPLETE) was classified as Host inside an AICore function.
Reproduced unchanged72da8fe on authorized27: all seven OFF dispatch variants
failed with the same diagnostic; the Host selection bridge passed. This was a
source regression, not a successful final OFF build. The earlier all-profile
compile snapshot predated the final relay gating edit; only prefixON was
revalidated afterward. The original validation statement has been corrected.

Bisheng's Host preprocessing pass defines __CCE__ but omits __CCE_IS_AICORE__.
The existing conditional annotation therefore did not match the non-template
AICore classes it serves. The device timer header now explicitly establishes
DEEPEP_LL_TIMER_DEVICE_CONTEXT while including the shared profile helpers, so
ProfileEnabled and ProfileIndex retain AICore qualifiers in both compiler passes.
The ordinary Host binding still includes the profile header directly and keeps
normal constexpr helpers. No phase IDs, accumulator layout, runtime gating,
transport, synchronization, architecture, optimizer flags or SDK changes.

Validation used the existing/home/tyf/0env.sh Python/CANN environment, S28/R28,
Ascend950/dav-3510, O2, jobs4, new independent object directories:

- Before fix: OFF37.562s, bridgepassed/sevenvariantsfailed (not a usable performance baseline).
- After fix: OFF37.794s, sevenvariants+bridgepassed.
- After fix: ON/full87.941s, sevenvariants+bridgepassed.
- combine OFF separatelycompiled,9.027s.
-45 targeted remote tests passed in39.27s, including real Bisheng mixed-pass
  recorder probes for OFF and all six ON profiles, Host-only constexpr use,
  device include preprocessing under absent/Host-only/device macros, and actual
  native recorder/relay guard mocks. Probe kernels are compiled only.
- Complete ON/full wheel built and linked; isolated project venv import,
  profile/layout and exactsource fingerprint verification passed.
- Local required structure/ST/profile checks:76passed/21compiler-environment skips.

Evidence remains under the authorized project at
`.tmp/profile-annotation-fix/.logs/`:before-off.log,compiler-macros.txt,
pytest-qualifier-01.log,after-off/timing.json,after-on-full/timing.json,
combine-off.log,wheel-on-full.log,wheel-install.log. Wheel is in
`.tmp/profile-annotation-fix/.tmp/wheels-on-full/`.
No NPU task was launched, no SDK rebuilt, and shared Python was not modified.
After pulling thefix, use a freshdiagnose run-name(e.g.small_b_phases_002);
keep thefailed directory and existing build caches for diagnosis/reuse.
