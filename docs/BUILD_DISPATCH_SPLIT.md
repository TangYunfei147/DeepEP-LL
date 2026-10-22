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

## Distinct stage semantics / schema13 (2026-10-22)

User explicitly prohibits using one stage name for different work.
Renamed ambiguous dispatch/combine preparation, load issue/completion and lane
wait sites; no new business operations, barriers or task/packet ordering.
Schema13 retains old IDs0..80 for offline reading and appends58 names (139 metrics).
Trace192 retained, carrier grows72→88 slots (4608→5632B/core), 512B aligned.
Old schema7/9/10/11/12 captures retain72-row validation and original names.
New ON captures require matching schema13 wheel/parser; wire/core19000+S unchanged.

Validation:
- Full local CPU suite868passed/109skipped,89.39s (.tmp/stage13-full-001.log).
- Native ON/OFF recorder, capacity/GM export, Host selector, format/semantic
  checks42passed,3.99s (.tmp/stage13-native-001.log). Overlaps full-suite tests.
- Additional unique-name inventory and semantic checks run after doc generation.
- Seven business headers stripped of LL_SEG labels/comments match9e24a79 tokens.
  This is source parity, not a CANN compilation or device equivalence proof.
- No CANN cold ON/OFF build/full wheel or NPU test in this revision. Existing
  A3_163_docker connection was unavailable in the preceding attempt.
  New timer metric arrays may change ON overhead; never interpret old/new ON
  kernel ratio as OFF algorithm speedup.
- Native/core/ABI/shape guards retained; trace still stores first2 intervals per
  stage plus all-call metrics, and preserves KERNEL/SEND_SUBMIT_END reservation.
  Trace overflow remains explicitly counted as omitted.
- Stage name/locations and repeated-name exceptions documented in
  tests/ll_stage_timer/STAGE_SEMANTICS.md.
