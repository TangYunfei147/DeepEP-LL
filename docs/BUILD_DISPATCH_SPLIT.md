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

## G4 group-compact candidate (2026-10-21)

Base: kernel/ll-g4-s3-baseline@ac876bab5c1f18155e9b47fc6446cc1d9bec7bc1.
Candidate: kernel/ll-g4-group-compact, implementation g4-group-compact,
protocol15000+S, unchanged Stage3/schema12 and seven translation units.

Local validation after changes:

- Full tests/ll_stage_timer + tests/elastic: 887 passed, 107 skipped, 172.41s.
  Skips are unavailable Torch/device/POSIX or C++ fixtures, not device passes.
  This includes 39 new tests; 38 execute the production compact scheduling
  helper or compare its results with the CPU route report, and one guards wiring.
- Separate existing Zig C++ checks for UB layout matrix, seven-mode Host selector,
  profiler matching and Stage3 policy: 13 passed, 29.47s. These overlap some
  tests above and must not be added to the full-suite total as distinct cases.
- EP16 B8/M256/H3584/K8/E384 BF16+FP8 CPU --plan passed; no NPU initialization.
- Source receiver/counts/local-send/combine algorithms retained. Only remote
  packet ownership, related source metadata access and UB planning changed.
- Pure CPU route accounting: EP16/64, B8/32/48/256, normal/hotspot, seed2;
  q/packet membership/receive row/receive core and total packet counts equal.

Reproduction:
python -m pytest tests/ll_stage_timer tests/elastic -q
python -m pytest tests/ll_stage_timer/test_dispatch_variants.py tests/ll_stage_timer/test_overlap_receive.py::test_native_layout_matrix tests/ll_stage_timer/test_overlap_ab.py::test_real_cpp_stage_policy -q

The new helper test accepts an existing CXX command. Local tests used an existing
Zig toolchain; it does not compile AscendC APIs or enforce AICore call domains.
Logs are under the candidate's .tmp/full-002.log, .tmp/native-001.log and
.tmp/plan-001.log. Generated files and raw route counts are not committed.

Initial local runs failed because pytest was absent from the runtime, its
subprocess isolation dropped PYTHONPATH, and sandboxed compiler temp-file access
failed. A project-local test venv with existing dependencies and project-local
temp/cache paths resolved these environment issues. Older source-policy tests
and the CPU report were adapted to the new owner stream; stale wheel rejection
now expects 15000+S.

No CANN ON/OFF cold build, full ON wheel, NPU correctness, or profiler A/B was
performed for this candidate. The historical A3 numbers above belong to the
earlier split-entry change, NOT this algorithm. Before adoption, run the same
benchmark_compile.py ON/OFF procedure and full ON wheel link on the authorized
target, then the correctness and OFF timing steps in
tests/ll_stage_timer/GROUP_COMPACT.md. No production architecture or SDK changes
were made.
