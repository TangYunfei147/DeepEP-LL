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

A/B分支的四阶段schema12验证另见OVERLAP_AB.md。四版均保持拆分/O2，
完整ON wheel及ON/OFF dispatch编译通过；不要将编译耗时当成阶段运行性能。

Stage5 prefix progression retains seven separate entries/O2. A3 cold dispatch
compile ON73.987s/OFF50.493s and complete ON wheel passed; see PREFIX_PROGRESS.md.
No compilation regression observed relative to the recorded Stage4 baseline.

Schema14 packet diagnostics (code f7075ad): A3/CANN9.3.0/O2/dav-3510,
S28/R28/jobs4, cold dispatch ON81.602s/OFF50.332s; full ON wheel passed.
All seven mode TUs remain independent. CPU/native suite: 322 passed.
Evidence: `.logs/packet-timing-002/`; details in PACKET_TIMING.md.
The ON measurement is above the recorded Stage5 schema13 73.987s; no claim
of compile-time speedup or NPU runtime improvement is made.

R-count/230KiB and shared-prefix/vector follow-ups retain seven TUs/O2. A3
S28/R28/jobs4 cold ON/OFF seconds: b2b2ed3 82.162/50.528; 0888f34 parallel
82.833/51.262 and r0 79.471/49.522; vector code5af523e parallel84.058/51.806
and r0 79.895/50.462. All five full ON wheel builds passed. Final CPU/native
suite381 passed. Evidence and target A/B steps: SHARED_PREFIX_VECTOR.md.
These are compilation timings, not kernel speedups; no NPU execution.
