# DeviceTimer fixed-overhead microbenchmark

Current schema15 compares the previous schema14 recorder at `98c5f50` with the
sparse recorder. The current capture records fixed first milestones, performs
no per-boundary accumulation and has no diagnostic SyncAll. The baseline keeps
its original boundaries and SyncAll/Flush. The scalar checksum workload is the
same; this compares two instrumentation designs, not matching per-stage costs.
The synthetic checksum loop does not model real packet waits or communication.
No schema15 NPU measurement is available yet: both authorized validation hosts
timed out on SSH on 2026-10-23. Historical results below are for schema14 only.

This standalone ACL executable uses the real `DeviceTimer` implementation. It
does not run dispatch, touch SHMEM, model traffic, or provide production latency.
It is intended to test whether an ON binary with a runtime null timer pointer
still pays constructor/stack initialization or boundary-dispatch costs.

After loading `/home/tyf/0env.sh` (and the existing conda shell initialization if
required), build from the checkout. No package installation or SDK is required:

```bash
python tests/ll_stage_timer/timer_overhead_build.py --output .logs/timer-overhead-001
```

The new output directory contains immutable copies of the timer/config
headers from `98c5f50` and the current checkout, their SHA256 hashes, the compiler
version, all commands/logs, four objects, and one executable. The current copies
include the profile header and uncommitted header changes. `--baseline` selects another Git revision.
Both revisions use the same S/R configuration from the environment (default
28/28), O2, dav-3510, and 56 AIV. The four implementations have distinct timer
type and symbol namespaces so they can be linked together safely.

Only launch when the authorized device is idle. The executable selects logical
device 0; all four device visibility variables must explicitly be `0,1` or `0`.
No other task is paused or terminated by this probe. Example:

```bash
export CUDA_VISIBLE_DEVICES=0,1 ASCEND_RT_VISIBLE_DEVICES=0,1
export ASCEND_VISIBLE_DEVICES=0,1 PROBE_VISIBLE_DEVICES=0,1
probe=.logs/timer-overhead-001/timer_overhead_probe
"$probe" baseline_off null empty 1 > .logs/timer-overhead-001/baseline-off-empty.csv
"$probe" baseline_on null empty 1 > .logs/timer-overhead-001/baseline-on-null-empty.csv
"$probe" current_off null empty 1 > .logs/timer-overhead-001/current-off-empty.csv
"$probe" current_on null empty 1 > .logs/timer-overhead-001/current-on-null-empty.csv
"$probe" baseline_on capture boundaries 1 > .logs/timer-overhead-001/baseline-capture.csv
"$probe" current_on capture boundaries 1 > .logs/timer-overhead-001/current-capture.csv
```

Also compare OFF and ON/null with `boundaries 1` and `boundaries 64`. The latter
executes repeated sender/receiver stage boundaries and a checked scalar
recurrence, without any payload processing or communication. It does not claim
the packet counts, overlap or instruction mix of a real B1 dispatch. All stages
used are elapsed-time stages, and the sequence is identical across revisions.

Every invocation preallocates events, results and capture buffers, performs 10
warmups plus 100 measured launches, and checks all 56 core checksums. Capture additionally validates each header, core identity and capacity,
using legacy metrics or sparse points as appropriate. Sparse capture uses the
same independent `Flush()` as the production entry; the baseline retains its
original final `SyncAll<true>()` and `Flush()`. OFF/capture is rejected. The timer pointer crosses the launch
ABI at runtime; it is not a compile-time null inside the kernel.

CSV fields contain raw ticks:

- `constructor_init_ticks`: first timestamp before object construction through
  `Init()`. Production INIT/KERNEL metrics exclude this interval.
- `init_span_ticks`: the INIT boundaries (legacy) or INIT_DONE point (sparse),
  plus surrounding timestamps.
- `kernel_span_ticks`: legacy KERNEL boundaries or sparse role milestones and
  WORK_DONE, optional scalar recurrence, plus surrounding timestamps.
- `export_sync_ticks`: legacy SyncAll/Flush or sparse Flush alone, plus timestamps.
- `total_ticks`: the sum of these four sequential intervals, within one core.
- `event_us`: full ACL event duration, including result export outside the
  measured device intervals. It is repeated on each of the 56 core rows; count
  it once per sample, never sum it over cores.

For schema15 the Host checks the sparse point bitmap and same-core order, while
the baseline checks its original INIT/KERNEL metrics. The schema15 boundary case
uses a fixed set of role-specific points, independently of the repeat count;
the baseline instruments the individual steps. Only the baseline capture uses
SyncAll. The `export_sync_ticks` CSV field is retained for comparison and means
export alone for schema15. All sources and headers are saved with the build.

The checksums/output stores and timestamps add fixed measurement work. Compiler
motion and code layout can also affect individual tick intervals, so compare
both device ticks and full-event durations, and repeat in reversed variant order
if the difference is small. No Exp/all_reduce workload is used. A null-timer
micro result cannot substitute for production OFF profiling or prove a network
bottleneck. Preserve ticks; apply the target's confirmed counter frequency only
when interpreting results.

## 2026-10-22 A5 validation host observation

Validation used host 177.22.27.1, logical device0 only, visibility0,1, and the
Python/CANN environment loaded from `/home/tyf/0env.sh`. Each launch rechecked
NPU occupancy and found no existing device task. No communication was run.
Compiler target remained dav-3510/Ascend950. The baseline was ae69d20 (timer
headers identical to d622b45); raw artifacts remain in the authorized project:
`.tmp/timer-profile-validation/.logs/timer-overhead-02/`.

| Synthetic case | old OFF event p50 us | old ON-null | current OFF | current ON-null |
|---|---:|---:|---:|---:|
| empty | 8.0655 | 8.0700 | 8.1005 | 8.1370 |
| one boundary sequence | 7.4455 | 7.2315 | 7.4155 | 7.4165 |

Active capture of the boundary sequence was18.0570us (old) and17.9470us
(current). Same-core export/synchronization p50 was8833/8902.5ticks. Values use
10 warmups/100 measurements and the final50; event values counted once per
launch, device tick medians across those launches/cores. These are separate
invocations, not a formal interleaved statistical speedup test.

No approximately21us ON-null penalty appeared in this probe. This does not
rule out production register/stack pressure or other fixed initialization work.
The roughly10us active-capture perturbation supports measuring full profiler
kernel duration for each diagnostic profile; it must not be subtracted as a
constant from real dispatch data. Individual inner intervals shifted between
OFF and ON because the compiler can defer argument loads and move work; the
constructor interval alone is not a clean measure of zero-initialization cost.
