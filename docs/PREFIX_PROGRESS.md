# Stage5: early R prefix publication and deferred prefix consumption

Baseline: experiment/ll-g4-overlap-ab Stage4 (`fbd1509`; documentation head0a2130b).
This single change is identified as overlap_stage=5. Historical stages1..4 remain
available by their exact commits. Do not edit the stage constant to reproduce them.

## Execution

On partitioned cooperative R cores:

1. SendCnt, RecvCnt and CalcCumsum execute before local payload preparation.
2. PreparePrefix allocates its existing disjoint UB once. An initial bounded
   ready check may issue the prefix DMA before local sender preparation.
3. If prefix flags are not ready, return to local sending. Check again at the
   existing eight-task quantum boundary. Never spin inside this callback.
4. On readiness, clear this core's ready row once, issue GM-to-UB prefix load and
   set MTE2_S event5. Return without waiting for the DMA.
5. At the next callback, consume event5 before reading prefix values or preparing
   receive tasks. If there are no sends left (including zero local tasks), the
   existing final progress loop consumes the load without inventing a send batch.
6. Subsequent callbacks only process bounded receive work.

When readiness was available initially, the load overlaps sender buffer/weight
preparation rather than automatically delaying receive by a full send quantum.
If readiness arrives later, consumption may occur after the next quantum; this
tradeoff must be evaluated on the target. RecvCnt/GetCumSum still wait for metadata
at the front. S and serial fallback retain blocking prefix acquisition.

## Lifetime and events

No additional UB packet buffers or atomics. Existing prefix/ready/clean/output
buffers remain allocated through final drain. Prefix state is core-private.
MTE2_S event5 belongs to the prefix only before receive starts. Sender input uses
6/7; its immediate MTE2_S control synchronization explicitly uses7. Prefix event5
is consumed before receiver lanes0..5 can reuse that event type. All initial-send
lane/MTE3 reuse protections remain. No new PIPE_ALL or cross-core barrier.
The existing prefix load's Set/Wait pair is separated in time, not duplicated.

## Diagnostics: schema13

The carrier stays4608B/core; stage IDs0..80 retain their names and new IDs81..83
are PREFIX_LOAD_ISSUE, PREFIX_LOAD_WAIT and PREFIX_READY_SPAN.

- PREFIX_COMPLETE accumulates active preparation/check/consume pieces. R sends
  between pieces are excluded; missing wall time is not claimed as saved time.
- WAIT_CUMSUM accumulates individual readiness checks including successful clear.
- PREFIX_LOAD is issue-to-consumption span; it can contain intervening work and
  must NOT be interpreted as pure DMA latency.
- PREFIX_LOAD_ISSUE and PREFIX_LOAD_WAIT isolate the request issue and actual
  wait instruction, respectively, without inserting additional synchronization.
- PREFIX_READY_SPAN covers preparation through usable-prefix completion, including
  intervening local work. Parent/child/span metrics must not be added.
  In the default core trace view, PREFIX_LOAD and PREFIX_READY_SPAN use separate
  per-core span tracks so crossing intervals are not misrepresented as execution nesting.

Schema12 remains readable offline but diagnostic ratios across12/13 are rejected.
Compare complete dispatch performance with timer OFF. The changed MTE2 ordering
can affect TOKEN_INPUT_WAIT as prefix and input loads share MTE2; they do not run
as independent DMA engines. Earlier count waiting can delay local initial sends.
This change does not establish a performance win without NPU A/B data.

## Target update

All ranks rebuild the same Stage5 full wheel, with the same S/R and shape as the
Stage4 baseline. No SDK rebuild is required solely for this change.

```bash
git pull --ff-only origin experiment/ll-g4-overlap-ab
export DEEPEP_LL_EXPECT_IMPLEMENTATION=g4-overlap-ab
export DEEPEP_LL_STAGE_TIMER=ON
python -m pip install -v --force-reinstall --no-build-isolation --no-deps .
```

Use the existing ST command and a fresh RUN_NAME, --ll-timer-mode observe; analyze
with this checkout's analyze.py. For OFF A/B, rebuild OFF and omit capture options.
Preserve seed, route, correctness repeats, S/R, warmup and sample counts.

## Validation (2026-10-21)

A3_docker, CANN9.3.0, Ascend950/dav-3510, O2, S28/R28, jobs4:
316 CPU/C++ checks passed, full timer ON wheel built/linked, all seven dispatch
modes plus bridge compiled with ON and OFF. Dispatch cold compilation wall time
was 73.987s ON / 50.493s OFF; these are compile times, not NPU performance.
No wheel installation or device execution. The original remote branch was restored.

Evidence in /home/tyf/Q4_remote_test/DeepEP-LL-g4-overlap-recv:
`.logs/prefix-progress-001/{cpu.log,wheel.log,on/timing.json,off/timing.json}`;
wheel in `.artifacts/prefix-progress-001/`. Validation used source snapshot e8f9665
before this documentation was added; the final single commit has identical
C++ and Python implementation/test sources. The original bundle is retained
under the authorized root as `prefix-progress.bundle`.
