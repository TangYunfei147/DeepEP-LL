# cb59 dispatch replay with current sparse capture

Reference: `cb59bff99ecd39d27ef57788c5e6fd0009e4e636` from
`kernel/ll-configurable-groups`. Select `DEEPEP_LL_DISPATCH_IMPL=cb59` or
`current` when building and running. The default remains `current`.

The user confirmed EP16/M32/H3584/K8/E384, prequantized FP8, timer OFF:

| B | Original cb59 us | Current P1/VF 39eadb0 us | Difference us |
|---|---:|---:|---:|
| 1 | 17.5150 | 30.7200 | 13.2050 |
| 2 | 17.5547 | 31.0403 | 13.4856 |
| 4 | 18.0762 | 31.2463 | 13.1701 |
| 8 | 18.1107 | 31.8050 | 13.6943 |
| 16 | 19.1610 | 32.8017 | 13.6407 |
| 24 | 19.9418 | 34.3384 | 14.3966 |
| 32 | 21.9017 | 34.5815 | 12.6798 |

These separate runs establish a useful faster reference, but do not isolate
which change caused the difference. Original cb59 defaults to G16 and queries
the actual even AIV count, splitting it1:1. With56 AIV it is S28/R28; it does
not support the modern send/receive environment knobs, so exporting40/16 does
not configure that commit. M32 disables dedup in both implementations: G alone
does not establish a relay explanation for the difference.

## What the replay preserves and controls

The replay preserves cb59's full-AIV count production for B<512, R-way count
reception, Scalar local prefix, R×R total/ready matrices, one prefix copy per R,
role-specific initialization, receiver scheduling,190KiB UB arena, and existing
DMA/event/barrier/cache behavior. It does not import the current P1 VF, shared
receive, priority/pending algorithm, q table or overlap preparation.

Six algorithm files in `csrc/kernels/dispatch/low_latency/cb59/` are generated
from the immutable Git reference. The generator changes only includes,
namespace and diagnostic plumbing. The checked-in reference token sequences,
synchronization sequences and negative mutation tests detect business changes.
Regenerate via `python scripts/generate_cb59_replay.py`; verify with `--check`.
Edit the generator for future extraction changes, not generated algorithm files.

Both replay and current builds use the same present-day Host/runtime, locked
SHMEM master8e3fc61, explicit MTE Session, ordinary page configuration, same
input/correctness/profiler code, and seven separate dispatch mode translation
units. Only the selected algorithm is instantiated per TU. Public metadata and
payload layout are shared; current combine remains available. The replay uses
its original private count layout and Host UB admission formulas.

For the first comparison both launch56 cores with S28/R28. Replay requires raw
P0, meaning the old R=28 producers; current uses P1. Replay preserves G16;
current keeps its G4 default. Native protocol21028 distinguishes replay from
current19028; the build metadata, plan, native module and all-rank checks also
verify `dispatch_impl`. A common source hash cannot identify these two builds.

This is **the old algorithm under a controlled current environment**, not the
original binary. Original cb59 used SHMEM dev4dd80e2, default1GiB huge pages and
MTE+UDMA initialization; its LL payload path was still MTE. Those settings and
the monolithic original device entry are not recreated in this first test.
If replay still cannot reproduce17–22us, the SDK/page/mapping/compiler-entry
differences remain separate candidates; do not immediately blame the replay
algorithm or automatically change these settings together.

## What to compare

Use full OFF profiler kernel tasks as the performance result. ON sparse capture
keeps schema15, the512B per-core carrier and14 first timestamps. It adds no
business barrier or final SyncAll. Replay senders are labelled send-only;
current's existing shared-receive capability label remains separately encoded.

Compare same-core ENTRY→PREFIX_READY, ENTRY→FIRST_RECV_PROBE and ENTRY→WORK_DONE
distributions, plus ON/OFF perturbation. Do not subtract cross-core clocks or
sum per-group maxima/percentiles. The algorithms have different producer
partitions, so the comparison tool withholds per-core stage acceleration ratios.

Two details matter when interpreting intermediate points:

- Replay `COUNTS_RECEIVED` follows the old count extraction/Sum and precedes
  local Scalar prefix and count clear. Current's point follows clear submission.
  Therefore this adjacent span is not an identical isolated micro-operation.
- Replay `PREFIX_READY` follows its original CompleteCumsum and PIPE_ALL before
  receiver reset; current follows its own existing handoff. The original
  synchronization costs belong to each algorithm and are not removed for timing.

Current has less shared-control write traffic but adds generation handling,
shared-prefix/cache-bypass accesses and overlap preparation. Replay has more
distributed work and replicated prefix writes. Byte count alone cannot decide
which is faster. The existing measurements do not attribute the full13us gap
to any one of these mechanisms. Do not remove current L2 bypass without proving
visibility and reuse correctness.

## First target comparison reported on 2026-10-25

The user supplied the complete first-pass summaries for `cb59_cb59_run_001`
and `cb59_current_run_001`, both at source revision
`c227a762024b4afda416cef2f6618a4b62aa686e`. Each case contains80/80
rank-rounds and56/56 cores, with no listed source issues. Both record
EP16/M32/H3584/K8/E384, FP8, cpu_sync0 and S28/R28. Replay records G16,
raw/effective P0/28 and protocol21028; current records G4, P1/1 and
protocol19028. These are complete summary observations, not a substitute for
the raw correctness logs, build manifests or repeated independent performance
runs retained on the target.

Full OFF profiler tasks, maximum of rank means, in microseconds:

| B | Historical original cb59 | Replay | Current | Current minus replay | Replay latency reduction vs current |
|---|---:|---:|---:|---:|---:|
| 1 | 17.5150 | 22.4722 | 31.2561 | 8.7839 | 28.10% |
| 2 | 17.5547 | 22.4240 | 31.3965 | 8.9725 | 28.58% |
| 4 | 18.0762 | 22.9355 | 33.4821 | 10.5466 | 31.50% |
| 8 | 18.1107 | 22.9689 | 32.9708 | 10.0019 | 30.34% |
| 16 | 19.1610 | 23.6320 | 33.9398 | 10.3078 | 30.37% |
| 24 | 19.9418 | 24.7771 | 34.4562 | 9.6791 | 28.09% |
| 32 | 21.9017 | 26.3766 | 34.9448 | 8.5682 | 24.52% |

The replay wins all seven cases under the current comparison setup. The
historical original column is a separate earlier run: replay remains
4.4710–4.9572us slower, approximately a fixed offset over this B range. This
does not identify SDK, page/mapping, compiler-entry or environment costs
individually. An original-binary rerun under the same target conditions is
needed before treating that remaining offset as a reproducible regression.
OFF compiles sparse Mark/Flush away; the observed ON perturbation is not an
explanation for this OFF residual.

Sparse ON evidence below uses the median of each rank-round's maximum
same-core span among the initial R group. Each range spans the seven B cases;
it is not a confidence interval or an additive decomposition:

| Span | Replay us | Current us | Interpretation |
|---|---:|---:|---|
| ENTRY to PREFIX_READY | 14.816–15.359 | 23.818–24.962 | Current is already8.6235–9.8635us later at prefix readiness. |
| ENTRY to first receive probe | 17.095–17.769 | 25.3835–27.059 | Most observed delay appears before reception starts. |
| ENTRY to WORK_DONE | 19.938–24.2765 | 28.416–33.0585 | The entry-span difference persists through completion. |
| First probe to first ready | 1.010–2.280 | 1.0785–1.984 | No corresponding large increase in current's first-packet wait. |

The initial R-group INIT medians are about2.8–3.1us for both algorithms.
Current's local-prefix span is about1.9us versus replay's0.29us, but its P1
producer processes384 count records while each of replay's28 producers
processes13–14. Here one record is one local expert from one source rank:
24 local experts times16 source ranks, not384 local experts or packets.
Their COUNTS_RECEIVED boundaries also differ as documented
above. This is not an isolated Scalar-versus-vector comparison. Count
production, distributed count reception, prefix layout/publication and
receiver preparation remain the main combined candidates; this experiment
does not isolate any one of them as the cause of the full OFF improvement.
With M32, both use the non-dedup path and only the28 R cores receive.
Current's sender role capability label does not mean shared reception was
active in these cases; all observed first probes belong to R cores.

The ON-minus-OFF full-task difference is1.5142–2.2688us for replay and
2.1539–8.1108us for current. Consequently, ON span differences cannot be
assigned one-for-one to the OFF gap. Current B8 also has one observed
rank0/iteration2/core28 PREFIX_LOCAL_TO_READY span of39.305us and an
ENTRY-to-WORK_DONE span of58.990us; its median prefix span is8.009us.
That is an ON capture outlier, not evidence of the same OFF tail behavior.

Next isolate the residual historical-versus-replay offset, then evaluate
individual changes on the faster replay path while preserving its existing
distributed control flow. The summaries do not justify assuming a fixed
gain from VF conversion, removing cache bypass, or reducing prefix producer
count. No kernel change or new target execution was performed for this
analysis.

## Build all four wheels first

Run from the target project root after updating this branch and loading its
existing Python/CANN environment. Reuse the already validated complete SDK;
do not rebuild SDK merely to switch the algorithm. Fresh names preserve old
results. These functions return failures without closing the terminal.

```bash
source /home/tyf/0env.sh
export DEEPEP_LL_SEND_CORES=28 DEEPEP_LL_RECV_CORES=28
unset SHMEM_ROOT CATLASS_ROOT
export LD_LIBRARY_PATH="$PWD/third-party/shmem/install/shmem/lib:${LD_LIBRARY_PATH:-}"

build_cb59_ab() {
  local impl prefix
  for impl in current cb59; do
    prefix=1
    if [ "$impl" = cb59 ]; then prefix=0; fi
    DEEPEP_LL_DISPATCH_IMPL="$impl" DEEPEP_LL_SMALL_B_PREFIX_CORES="$prefix" \
      python tests/ll_stage_timer/diagnose.py --build-only \
        --run-name "cb59_${impl}_build_001" || return $?
  done
}
CB59_BUILD_OK=0
if build_cb59_ab; then
  CB59_BUILD_OK=1
  echo 'Both algorithms: OFF and sparse wheels are ready.'
else
  echo 'Build stopped; keep its logs. Terminal remains open.'
fi
```

## Run sequentially on the allocated16-card target

Keep the same terminal/environment and set MASTER_ADDR, MASTER_PORT and
SHMEM_PORT to this job's actual allocated values. This does not authorize any
remote validation-host NPU use.

```bash
run_cb59_ab() {
  local impl prefix
  for impl in current cb59; do
    prefix=1
    if [ "$impl" = cb59 ]; then prefix=0; fi
    DEEPEP_LL_DISPATCH_IMPL="$impl" DEEPEP_LL_SMALL_B_PREFIX_CORES="$prefix" \
      python tests/ll_stage_timer/diagnose.py --skip-build \
        --wheel-map "diagnostic-runs/cb59_${impl}_build_001/node_000/wheel-map.json" \
        --run-name "cb59_${impl}_run_001" \
        --bs 1 2 4 8 16 24 32 --m 32 \
        --hidden 3584 --topk 8 --experts 384 --dtype fp8 --cpu-sync 0 \
        --devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
        --num-nodes 1 --node-rank 0 \
        --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
        --shmem-port "$SHMEM_PORT" || return $?
  done
}
if [ "${CB59_BUILD_OK:-0}" = 1 ] && run_cb59_ab; then
  echo 'Both algorithm runs completed.'
else
  echo 'Build incomplete or run failed; terminal remains open.'
fi
```

First send only the two `SEND_FIRST.txt` files under
`diagnostic-runs/cb59_{current,cb59}_run_001/`. The existing summary size and
ten-send budget remain; no full archive is needed. The two implementations use
identical uniform-random route seeds,10/100/tail50 OFF sampling and the same
separate sparse capture. This comparison does not assume an improvement.

Current validation evidence is in `docs/BUILD_DISPATCH_SPLIT.md`. Per the user's
workflow, this change receives local checks only; CANN compilation, target
correctness and timing are performed by the user.
