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
