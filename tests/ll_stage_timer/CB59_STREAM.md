# cb59 record-owned receive candidate

This is an independent candidate on `kernel/ll-g4-small-b-prefix`:
`DEEPEP_LL_DISPATCH_IMPL=cb59-stream`. The immutable `cb59` replay remains the
A/B reference. The current P1 implementation remains selectable as `current`.

The user requested receive scheduling and routing protocol redesign before
kernel initialization optimization. This change redesigns receive ownership
and the local count/prefix handoff. It preserves the existing source-side
count/send algorithm, packet addresses, packet format and public output order.

## Scope and identity

| Item | Reference | Candidate |
|---|---|---|
| Build selector | `cb59` | `cb59-stream` |
| Native implementation | `cb59-replay` | `cb59-stream-recv` |
| Core protocol | 21028 | 22028 |
| Cores / group / raw P | S28/R28, G16, P0 | S28/R28, G16, P0 |
| Stream path | none | EP16, M<=32, 1<=B<=32 |
| Other shapes | original cb59 | original cb59 receive protocol |

The per-case `effective_dispatch_impl` records `cb59-stream` or `cb59`.
`ll_dispatch_capability` records the seven modes, shape scope and fallback.
The shape gate is common to all cores in a call; resource admission failure
rejects the call before Session creation rather than selecting another path.
All seven input/quantization modes remain in their existing separate TUs, at
the same optimization level. No extra launch or device-to-host query is added.

Use complete matching wheels on every rank. The native id, core protocol,
manifest, source identity and all-rank checks distinguish this candidate from
the replay. Reuse the already validated project SDK; changing this selector
does not require rebuilding SHMEM.

## Receive dependency change

The reference already sends to fixed `source/destination/localExpert/q` packet
slots without waiting for the destination prefix. Its receiver waits for a
complete prefix, partitions the final total rows across R cores, and builds
packet tasks from that full table.

The candidate assigns each R core a contiguous slice of the E count records,
where one record is one local expert from one source rank. For E384/EP16,
the 28 receivers own 13 or 14 records each:

1. Wait for every count in this owner's slice, including explicit zero counts.
2. Compute its local inclusive prefix, local total and last-expert partial sum.
3. Publish that local total to later owners before waiting for any predecessor.
4. Wait only for earlier owners' local totals. Their sum is this slice's exact
   dense output offset. R0 has no predecessor dependency.
5. Construct packet tasks from the local prefix and start receiving. There is
   no whole-rank row count, full prefix reload or all-owner prefix-ready wait.
6. Write this slice of public `sendCounts`. The owner containing an expert's
   last record writes that expert's count. Predecessor sums and last-expert
   partial sums handle experts crossing slice boundaries, for both count modes.

The stable output sequence remains local expert, source rank, then q. Each
packet and each output row has one owner. The existing payload, scale,
source/token/original-k metadata, pending deletion, lane retirement and flag
cleanup operations are preserved. Combine can consume the same dense layout.

## Control reuse and UB lifetime

The existing R-by-R sum matrix is retained. Producer i writes only consumers
j>i; consumer j reads and clears only its own column for producers i<j.
Inactive owners receive no publications. Thus every written ready record has
one consumer and is cleared, while the unused triangle stays zero. The full
legacy protocol also clears its matrix, permitting shape-gated legacy and
stream calls to alternate in the same two-bank lifetime.

The source count record buffer remains outside the shared scratch arena.
Transient source-count vector work has retired before scratch reuse. Stream
control and receive work share a new 16-buffer pool over that scratch region:
eight persistent protocol buffers, two task tables, four probe buffers and one
packet-lane buffer. Each asynchronous publication/clear has a distinct source
that remains unchanged until the original final `counts.Finish()` drain.

There is no TPipe Reset or new full-pipeline barrier between stream count
preparation and reception. Empty owners still complete their protocol work;
the wrapper always executes the original final drain.

Host/device budget formulas share `common/ll_stream_receive.h`. The old
`ceil(total_capacity/R)` task bound is insufficient with record ownership.
The candidate conservatively uses
`min(total_capacity, ceil(E/R)*B*K)` rows per owner, plus persistent control,
source count records, probe buffers and at least one unchanged packet lane.
The UB limit remains 190 KiB; up to eight lanes are selected from the remaining
space. No observed uniform-route count is used to reduce the bound.

## What this experiment can establish

Full OFF profiler task time remains the performance result. Sparse schema15
and its 14 points remain available. For an effective stream case,
`PREFIX_READY` now means **this owner's dense output offset is ready**;
it does not mean that the full-rank prefix is complete. The one-file summary
includes this warning and the effective implementation. Cross-implementation
per-core stage speedup ratios are withheld.

Uniform small-B cases may benefit from removing replicated prefix publication,
the full-ready rendezvous and full-table receive setup. Hot expert routes may
concentrate work in fewer owners than the original row-balanced receiver.
Normal and hotspot cases must be reported separately. No latency improvement
is claimed before target measurements.

## Target procedure: build all four wheels first

Update this branch to the delivered commit, then use the project root and the
existing Python/CANN environment. Set the actual allocated MASTER_ADDR,
MASTER_PORT and SHMEM_PORT before any run. The device list below is for the
allocated 16-card target, not an authorization for remote validation hosts.
Fresh directory names preserve previous results.

```bash
source /home/tyf/0env.sh
export DEEPEP_LL_SEND_CORES=28 DEEPEP_LL_RECV_CORES=28
export DEEPEP_LL_SMALL_B_PREFIX_CORES=0
unset SHMEM_ROOT CATLASS_ROOT
export LD_LIBRARY_PATH="$PWD/third-party/shmem/install/shmem/lib:${LD_LIBRARY_PATH:-}"

build_stream_ab() {
  local impl
  for impl in cb59 cb59-stream; do
    DEEPEP_LL_DISPATCH_IMPL="$impl" \
      python tests/ll_stage_timer/diagnose.py --build-only \
        --run-name "stream_${impl}_build_001" || return $?
  done
}
STREAM_BUILD_OK=0
if build_stream_ab; then
  STREAM_BUILD_OK=1
  echo 'Both algorithms: OFF and sparse wheels are ready.'
else
  echo 'Build stopped; preserve the logs. Terminal remains open.'
fi
```

## Check dispatch and combine before performance sampling

This explicitly installs the candidate's already-built OFF wheel into its
isolated build venv. It does not rebuild or modify shared Python. Normal and
hotspot checks include repeated fresh routes, two-bank reuse and repeated
combine. The following full-matrix comparison will independently reinstall
and verify its own OFF/sparse wheels.

```bash
check_stream_combine() {
  local node py wheel route
  node=diagnostic-runs/stream_cb59-stream_build_001/node_000
  py="$node/venv/bin/python"
  wheel=$("$py" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); w=Path(json.loads(p.read_text())["off"]); print(w if w.is_absolute() else p.parent/w)' "$node/wheel-map.json") || return $?
  "$py" -m pip install --no-index --no-deps --force-reinstall "$wheel" || return $?
  for route in normal hotspot; do
    DEEPEP_LL_DISPATCH_IMPL=cb59-stream "$py" tests/ll_stage_timer/run.py \
      --devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
      --num-nodes 1 --node-rank 0 \
      --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" --shmem-port "$SHMEM_PORT" \
      --num-tokens 1 8 32 --num-max-tokens-per-rank 32 \
      --hidden 3584 --num-topk 8 --num-experts 384 --dtype fp8 --do-cpu-sync 0 \
      --route "$route" --skip-perf-test --warmup 1 --num-iters 1 \
      --correctness-repeats 6 --repeat-combine 2 \
      --output-dir diagnostic-runs/stream_combine_checks \
      --run-name "${route}_001" || return $?
  done
}
STREAM_CHECK_OK=0
if [ "${STREAM_BUILD_OK:-0}" = 1 ] && check_stream_combine; then
  STREAM_CHECK_OK=1
  echo 'Dispatch/combine checks completed.'
else
  echo 'Build incomplete or correctness failed; preserve logs.'
fi
```

## Run the matched OFF/sparse comparison

```bash
run_stream_ab() {
  local route impl
  for route in normal hotspot; do
    for impl in cb59 cb59-stream; do
      DEEPEP_LL_DISPATCH_IMPL="$impl" \
        python tests/ll_stage_timer/diagnose.py --skip-build \
          --wheel-map "diagnostic-runs/stream_${impl}_build_001/node_000/wheel-map.json" \
          --run-name "stream_${impl}_${route}_run_001" \
          --bs 1 2 4 8 16 24 32 --m 32 \
          --hidden 3584 --topk 8 --experts 384 --dtype fp8 --cpu-sync 0 --route "$route" \
          --devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
          --num-nodes 1 --node-rank 0 \
          --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
          --shmem-port "$SHMEM_PORT" || return $?
    done
  done
}
if [ "${STREAM_CHECK_OK:-0}" = 1 ] && run_stream_ab; then
  echo 'Both algorithms and both route distributions completed.'
else
  echo 'Prerequisite incomplete or run failed; terminal remains open.'
fi
```

Send the four `SEND_FIRST.txt` files under the corresponding run directories.
No trace archive is needed for the first pass; the existing per-suite ten-send
budget remains. OFF uses 10 warmups, 100 samples and each task's last 50. Sparse
capture is independent and reports its own complete-task perturbation.

For deployment beyond this first comparison, verify other modes and shapes
through the shared ST, including legacy fallback calls on the same Buffer.
This first target recipe focuses on the user's EP16/M32/H3584/K8/E384 FP8 case.

## Local validation, 2026-10-25

The integrated CPU suite passed 978 tests, with 75 environment skips and 8
existing Windows/POSIX compiler cases deselected. Explicit local C++ compiler
checks additionally exercised Host selection/budgets, the real StreamCounts
header in 49 cases with 28 CPU threads, and the real task builder. Independent
event models include legacy/stream alternation, delayed/zero counts, hotspots,
empty owners and cross-slice expert counts. The original six replay files pass
exact regeneration, business-token and synchronization-sequence checks.

CPU emulation does not establish NPU cache coherence, asynchronous DMA timing,
compiler code generation, device correctness or performance. No SSH, CANN
compilation or NPU execution was performed for this change. Cold compilation,
full wheel linking and target correctness/performance remain target checks.

## First user target A/B results, 2026-10-25

The user subsequently supplied all four `stream_{cb59,cb59-stream}_{normal,hotspot}_run_001`
summaries from `ae87d842d76d1bb5130df9870b71e44aac8eef4d`, with source SHA256
`cb08d1ecf819e922a5fc21aa8a48911ab4c4d8341abd7780291065e9db7ef139`.
All use EP16/M32/H3584/K8/E384, FP8, cpu_sync0, G16 and S28/R28/P0.
Every case reports 80/80 rank-rounds and 56/56 cores, with no listed source
issues. The recorded protocol/selection is 21028/cb59 or 22028/cb59-stream.

These are complete OFF profiler tasks, maximum of rank means, in microseconds.
Reduction is `(cb59 - stream) / cb59`; a negative value means this stream run
was slower. Each row is one reported A/B pair, without independent repeat-run
variance or a confidence interval.

| B | Normal cb59 | Normal stream | Reduction | Hotspot cb59 | Hotspot stream | Reduction |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 22.0172 | 18.8280 | 14.49% | 22.6302 | 18.2588 | 19.32% |
| 2 | 22.2737 | 18.8395 | 15.42% | 23.6069 | 18.9810 | 19.60% |
| 4 | 22.7154 | 19.3781 | 14.69% | 24.8244 | 21.6766 | 12.68% |
| 8 | 22.7884 | 19.0634 | 16.35% | 28.9663 | 26.2242 | 9.47% |
| 16 | 23.5796 | 20.3110 | 13.86% | 38.2222 | 38.7090 | -1.27% |
| 24 | 24.4532 | 22.3279 | 8.69% | 51.2471 | 53.1069 | -3.63% |
| 32 | 26.0140 | 25.6531 | 1.39% | 64.0493 | 65.4232 | -2.15% |

The candidate is useful for small B: both route distributions improve at
B1/2/4/8. Normal also improves at B16/24, while its B32 advantage is only
0.3609us. Hotspot B16/24/32 regresses by 0.4868/1.8598/1.3739us in this run.
Do not make an unconditional default replacement from these results; repeat
the boundary cases in reversed A/B order using the same existing wheels.

### What the sparse captures support

Normal ENTRY-to-first-probe median rank-round maxima advance by 2.6915–3.2260us.
For hotspot, the advance is 2.2210–4.3590us on the ten rank-rounds that actually
receive payload. Thus the intended earlier receive start is observed.
Initial R-group INIT does not decrease: normal medians rise by about
0.0925–0.3310us. This experiment did not optimize kernel initialization.

| Case | ENTRY to first probe, cb59 -> stream | FIRST_READY to RECV_DONE, cb59 -> stream |
|---|---:|---:|
| Normal B8 | 17.0285 -> 14.1235 | 2.6625 -> 2.8925 |
| Normal B32 | 17.7765 -> 14.5505 | 5.1575 -> 7.1830 |
| Hotspot B32 | 20.1925 -> 16.6630 | 39.6635 -> 44.6240 |

These ON spans are independent rank-round/core maxima, not an additive OFF
decomposition. Starting FIRST_READY earlier can itself lengthen the later
span; that span also includes waiting for subsequent packets. For example,
hotspot B32 sender FIRST_SEND-to-SEND_DONE increases from 39.7220 to 40.4495us,
yet sender ENTRY-to-WORK_DONE is almost unchanged, 48.1945 vs 48.0450us.
Longer relative spans alone do not prove extra execution work.

Normal ON-minus-OFF overhead is 1.8660–2.6793us for cb59 and 1.4769–3.9421us
for stream; most stream small-B captures have greater perturbation. Hotspot
overhead is 1.4636–2.3321us for cb59 and 1.8192–3.6618us for stream. Use OFF
for the gain/regression decision and ON to locate investigation priorities.

Hotspot selects the first 48 experts here, so payload goes to ranks0 and1.
Seventy of the 80 rank-rounds have no receive payload. Their full-R medians
for WORK_DONE or receive preparation do not describe the hot ranks; first
probe/ready spans have ten observations. The activity maximum of28 means
that all R cores can participate on a hot destination, not that the hotspot
necessarily limits work to one or two R cores.

### Next decision

Keep the earlier local-prefix handoff as a successful small-B candidate.
Investigate completion latency at B24/32 and hotspot B16/24/32 before adding
further synchronization. Record ownership changes per-core workload, but the
summary does not measure packet counts or later packet arrival order; it
does not establish load imbalance as the sole cause of the regression.
Use the actual route seed/config to derive per-owner work offline and compare
with same-core ENTRY-to-WORK_DONE observations. If heavy slices explain the
late cores, test bounded q-range splitting or coarse assistance as a separate
candidate while preserving early startup and unique output/cleanup ownership.

Compared only as a historical reference, stream normal B1..16 is now
0.9527–1.3130us above the original fast cb59 measurements; B24 and B32 remain
2.3861 and 3.7514us above them. The original binary/environment has not been
rerun in this A/B, so these residuals cannot be assigned to SDK or page costs.
This section records user-provided target results; the assistant did not
perform a new remote run or change the kernel while analyzing them.
