# Dispatch ablation audit, 2026-10-22

The existing runs record the performance of particular experimental programs.
They do not decompose an 88/93us production dispatch into additive scalar,
memory-copy and communication costs. The receiver report's flag/output deltas
are hypotheses for further work, not estimates of production savings. This
audit updates documentation and offline analysis/export only; no kernel,
communication protocol or measurement sequence is changed.

## What each mode measures

| Mode | Measured program | Permitted interpretation | Not established |
|---|---|---|---|
|0 baseline|Complete OFF dispatch with identifying payload|Reference latency under this run's routes, build and sampling conditions|Equivalence to a separately quoted88us result without matching its identity/configuration|
|1 fixed-payload|Same complete dispatch with zero payload/unity scales|Sensitivity to this input-content change|Transport bandwidth or isolated memory-copy time|
|2 preload|Constant UB input replaces payload/scale GM input; added initialization remains timed|Net effect of this replacement program|Original input cost or a bound on better input-path implementations|
|3 prepack|Initialized output lanes replace payload packing; tail construction moves after lane retirement|Net effect including initialization and changed overlap|Isolated packing cost or proof that payload optimization cannot help|
|4 ready-replay|Retained tasks/packets, ready receive, no duplicate relay|Conditional receive processing duration for that version|Live receive, cold receive, preparation, relay or network cost|
|5 ready-breakdown|Three ready-replay arms inside one kernel, with restore/check work between arms|Observed contrasts under the recorded sequence and state|Production flag/output savings or a decomposition of mode4's old40us|

The original OFF suite runs modes serially in fixed order. Its last 50 samples
measure within-run variation; they are not repeated interleaved baseline/candidate
trials and do not control all between-run drift. Retain the original records.

## Preload and prepack change more than the named operation

[dispatch_server_dedup_send.h](../../csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h)
`IssueInput` initializes scale packs by scalar writes on the first use of each
input lane. At H3584 with int32 scales that is 28 writes per input lane.
`InitAblationPackets` adds output-lane clearing and 28 scalar scale writes per lane.
These execute within the measured kernel.

`ConsumePacket` normally constructs `FillTail` before waiting for the selected
output lane. Mode3 constructs it in that output lane after the wait, with another
scalar/vector dependency. It still computes real tail/q metadata. The replacement
therefore changes scheduling and overlap as well as removing Copy instructions.
A flat or negative full-kernel delta does not show that the removed work is free.
The earlier suggestion that a small fixed-payload-to-prepack delta rules out
payload-path contributions to an 18us target is not supported.

None of these modes separately controls TX/RX/relay bytes or sweeps packet sizes
with a matching completion boundary. Bandwidth and fixed communication startup
cost remain unmeasured by this suite.

## Receiver replay is conditioned differently from live receive

[dispatch_variant_kernel.h](../../csrc/kernels/dispatch/low_latency/dispatch_variant_kernel.h)
executes real dispatch before entering replay. Replay retains prefix/task tables,
UB allocation and the completed input windows. It resets pending indices but
omits task preparation, live arrival and duplicate relay. It still performs
payload loads, metadata work, flag cleanup and final drain. All 56 cores participate
in ready receive rather than progressing through the production send/receive
schedule. Its duration must not be subtracted from complete dispatch.

In mode5, the first arm follows real dispatch. Each subsequent arm follows
`ValidateReplayOutput`, which reads every retained packet and output and compares
payload/scale words using Scalar operations. Flag restoration adds writes before
every arm. Excluding those operations from the timer does not exclude their
effects on caches and pipeline state. Local `SyncAll` does not establish a shared
cross-rank quiet boundary.

Only ABC/BCA/CAB orders are rotated. Each arm occupies every position equally,
but the predecessor relationships are fixed. A position-balanced mean cancels
an additive position effect only under that model; it does not generally cancel
arm-by-position or predecessor interactions. The observed large first-position
penalty therefore prevents treating the existing4.6us/2.2us aggregate deltas as
isolated flag/output costs. Per-rank medians do not remove this issue.

The original full-dispatch statistic is `max_rank(mean_kernel_duration)`. The
original mode5 headline statistic is `mean_iteration(max_rank_core(duration))`.
These are different operations. Moreover, arm maxima may select different cores
or ranks. Preserve both definitions explicitly, and inspect matched core/rank
and execution-position distributions before summarizing an observed contrast.

## Correctness evidence and unresolved marker cause

The failure with four FP8 elements at480..483 and the reported errors7.0/1.46875
is reproduced by writing float1.0's four bytes over compact payload. The guard
fix suppresses both relay preparation and relay issue during replay; removing
that unsafe write is justified. It does not explain how the earlier replay
combined a FIRST_HOP header with a compact UB view. That device path remains
unresolved. Passing subsequent output checks does not retroactively establish
the full cause.

[dispatch_server_dedup_recv.h](../../csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_recv.h)
records the prefill layout in the task-position high bit. Baseline/no-output
decode restored flags, whereas skip-flags uses that saved bit. Current counters
verify probe batches and output byte totals, not the equality of layouts or
packet visitation order. A KeepFlags/Compact mismatch can still yield correct
payload while changing priority, full-versus-compacted DMA and output-copy count.
Thus existing payload checks cannot prove that flag processing was the only
effective difference between arms.

Current verification has additional limits:

- Per-arm checks compare payload and scale with retained packet data using the
  same task-to-slot mapping. They do not independently verify metadata or mapping.
- Host verification runs after all three arms. Replay may overwrite an earlier
  result before that check; initial prefill output is not independently checked.
- No-output retains payload from prefill or a preceding arm. Its passing check
  shows that retained data still agrees with packets; it is not an independent
  output calculation or a sentinel-based proof that no output write occurred.
- Constant-payload modes cannot identify payload rows by content. Cross-mode
  metadata fingerprints strengthen routing checks but do not restore the lost
  payload identity test.

Before a redesigned performance experiment, record prefill and per-arm counts
for application role crossed with KeepFlags/Compact, and an ordered digest of
task/slot/layout visits. Capture the first contradictory header/layout with its
task and slot identity. Compare per-arm metadata with an independent CPU reference
or a separately validated prefill snapshot. Use per-arm sentinel/generation
patterns to detect missed writes and output mutation, and vary scale patterns in
correctness-only tests. These checks must execute outside measured intervals and
be planned so they do not make one arm's preparation different from another's.

## Effective route seed in historical captures

[test_ep_low_latency.py](../elastic/test_ep_low_latency.py) regenerates routes for
each correctness repeat and leaves the last routes in the sampling closure.
Consequently, with checking enabled, measured routes use
`configured_seed + correctness_repeats - 1`. The default is seed0+3-1=2.
Historical capture metadata still labels the configured seed0.

This update preserves the original seed field and workload, and adds
requested_seed, actual_route_seed and route_digest to future captures and run
results. Offline reports show the recorded effective seed where available;
legacy captures without it remain unknown rather than silently assigning seed2.
Route digests, where present, provide a stronger reproduction check. This labelling problem does not by itself
invalidate comparisons that used the same actual routes, but it matters for
reproduction and comparisons against other runs.

## What can be done with the existing data

Without a rebuild or new device run, the raw schema15 captures allow:

- Checking actual recorded order across cores/ranks, capture completeness,
  frequency, counters and existing per-arm verification results.
- Showing arm-by-execution-position distributions, matched same-core/rank
  contrasts and the core/rank that owns each maximum. Keep original aggregate
  statistics for traceability instead of replacing them silently.
- Distinguishing requested seed from a recorded effective seed (unknown in old captures) and retaining
  wheel/source/configuration identities.
- Exporting bounded JSON parts for review without requiring the large CSV.

```bash
python3 tests/ll_ablation/receiver_export.py \
  .ablation/results/recv40-002-ready-breakdown \
  --output-dir .ablation/results/recv40-002-export
```

The output directory must be new. Each JSON part contains only ASCII and is at
most 60000 bytes/characters. Send the parts separately with their identifiers.
The updated receiver suite invokes reporting and compressed raw-trace text export
automatically; see [TRACE_TEXT.md](TRACE_TEXT.md). This JSON export remains an
explicit alternative.
Offline analysis preserves input captures and requires no wheel installation or
NPU launch. It cannot reconstruct missing layout/order hashes, validate past
prefill output independently, or remove state contamination already present.

## Controlled follow-up, not implemented in this audit

1. Establish the layout/marker invariants and correctness checks above. Add A/A
   controls that execute identical work under different labels to measure the
   harness's position/state effect before interpreting A/B differences.
2. Measure one selected arm per invocation with identical prefill, preparation,
   synchronization and cache-conditioning rules for every arm. Separate
   correctness validation from performance sampling. Balance/randomize arm order
   across repeated launches and repeat A/A controls. A one-arm design alone does
   not prove identical cross-rank arrival or cache state; verify the chosen rules.
3. Keep a separate OFF full-dispatch baseline/candidate comparison. Run repeated
   interleaved trials with matching real routes and report distribution and run
   variability. Only this layer establishes production latency improvement.
4. If bandwidth/static cost is still the question, use separate TX/RX/relay
   primitives with actual topology and protocol dependencies, exact byte/packet
   counters and transfer-completion timing. Sweep packet counts and payload sizes;
   keep those fitted costs separate from overlapped production critical paths.

Steps1–4 need new implementation and target measurements. The present audit does
not claim those experiments have been built, compiled or run.

## Instrumentation decision and validation of this update

Schema15 is sufficient for offline order/position, maximum-owner and same-core
duration analysis. It does not record application-role by layout counts, an
ordered consumption digest, or guaranteed restoration/synchronization/validation
intervals. Its192 trace records can fill during the initial dispatch, so missing
replay timestamps must remain unknown. New correctness diagnostics should record
layout/role anomalies and ordered task/slot/layout digests; a separate diagnostic
capture should expose restoration, local barrier, verification and replay start
relative to the same core's prefill completion. Do not subtract clocks across
ranks or include per-packet debug hashing in the performance comparison.

Adding all these fields needs an explicitly versioned layout/capacity change and
new captures. This update does not alter the kernel, schema15 or installed wheel.
It adds report v2, bounded transfer files and truthful future route metadata.
CPU validation:104passed,2 local C++ compiler skips. A complete CLI export from
synthetic raw captures in paths containing spaces produced6parts, largest59088
bytes, and passed package verification. Full56*16*9 core values, Unicode/large
record fragmentation, corruption/missing-part detection and size limits are
covered by tests. These are parser/transport checks, not new NPU measurements.
