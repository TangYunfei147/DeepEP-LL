# Small non-dedup parallel prefix candidate

## Current EP16 producer-count experiment (2026-10-25)

Current recorder is schema15/sparse; the older schema13 discussion below is
historical. This experiment targets one physical host with 16 contiguous ranks,
four contiguous G4 groups, and M<=32/B<=32. Each NPU has three fullmesh ports,
four intra-host Clos ports and one inter-host port. The LL MTE API cannot select
those ports explicitly; this change reduces prefix control work, with no claim
of port striping or measured bandwidth improvement.

`DEEPEP_LL_SMALL_B_PREFIX_CORES=P` selects 1..R prefix producers. The default 0
means R and preserves the baseline. Only EP16 small direct calls use a reduced
P; other EP sizes retain R, and dedup retains its original R0 prefix path. The
first P receiving cores consume disjoint count slices and publish the prefix;
all R cores keep their original output-row receive ownership. Nonproducers
advance their own generation through a zero-total record but do not become
ready dependencies. Sender order, q, packet/count layout, combine and the
230KiB UB limit remain unchanged. Host and device use the same P-dependent
budget; no lane count is deliberately reduced.

P is a build constant recorded by native and Python wheel identities, build
caches, all-rank checks and result metadata. `small_b_prefix_cores` is the raw
value; `effective_small_b_prefix_cores` reports P/R only for the small direct
path and is null for other modes. Old results without P remain unknown. Do not
mix different P wheels in one job. Full OFF timing can compare different P;
per-core stage ratios cannot assume the same producer roles across P.

The prior S28/R28 trace places approximately21–23us between PREFIX_LOCAL_DONE
and PREFIX_READY. This includes publication, short DMA, waits, Scalar checks
and prefix loading, not just inter-core waiting. It does not prove that the
whole historical latency gap is caused by parallel prefix. Keep S30/R26 fixed
in the new experiment, with P=26 as its control; do not directly attribute a
comparison against the old S28/R28 run to P.

For E384/R26, a successful one-pass control read uses:

| P | Own-generation bytes | Preceding-total bytes | R×P ready bytes | Total bytes |
|---|---:|---:|---:|---:|
| 26 | 832 | 10400 | 21632 | 32864 |
| 6 | 832 | 480 | 4992 | 6304 |
| 1 | 832 | 0 | 832 | 1664 |

Retries add traffic. All R still load the complete prefix (39936 bytes for
E384/R26); each producer scans more records as P falls. These are protocol
operation/byte counts, not measured network bandwidth or latency. P6 gives
64 records per producer for this shape, but the implementation handles general
E and empty slices and does not specialize route IDs, seeds or expert choices.

### Build every variant before any device run

First follow [the mainline SDK transition](../../docs/SHMEM_MASTER_UPDATE.md).
It requires the new SDK even for P26; reuse of the old SDK is rejected.
From the project root, after loading the target's existing environment:

```bash
export DEEPEP_LL_SEND_CORES=30 DEEPEP_LL_RECV_CORES=26
for p in 26 6 1; do
  DEEPEP_LL_SMALL_B_PREFIX_CORES="$p" \
    python tests/ll_stage_timer/diagnose.py --build-only \
      --run-name "prefix_p${p}_build_001" || exit $?
done
```

This finishes all six OFF/sparse wheels and Host-only identity checks before
any sample is launched. It does not query or run a device. Existing output
directories are rejected; use a new shared suffix for a new attempt.

Only on the allocated 16-card target, set the actual MASTER_ADDR, MASTER_PORT
and SHMEM_PORT for this job, then run the candidates serially:

```bash
for p in 26 6 1; do
  DEEPEP_LL_SMALL_B_PREFIX_CORES="$p" \
    python tests/ll_stage_timer/diagnose.py --skip-build \
      --wheel-map "diagnostic-runs/prefix_p${p}_build_001/node_000/wheel-map.json" \
      --run-name "prefix_p${p}_run_001" \
      --bs 1 2 4 8 16 24 32 --m 32 \
      --hidden 3584 --topk 8 --experts 384 --dtype fp8 --cpu-sync 0 \
      --devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
      --num-nodes 1 --node-rank 0 \
      --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
      --shmem-port "$SHMEM_PORT" || exit $?
done
```

Use uniform random top-k and identical seeds across P. The first sweep uses
the ST's existing deterministic seed sequence. Confirm a promising result with
paired independent seeds and alternating candidate order, rather than tuning
for the one initial seed. First send only the three resulting SEND_FIRST.txt
files (three sends total); subsequent requested supplements across this
experiment must keep the total at most ten. Do not send CSVs or archives.

Compare full OFF task time first, then active ON/OFF perturbation, the same-core
ENTRY→PREFIX_READY span, first receive probe and receive completion. Sparse
spans can overlap and must not be summed or subtracted across ranks. A shorter
wait alone is insufficient. Early payload prefetch is intentionally a separate
candidate because current receive ownership depends on complete prefix/T.

CPU model validation covers P0/1/4/6/8/26, S30/R26 and default S28/R28,
generation wrap, empty slices, repeated calls and unchanged non-target plans.
Current CANN and device status is in
[BUILD_DISPATCH_SPLIT.md](../../docs/BUILD_DISPATCH_SPLIT.md); old successful
builds below do not qualify the new SDK/P configuration.

## Historical parallel-prefix implementation

Branch `kernel/ll-g4-small-b-prefix`, baseline `kernel/ll-g4-small-b@c68d8c6`.
Active only for M<=32/B<=32. Dedup keeps the original R0 prefix. The new wheel
identity is `g4-small-b-parallel-prefix`; install matching wheels on all ranks.
Core split, packet/count wire layout, sender order, receiver and combine remain
as in the small-B baseline. Timer schema13 is retained.

Each R core receives one contiguous slice of the E incoming count records,
performs the existing integer prefix VF locally, publishes its slice total, adds
the preceding slice totals, and writes its disjoint part of one shared prefix.
All R cores then load that table for their existing receive-task construction.
Empty slices still publish total0 and participate in completion.

Control records use previously unused offsets64/96 of each R core's fixed512B
magic slot, inside the existing zero-initialized36KiB control prefix. Total and
completion records are32B, local-rank only and L2-bypassed. Every small-prefix
invocation advances a private uint32 generation (wrap UINT32_MAX->1). All R cores
advance once, including empty slices; successful same-stream kernel completion
prevents a later invocation overwriting a total before any reader has consumed it.
Dedup calls do not change these generations. R0 still updates the old alternating
epoch after the parallel prefix is ready, preventing stale ready detection on a
later dedup call. No shared flag is cleared by competing consumers.

The implementation relies on the same32B count/control-record publication
semantics as the existing protocol. Prefix writes complete via MTE3_S before
completion publication, and the reused UB publication source is retired first.
Serial fallback drains metadata sources before Receiver resets TPipe.

## Redundancy and cost audit

- Removed a draft all-R total barrier: a producer only needs preceding totals.
  R0 and empty slices skip that total read entirely. R28 nominal one-pass total
  snapshots decrease from25088B to12096B across the rank. Poll retries can add more.
- Reused retired count-reduction scratch for the total snapshot; no new TBuf
  member or separate total-array allocation. Do not overwrite it before V_S.
- One shared prefix table is written once in disjoint slices. No per-consumer
  prefix replicas or R-by-R total broadcasts are created.
- Retained the final full-prefix load because the current receiver builds tasks
  from global output-row boundaries. It is not another prefix computation.
- Retained one own-generation read per R, two control publications per R and a
  final R-record readiness snapshot. With R28 and one successful poll per
  phase, control reads total38080B (896B own-generation +12096B preceding totals
  +25088B final readiness), versus896B for one R0-epoch poll per receiver.
  Additional control writes are1792B per rank. Retries can change both totals. They are new overhead relative to R0-only
  processing and must be included in whole-kernel A/B, not hidden in phase labels.
- Retained prefix MTE3 completion before ready and the compatibility epoch write.
  Removing these would risk source reuse or stale-prefix consumption.
- Sender-side B1 q buffers still have conservative allocation metadata, but the
  dense histogram/scan is skipped. Do not describe reserved bytes as executed work.

With E384/R28, conservative per-R prefix-stage/complete reservations are2176B
and2624B. Old R0 reserved15744B+1760B; other R cores reserved0+1760B. Thus R0's
reservation decreases12704B; other R cores gain3040B. Some actual allocations
are smaller, because only preceding totals are read. Host/device budgets use the
same helpers and remain bounded by230KiB. No new GM allocation is needed.

## Interpretation and validation

PREFIX_COMPLETE includes readiness waiting. This candidate moves part of the
dependency wait into BUILD_RECV_CUMSUM; a smaller PREFIX_COMPLETE alone is not a
speedup. Compare OFF full task time, ON INIT/KERNEL, first receive Probe, plus
BUILD_RECV_CUMSUM/PREFIX_COMPLETE/WAIT_CUMSUM/PREFIX_PUBLISH_WAIT together without
adding nested parent/child metrics. Report per-core distributions and total packets.

For B1/K8 and S>=8, the direct sender owns8 nonempty route shards. If a report
shows7, check whether it lists only long cores and verify sum(TX_PACKETS)==8 for
the source rank/sample; actual S and K are needed before diagnosing a missing send.

Native tests cover actual VF scan with CPU register mocks, adversarial producer
ordering, empty slices, zero counts, varying E, generation wrap and alternating
dedup/small calls, plus UB/control bounds. These tests do not replace target NPU
correctness. Regress M32/B1,2,4,8,16,32 and M256/B1,8,32,256, repeated invocations,
and every input mode. Keep S/R and input routes fixed in A/B. Compile/device
validation status is recorded in docs/BUILD_DISPATCH_SPLIT.md.

## Prior work audit and remaining candidates

The existing small-B sender still allocates q scratch descriptors even when B1
skips q construction, and keeps the conservative two-input-buffer setup. These
are residual setup costs, but the user's current trace reports normal send cost;
they are not combined with this prefix candidate. Prequantized input clearing
also remains until complete valid-byte/padding overwrite is proven.

The plain-q setup forB8/16/32 and the count VF can still have call overhead on
tiny slices. Parallel prefix uses R short VF invocations instead of one VF scanning
all E records: aggregate vector work can rise even when wall-clock critical work
falls. Limiting prefix producers or scalar-prefixing very short slices is a future
A/B variable, not an assumed gain. Current candidate keeps receiver ownership fixed.

Final prefix loads, per-slice total publication, publication completion waits and
legacy epoch maintenance are required by the selected protocol. Removing them
without replacing their dependency would change correctness rather than remove
redundant work. Verify whole OFF latency and first Probe; stage redistribution alone
is insufficient evidence.

Budget audit correction: the parallel stage/complete reservation is applied only
to R cores. S core resource plans retain the baseline values, avoiding an unrelated
small-B sender-lane reduction at large H. A source guard checks this condition.

## Validation result

Final source ae69d20 passes native protocol/UB tests, all7input-mode ON/OFF CANN
compilation and full ON wheel linking. Final compile times OFF37.614s/ON71.513s;
no device run or OFF performance result. See docs/BUILD_DISPATCH_SPLIT.md for
logs, artifact hash and prior corrected test/compiler failures. Use the existing
small-B test matrix and add M256 regressions; compare against c68d8c6 with the same
S/R and OFF build. Matching full wheel and fresh test processes on every rank are
required; the launcher rejects an old small-B wheel via implementation identity.
