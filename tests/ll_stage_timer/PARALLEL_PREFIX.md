# Small non-dedup parallel prefix candidate

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
  final R-record readiness snapshot. They are new overhead relative to R0-only
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
