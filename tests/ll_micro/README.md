# A5 scalar and transfer isolation experiments

Branch: `experiment/ll-g4-scalar-probes`, based on `197052c` (receiver metadata
optimization). User reports a stable ~1% improvement for that revision, and no
net improvement with packed q `af094ce`. Those are target-machine observations,
not measurements from this harness. Production dispatch is unchanged from
197052c; the new standalone operators never run an incomplete LL protocol.

## Scope and interpretation

This is a primitive-level diagnostic harness, **not an end-to-end dispatch
ablation**. It does not bypass LL receive flags while leaving a sender running.
It cannot decompose a full overlapped dispatch by subtracting independent times.
The next production changes should be chosen using these measurements and
validated with the existing OFF dispatch A/B workflow.

The newly authorized 177.22.26.1/27.1 are Ascend950DT with a different interconnect
from the target. Scalar/vector findings are candidates for transfer; communication
numbers are specific to the measured mapping. Only devices 0/1 are exposed;
scalar tests use device 0. If **any** NPU task is detected, the runner stops and
leaves results intact. It never stops other users' processes.

## Scalar experiment matrix

10 warmups, 100 recorded rounds, rotating mode order; report last 50 separately
for every core. Matrix: 1/56 AIV, 1/32 repeats, count 1/32. Every launch checks a
checksum. Raw ticks time only the operation loop and its final drain. Event time
also includes initialization/checksum generation and must not replace raw ticks.
Conversion on this A5 follows the previously confirmed 1000 ticks/us; retain ticks.

| Mode | Operation | Comparison |
|---|---|---|
| 0 | Empty loop + final barrier | Instrumentation/loop floor |
| 1 | Three Scalar Get/Set per record | Old destination snapshot |
| 2 | Vector copy four words per record + S/V dependencies | Single small call versus batched records |
| 3 | Production plain q builder for 72 routes, seeded histogram | Current q setup |
| 4 | Same histogram/update with expert/q packing | Extra setup cost of packed q |
| 5 | Scalar pending index stores | Per-row initialization |
| 6 | Signed CreateVecIndex + dependencies | Vector replacement |
| 7 | Move full B256/K8 route IDs (8192B) from GM each repeat | Route-input transfer cost |
| 8 | Same deterministic IDs already in UB | Constant/preloaded top-k IDs; not free dynamic routing |
| 9 | Reference existing header; no destination copy | Current receiver optimization |

Mode 3/4 histogram reset/dependencies are included equally. They measure setup,
not the entire FillTail saving or first-packet effects. Mode 8 preloads constants
outside the measured loop, so it deliberately excludes initialization; event time
still contains it. No claim of a full routing algorithm elimination is valid.
Mode 9 still consumes the header in the checksum outside the measured loop; in
production the existing metadata MTE3 output and type snapshot remain necessary.

## Transfer experiment matrix

`peer_probe bytes cores lanes chunk mode warmup samples direction`

| Mode | Input GM to UB | UB to output GM | Meaning |
|---|---|---|---|
| 0 | No | No | Empty launch floor |
| 1 | No | No | UB initialization/control floor |
| 2 | No | Yes | Preloaded deterministic UB, pure write path |
| 3 | Yes | Yes | Load then write with bounded lane reuse |
| 4 | Yes | No | Load-only, no receive waiting or payload publication |

Directions: `tx` local source / imported peer destination; `rx` imported peer
source / local destination; `relay` imported peer source / a **disjoint region on
the same peer** destination. Relay is a two-device read/write loopback, **not** a
three-rank or cross-machine forwarding measurement. For a real first-hop relay
in LL, separately use the production RELAY_ISSUE/complete diagnostic and the
target topology. This harness does not validate multi-host pointer mappings.

Full matrix: 4KiB/128KiB/1MiB total per launch, 1/28/56 AIV, 1/4 lanes, 4096B
chunks. Total bytes are partitioned across cores, not per-core bytes. Mode2 is
only part of tx matrix. Mode3 covers all directions; mode4 covers tx/rx. No
packet flags, routing fanout, FP8 scale conversion or receiver polling is added.
Thus it establishes primitive ceilings/floors, not LL effective wire bandwidth.

Mode2/3 verify all payload bytes and guards before/after sampling. Mode0/1/4
verify the destination remains unchanged; mode4 does not export a payload and
does not independently validate every UB-loaded byte. DataCopy and dependency
waits remain real operations. Compiler disassembly is needed for instruction
counts. End-of-kernel drains are included in event timing. Payload GB/s is
bytes/event_us/1000; relay read+write traffic is twice that payload amount.
No Exp/scalar-all_reduce conditioning: these are warm microbenchmarks, not the
formal production profiling protocol.

## Build and run on target

Obtain this branch through Git. Activate that machine's existing Python/CANN
environment (on the newly authorized A5, initialize conda then source
`/home/tyf/0env.sh`). No SDK/submodule/wheel installation is needed for these
standalone ACL executables; production LL build remains unchanged.

From the checkout root:

```bash
export PROBE_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1
export ASCEND_RT_VISIBLE_DEVICES=0,1 ASCEND_VISIBLE_DEVICES=0,1
export TMPDIR="$PWD/.tmp" TMP="$PWD/.tmp" TEMP="$PWD/.tmp"
export SHMEM_LOG_PATH="$PWD/.logs/shmem"
export ASCEND_WORK_PATH="$PWD/.logs/ascend" ASCEND_PROCESS_LOG_PATH="$PWD/.logs/ascend"
mkdir -p "$TMPDIR" "$SHMEM_LOG_PATH" "$ASCEND_WORK_PATH"
bash tests/ll_micro/build.sh
python tests/ll_micro/run.py --suite scalar --run-name scalar-001
python tests/ll_micro/run.py --suite transfer-smoke --run-name transfer-smoke-001
python tests/ll_micro/run.py --suite transfer --run-name transfer-001
```

Use a fresh run name each time; existing results are not overwritten. `--plan`
prints the exact launch matrix without using an NPU. Results live under
`tests/ll_micro/results/<run-name>/`: identity, occupancy snapshots, raw CSV,
stderr and per-core tail50 summary. Keep machine topology with the results.

## Production optimization decisions

1. Retain receiver direct-header use as the measured target baseline.
2. Do not retain packed q merely because its FillTail stage is shorter: compare
   setup cost, q extraction, and OFF total latency; current target evidence is negative.
3. Only batch metadata vector work when a real existing ready batch is available;
   do not delay the first ready packet just to create a batch.
4. A faster vector index microkernel does not establish faster flag polling. Any
   flag reduction/decode change must preserve exact readiness semantics, packet
   markers, lane event ownership and relay-before-output ordering.
