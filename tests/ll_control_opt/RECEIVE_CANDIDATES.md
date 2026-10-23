# Independent receive experiments from868ddf8

The four default entries are baseline868ddf8, source-mask-revert85d05f4,
flag-batch8 1c29599 and flag-prefetch16 c1e4faa. All three candidates directly
parent868ddf8; none contains another candidate or metadata-batch d00e87f.
The delivery branch contains all revisions for a normal Git fetch. Its own
kernel tree is the unchanged868ddf8 baseline. Use the suite to build/run the
candidate revisions; installing the delivery checkout directly tests baseline.

## What changes

- Source-mask revert restores only the sender's original per-FIRST_HOP UB mask
  update. Its sender file is identical to197052c; pending-suffix and flag VF stay.
- Batch8 changes only the maximum number of packets examined in one probe.
  All flags, sums, work and snapshot allocations retain16-packet capacity,
  keeping both overlap and serial fallback lane selection unchanged.
- Prefetch16 preserves the16-packet probe and primes the first batch's payload
  lanes before queuing the next adjacent batch's flag DMA and VF/Sum. It joins
  the second result after consuming the first batch, then consumes the second.
  The two snapshots use stable task IDs and pending remains immutable until
  both are consumed. Removal proceeds from the second batch to the first,
  with descending original indices inside each batch.

Only64extra UB bytes are needed for the second task-ID snapshot. Flags, sums
and reduction work are reused after the first result has reached Scalar.
The planner selects the original payload lanes before checking for spare64B;
insufficient spare space retains the original single-probe path. No lane is
removed and the Host's minimum accepted shape does not change. The checked
B=M256/H3584/K8/E384/EP16 layouts keep6send and6receive lanes, including R0,
with S28/R28 and S30/R26, for all three input layouts.

The split probe owns MTE2_V event6 and V_S event6. Immediate receive dependencies
retain event7; payload-lane event domains and relay/output retirement are
unchanged. Every queued probe is joined before returning to the caller, even
for empty batches. There is no new global barrier or callback with a pending
probe. All seven dispatch translation units and production O2 remain intact.

## Interpretation

Prefetch changes the receive schedule: two adjacent snapshots are chosen before
pending compaction, whereas the baseline chooses its next snapshot afterwards.
FIRST_HOP priority remains inside each16-packet batch. If consuming the first
batch makes a second-batch terminal ready, the earlier flag observation may
still report it pending; a later pass will retry it. The shared MTE2 engine may
also incur more contention. These effects can outweigh overlap; no performance
improvement is assumed.

For optional timer ON diagnosis, RECV_POLL and RX_FLAG_REDUCE on the prefetched
path record submission and residual join as two real instruction fragments,
with two calls per probe. They do not include intervening first-batch payload
work. Do not compare their per-call averages directly with the baseline's
single fragment. RX_PROBED, RX_READY and RX_EMPTY_POLLS retain their packet/probe
meanings. Ordinary suite performance always uses timer OFF.

## Target reproduction

Use the existing target Python/CANN environment. Update once on every machine:

```bash
git fetch origin kernel/ll-g4-control-opt
git switch kernel/ll-g4-control-opt
git merge --ff-only origin/kernel/ll-g4-control-opt

# Compile four exact revisions, reusing validated existing wheels/SDK/native cache.
# No NPU or deployment variables are required for this phase.
python tests/ll_control_opt/suite.py \
  --build-only --resume --send-cores 30
```

Keep the existing NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT, SHMEM_PORT and
ASCEND_RT_VISIBLE_DEVICES settings (exactly16local devices). Once compilation
has finished everywhere, use the same fresh run-prefix on every machine:

```bash
python tests/ll_control_opt/suite.py \
  --run-only --send-cores 30 --rounds 4 --order alternating \
  --devices "$ASCEND_RT_VISIBLE_DEVICES" \
  --num-nodes "$NNODES" --node-rank "$NODE_RANK" \
  --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
  --shmem-port "$SHMEM_PORT" \
  --output-dir "$PWD/results/ll-control-opt" \
  --run-prefix b256-rx868-001
```

Each round tests all four revisions. Odd rounds use the manifest order; even
rounds reverse it. The original ST retains correctness gates,10warmups,
100samples and each rank's last50. Shape is B=M256/H3584/K8/E384/FP8; combine
uses BF16 input/output. The metric is pooled summary.csv mean_kernel_us.
No earlier performance result is overwritten. Use a new prefix after a failed
device run. Add --plan to inspect the exact commands without building/running.

Node0 comparison.csv is under results/ll-control-opt/b256-rx868-001/.
reference_delta_us/pct and baseline_delta_us/pct compare each candidate with
868ddf8 in the same round. parent_delta is empty for these independent candidates.
Raw profiler outputs and per-revision/per-round summaries remain available.

## Local validation,2026-10-23

The user deferred SSH/CANN validation because27was unavailable. No remote
connection or NPU kernel ran for these candidates. Local CPU checks passed:

- Source-mask revert:45required source/Host selector/ST checks.
- Batch8:51required checks plus overlap layout/progress coverage.
- Prefetch:52checks, including actual C++ pair scheduling and production pending
  table with delayed arrivals, partial/empty batches, once-only consumption,
  FIRST_HOP-dependent terminals, S28/S30 layouts and spare64B boundaries.
- Suite/dependency preparation:58checks including real local Git sibling
  checkouts, dirty-worktree/SDK/wheel guards, explicit comparisons, alternating
  execution and loopback readiness handshake.

These batches overlap and are not a combined test count. Local C++ compilation
uses a CPU mock/Host selector; it does not validate Bisheng call domains, actual
DMA events, CANN wheel linkage, device correctness or speed. The target build-only
command above must succeed before performance runs. Earlier CANN results for
d00e87f do not establish compilation of these new candidates.
