# Single-host shared-payload quick prototype

This is the first functional slice of the proposed larger single-host design.
It replaces full-packet relay writes with references to immutable first-hop GM
payload, while keeping the existing routing, counts, source packets, final output
layout and dispatch/combine handle. It does not yet cache input tokens, batch
notifications or change S/R partitioning. No performance win is assumed.

The delivery branch defaults to mode0 (original transport). The manifest freezes:

| Label | Source | Protocol | Change |
|---|---|---|---|
| baseline | 9027275 | 19000+S | Previous retained measured baseline |
| fence-control | 0c3c150 | 23000+S | Prototype code, full relay packets, final lifetime fence |
| shared-pull | 762b5c6 | 24000+S | Same fence; relay references and destination-side payload reads |

All three use full timer-OFF wheels, the same locked SDK and HT OFF. Prototype
mode is recorded in build metadata and verified against actual native imports;
mode/protocol and all other capabilities must agree across every rank. Mismatched
Host/device protocols are rejected. Historical diagnosis commands stay pinned to
their own original sources.

## Scope

The prototype Host and ST reject configurations outside one host with 16 local
devices, EP16, B=M256, H3584, K8, E384, logical G4 and prequantized FP8 input.
Use S30/R26 as in the previous experiments. Each core must retain the Async
overlap layout. Plain/quantizing and other shapes are not prototype claims.
Stage capture is explicitly rejected: schema17 byte/rate reports do not describe
the new transport. Use complete OFF profiler tasks for this experiment.

When a FIRST_HOP is consumed, its original GM packet remains unchanged. Instead
of copying its full 4096B wire packet to each remaining expert's terminal slot,
the owner publishes a 32B descriptor there: original source/token/k, a reference
tag, owner rank and 64-bit byte offset. The descriptor's completion is confirmed
before publishing the existing per-block ready flags. With eight blocks, each
reference writes 32+8*32=288 bytes, including flags; this is not the total device
traffic. Notifications are deliberately unbatched in this first prototype.

For compact ready packets, the final receiver first loads the 32B header. Normal
DIRECT packets use their existing body. A reference redirects the compact payload
read to the first-hop owner's GM slot. Owner must be in the same G4 and the slot
must be aligned and within the current dispatch window. After loading, the
receiver restores the ordinary Terminal output role and output-slot metadata.
The existing combine can then exercise the unchanged handle contract.

The reference reuses the lane's existing 64B metadata scratch. It does not keep a
pointer into transient UB or overwrite the original shared GM packet. Original
first-hop flags may be cleared, since reference consumers read only the retained
body. The packet body must remain immutable until all reference reads complete.

## Lifetime and deliberate costs

After all work is drained, every local AIV joins SyncAll. Core0 increments a
32-bit completion counter in the unused tail of magic slot71, publishes it with
MTE3 completion, and reads every rank's counter with L2 bypass until all have
reached this invocation. Counters are monotonic, not alternating parity; a faster
rank's newer value is acceptable. Overflow is explicitly rejected. No other rank
writes a producer's counter. New suite processes/Session allocations initialize
the communication memory to zero.

The local join ensures all local payload reads have completed before publication;
the rank join prevents a following dispatch/combine from overwriting GM still
being read. The selected counter lies below the 36KiB control prefix, beyond the
56 active per-core slots and separate from count/prefix/Combine controls. The
extra 32B fence scratch is allocated only after the old work has drained and the
pipe is reset; the active send/receive UB and lane budgets are unchanged.

The fence-control variant pays this same lifetime fence with conventional relay
packets. The shared-pull variant additionally pays header inspection for compact
packets and two conservative MTE3_S joins per reference publication. It issues
separate 32B flag writes from one immutable source block, rather than incorrectly
using a contiguous multi-block source. These costs can overwhelm saved bytes;
they are intentionally visible in the task duration. A regression rejects this
prototype as a speedup, but alone does not disprove a more complete batched design.
Code generation can also differ between the two variants, so their difference
is not a perfect isolated measurement of remote-read latency.

## Target command

Keep the same 16-device selection and MASTER_ADDR/MASTER_PORT/SHMEM_PORT environment.
From the project root:

```bash
run_shared_pull() {
  git pull --ff-only || return $?
  python tests/ll_control_opt/suite.py \
    --commits tests/ll_control_opt/commits-shared-pull.json \
    --devices "$ASCEND_RT_VISIBLE_DEVICES" \
    --num-nodes 1 --node-rank 0 \
    --resume --send-cores 30 --seeds 0 --rounds 2 \
    --order alternating --run-prefix shared-pull-001 || return $?
  python tests/ll_control_opt/export_report.py \
    results/ll-control-opt/shared-pull-001 || return $?
}
if run_shared_pull; then
  printf '%s\n' 'Complete: send results/ll-control-opt/shared-pull-001/analysis-report.txt'
else
  printf '%s\n' 'Stopped: terminal, logs and existing results preserved.'
fi
```

This quick pass builds three complete wheels before starting devices (verified
existing baseline artifacts can be reused), then runs six complete ST calls: one
requested seed0, actual measured routing seed2, two alternating sweeps, all three
versions. Every call retains correctness checks, 10 warmups and 100 collected
tasks with per-rank tail50. Exp+scalar all_reduce precedes measured iterations.
The normal dispatch and BF16 combine results are both collected. All-rank loaded
library/SDK hashes and idle checks remain in force; no process is killed.

Append `--plan` to the suite command to preview without builds/devices, omitting
the exporter. A fuller pass can use `--seeds 0 17 101` and a fresh run prefix.
`--resume` reuses verified wheels only. Existing/partial result directories are
not overwritten; retry with a new run prefix. No command exits the interactive
shell. Logs/cache/raw data remain under the project.

Send only `results/ll-control-opt/shared-pull-001/analysis-report.txt`. The existing
exporter preserves every comparison row and rejects files above 65535 UTF-8
bytes. This quick pass has 12 rows and one return file. The shared-pull reference
delta is against fence-control; its absolute mean must also beat baseline before
calling it an end-to-end improvement. Compare dispatch and combine separately.

## Checks and limitations

Local checks compile the actual descriptor publisher, payload loader, Terminal
metadata helper and completion-fence bodies against CPU memory/event mocks with
assertions enabled. Tests check byte-exact payload redirection, owner/group/bounds
rejection, 32B scratch restoration, flag publication addresses, reads after flag
cleanup, monotonic epochs/overflow and delayed-rank lifetime schedules. Existing
selector/layout/ST/SDK/report regressions are retained. Unmodified business
sources retain the old token hashes; receiver/kernel/protocol are intentional
changes and no longer claim business equivalence with the old receiver.

These checks do not establish Bisheng compilation, actual MTE ordering, device
correctness or performance. No remote compilation or NPU execution was performed.
Target build or correctness failures must be investigated before reading timings.
