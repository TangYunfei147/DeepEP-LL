# Small-B dispatch audit (M=32)

User confirms M=32 for B=1/2/4/8/16/32. Current source af094ce; production
reference197052c differs only by packed q. Comparison available locally is
upstream/main07af8ead, not a confirmed user master SHA. Do not label this as an
exact master regression bisect until the user supplies that revision.

EpServerDedupEnabled uses M>32, so all six shapes are non-dedup. S sends all
routes, then retires and returns; R counts/prefixes/receives. No FIRST_HOP/local
sender task generation/56-core shared reception occurs. G4 grouping still affects
count-range partitioning, but not deduplicated packet forwarding in this mode.
No new device run was made for this audit; existing B256 timings cannot be used
as measured small-B costs.

## Findings

1. main07af8ead uses all64 cores for counting when B<512. Current code always
   uses only R cores. E384 gives6 experts/core on old64; current defaultR28 gives
   13..15 experts/core (local/remote ranges independently partitioned), or14..16
   atR26. Total work is not necessarily larger, but per-producer critical work
   and time-to-count-ready can increase. S no longer competes for count work,
   which helps large-B overlap but is not automatically a small-B win.
2. B1/2/4,K8 have fewer than64 routes and CountByExpert runs per expert with
   Duplicate/Sub/Abs/Mins/ReduceSum and V_S synchronization. With fewer count
   producers this is a particularly plausible regression. AtB8/16/32 the current
   path switches to range filtering+VF histogram; a producer can process separate
   local/remote ranges, adding fixed gather/VF/synchronization cost even though
   non-dedup does not need local-send descriptors.
3. Old sender q lookup returns0 immediately for index<K. AtB1 every q is zero:
   the old q arithmetic cost is effectively absent. Current nonempty route shards
   still seed an E384 histogram, synchronize, and build a q table. B2..32 provide
   very few packet uses per core to amortize setup. Packed q adds setup/extraction
   with no relay FillTail saving in non-dedup mode. Retain plain q as baseline;
   a dedicated B1 all-zero q fast path is the clearest first q candidate.
4. Sender Process skips only enableDedup && tokenCount==0. In non-dedup mode even
   route-empty S cores initialize input buffers, route GM copies, packet lanes,
   flags and events. This issue also exists in the old sender, so it is a shared
   optimization opportunity, not by itself evidence explaining the regression.
   Current additional histogram storage and overlap initialization add context.
5. For prequantized FP8 the current sender clears both input buffers in setup;
   old07af8ead only did so for NeedQuantize. This is a concrete new fixed cost.
   Prove valid payload/scales/tail/marker coverage before removing any clearing.
6. Current R0 polls allE count records and creates one shared prefix; old receivers
   partition count reception/prefix computation. AtE384 R0 snapshots12KiB of32B
   count records per pass. Removing cross-core prefix synchronization was intended
   to help, but critical R0 work is now concentrated. This tradeoff needs a matched
   small-B stage capture; B256 evidence cannot establish the winner.
7. Current fixed S+R=56 vs old64 and defaultS28 vs oldS32 changes task granularity.
   For K8, max routes per S core are below. These are deterministic counts, not
   a latency model. Actual compiled S/R must be recorded.

| B | total routes | old S32 max | current S28 max | current route-empty S cores |
|---|---:|---:|---:|---:|
|1|8|1|1|20|
|2|16|1|1|12|
|4|32|1|2|0|
|8|64|2|3|0|
|16|128|4|5|0|
|32|256|8|10|0|

8. LlMakeOverlapLayout still computes send/receive budgets per AIV and all cores
   initialize counts/sender/receiver views, even though non-dedup S never receives.
   Inactive roles and small workloads deserve an initialization specialization.
   Do not just remove the overlap plan: it controls pool sizes and event ownership.
9. No new global synchronization is introduced by production OFF at kernel exit.
   ON capture performs SyncAll and Flush; never compare currentON to masterOFF.

## Candidate order

- First: M<=32,B1 q=0 fast path; route-empty sender early return based on route
  shard rather than tokenCount; avoid unnecessary prequantized clearing after
  overwrite/lifetime proof. Keep epochs, count publication and metadata owners.
- Second: M<=32 use a contiguous expert count partition and benchmark a single
  small-input histogram rather than per-expert reductions or two range filters.
  Always publish zero-count experts; their readiness is required by receivers.
- Third: compare56-core counting againstR-only counting in a separate candidate.
  This requires updating scratch reservations/UB budgets for S and R and preserving
  count source lifetime. It is not safe to change only the participating-core count.
- Fourth: compare R0 prefix vs a small-B distributed prefix, keeping fixed56-core
  launch. Do not restore old64-core waits or change protocol without Host/device
  validation. Only investigate after the count producer bottleneck is measured.
- Fifth: add a role-specific small-B resource plan and bound lanes by actual
  packet tasks. This needs lifecycle checks; do not shrink a shared send/receive
  layout casually or conflate it with a protocol change.

## Required target evidence

Matched OFF results for each B with M32 and fixed EP/H/K/E/dtype/S/R. Capture
SEND_COUNTS/COUNT_COMPUTE/WAIT_RECV_COUNTS/BUILD_RECV_CUMSUM/PREFIX_COMPLETE,
SEND_SETUP/Q_SEED_SUBMIT/Q_LOCAL_UPDATE/TOKEN_INPUT_WAIT and INIT separately.
INIT is outside the recorded KERNEL interval; whole profiler task time remains
necessary. Record per-core first-send and first-probe, including no-send cores.
Confirm master SHA, whether fp8 means prequantized or in-kernel quantization,
and whether the reported regression is dispatch-only or includes combine.
