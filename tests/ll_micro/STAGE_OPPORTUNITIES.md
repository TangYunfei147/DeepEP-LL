# Stage optimization audit

Production reference: 197052c, G4/plain q table/receiver header reuse. The packed
q variant is not the baseline: user observed that it cancels the ~1% receiver
gain. Investigation branch is experiment/ll-g4-scalar-probes. Findings below
separate production source reasoning, single-sample profiling and microbenchmarks.

## Evidence boundary

communication_costs.csv covers only rank0/iteration0/all56cores. Approximate S30/R26
role split is inferred from stage activity. S/R cumulative times below are medians
across cores in this one sample, not medians across runs. Nested stages must not
be summed; observed submission/wait intervals do not identify a physical link
bottleneck. New local microbenchmarks do not establish OFF dispatch speedups.

26 remained busy at this audit. 27 was idle and was used only for device0 isolated
tests after a fresh check. No other process was stopped. Both have a different
interconnect from the target. No production synchronization or protocol changed.

## Ranked opportunities

| Stage | Existing evidence | Candidate | Priority and constraint |
|---|---|---|---|
| SEND_SETUP / Q_LOCAL_UPDATE | S8.52us total setup; q Scalar2.06us | Keep VF seed + plain q; avoid packed-q setup; identify allocations/weight loading separately | High for diagnosis, no new global table |
| Q_SEED_WAIT | 0.026us | Keep existing dependency | Low: removing required V_S is unsafe and tiny |
| ROUTE_CLASSIFY / source masks | S3.50us classify; source mask Get/Set per first hop lies outside this substage | Accumulate source mask in a Scalar register and store once per token | First low-risk production candidate |
| DIRECT_PLAN | S2.12us; writes token/k/expert/q, reads all later | Keep token/k/expert descriptor, fetch q only at actual DIRECT scheduling | Second low-risk candidate; retain original route index |
| TOKEN_METADATA | S6.60us,290ns/call vs local92ns/call | Remove redundant reads using already-live values; do not build another packed q table; consider a fused pack VF later | High; preserve exact relay entries, weights and firstK |
| TOKEN_PACK | S4.42us,~194ns/call | Fuse the two copies and marker update into one VF | Medium/high; benchmark instruction issue, not merely API count |
| TOKEN_INPUT_WAIT | S0.485us | Retain one-packet lookahead; test same-token reuse only with explicit lifetime proof | Low first; extra buffers could reduce receive lanes |
| TOKEN_SEND_ISSUE | S0.625us,~27ns/call | Keep original packet issue; use primitive matrix to distinguish bandwidth from issue overhead | Low for Scalar optimization |
| PREFIX_COMPLETE / output counts | R0 extracts24 expert totals with Get/Set | One strided gather, optional adjacent difference, int64 expansion | Medium/low; R0 not longest in sample, avoid extra per-element VF |
| BUILD_EXPERT_TASKS | Already vector pending initialization and vector aligned token segments | Retain long-segment vectorization, study short-record distribution before adding calls | Medium; don't reimplement the already-applied index optimization |
| RX_FLAG_ISSUE | R3.23us | Reuse local base/stride; examine batch snapshot copy or gather only if Scalar address processing is proven expensive | Medium; no speculative packet loads before readiness |
| RX_FLAG_REDUCE + DECODE | R2.52us +2.25us; per-batch work | Fused VF reduction + exact readiness/layout comparison + two bitmasks | Highest receiver algorithm candidate; larger correctness surface |
| RX_METADATA | R4.95us | Keep direct-header reuse; reuse role/sourceRank Scalar values in relay path | High for removing redundancies; tiny standalone copy VF offers little |
| RX_PAYLOAD_OUTPUT / RX_META_OUTPUT | R3.93us/2.11us | Batch only naturally consecutive ready output rows; avoid filling batches by waiting | Medium; row order is fixed but consumption is readiness/relay ordered |
| RX_PENDING_UPDATE / priority | Scalar descending scan and swap-remove | Enumerate only set bits in descending order | Medium/low; preserve snapshot indexing and relay priority |
| RELAY_ISSUE | Source/fanout dependent | Reuse sourceRank/peer base and precompute invariant strides | Medium; preserve relay-before-local-output and all lane retirement |
| CLEAR_LOCAL_FLAGS | ~0.91us | Coalesce only truly contiguous owned slot ranges | Low/medium; must clear every original slot, never the emptied pending list |
| Final lane/complete waits | Typically tens of ns in this sample | Keep dependencies; do not add drains to improve attribution | Low; completion may already be hidden earlier |

## Two small Scalar candidates: source proof

### Source mask register accumulation

`dispatch_server_dedup_send.h::SendWithDedup` currently stores
`directMask|localMask`, then for each FIRST_HOP reloads sourceMasks[token], ORs
one firstK bit, and stores again. `sourceMasks_` has no intervening consumer:
`Process` publishes the whole mask after SendWithDedup and EndInputPrefetch.
FillTail reads routeWeights, not sourceMasks. A local register accumulation with
one final store therefore preserves the final mask and packet issue order.

For f first hops per token, source-level access reduction is f Scalar reads and
f Scalar writes; it needs no extra UB, synchronization or communication. Measure
the whole SEND_TOKENS/overall OFF time: source mask updates are not all inside
ROUTE_CLASSIFY, so a flat classify metric does not disprove a gain.

### DIRECT q read at use

The plain q table is built once and remains immutable through all sends. The
current DIRECT plan reads q, stores it to the fourth descriptor word, then reads
it back during group-ordered emission. Instead keep token/k/expert and read
`qTable[(start+token)*k+kIndex-qShard.begin]` at actual emission. The original q
load moves later; one descriptor store and one descriptor load disappear.

Keep the existing four-word stride/allocation in the first A/B candidate to avoid
mixing Scalar savings with UB-budget/lane changes. Keep group rotation, per-group
stable order, deferred input descriptor and q first-token convention unchanged.
Because qTable lookup shifts into scheduling, pipeline overlap may change: the
isolated access saving is not proof of end-to-end gain. Check zero tasks, split
tokens in non-dedup, repeated expert IDs, large q and all supported dtypes.

## Receiver VF fusion: proposed, not implemented

Probe currently loads all eight marker words per packet block, Sum reduces each
row, then Scalar loads up to16 sums and constructs ready/keepFlags masks. A fused
VF could directly return the two bitmasks, eliminating the intermediate Scalar
GetValue/comparison loop and potentially Sum's scratch traffic.

Preserve exact conditions: compact sum == blocks*8; keepFlags sum == blocks*8+.5;
all other sums, including partially arrived packets, remain pending. Inspect all
marker words, not only the first block. Mask inactive rows, preserve layout hints
and MTE2->V->S dependencies. RX_READY must still count ready packets, not batches.
Do not replace marker semantics with a test of a single tail word.

Required validation before A/B: zero/all/mixed ready masks, partial block arrival,
Compact/KeepFlags mix, count1..16, every supported block count, NaN/unexpected
values and padded rows; compare against the original Sum+Decode. Use actual A5
compiler and runtime, then full dispatch correctness. Changes to probe execution
must not reorder the existing FIRST_HOP priority or descending pending removal.

## Order of experiments

1. Measure mask accumulation and deferred q access in isolation; retain checksum
   validation and runtime input bits to avoid an unrealistically constant mask.
2. If promising, create separate production commits for mask and DIRECT q. Run
   target OFF A/B at197052c, keeping ON data only as supporting diagnosis.
3. Implement receiver reduce/decode fusion in a separate candidate, with exhaustive
   protocol tests and A5 compiler checks before comparing performance.
4. Only then consider fused pack/metadata VF or batched output, which have broader
   lane lifetime and latency implications.

Do not change S/R, send order, receive ownership or packet format in the same
experiment. Do not sum microkernel savings across stages as predicted full-kernel
speedup. Sampling, core counts, datatype and code revision must stay matched.
