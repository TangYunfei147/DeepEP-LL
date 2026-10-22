# M<=32 / B<=32 direct dispatch candidate

Branch `kernel/ll-g4-small-b`, based on197052c (plain q table, receiver-header
optimization). Activation requires BOTH `!enableDedup` (M<=32) and actual B<=32.
B<=32 with M>32 follows the existing dedup path. B>32 non-dedup is unchanged.

Sender candidate:

- Return before setup for a zero-route shard. With K>=1, an empty route shard
  cannot own source-token weights; count publication remains in Counts, before
  sender Process. No send lane/event was created, so Retire has no lanes to wait.
- B1: send q=0 directly, without VF histogram or q-table construction. The legacy
  convention sets q=0 for every route index<K, including repeated expert IDs.
- B2/4: build only the owned q entries from their short original Scalar prefixes.
  Preserve full uint32 q and first-token behavior. B8/16/32 keep the VF seed path.
- Initialize/use no more send lanes than owned packets. Existing overlap pool
  budgets remain conservative; lane event IDs and per-core packet order are kept.

No input-buffer clearing was removed; quantization padding remains protected.
S/R, count participation, R0 prefix, wire flags, receive/output order and combine
are unchanged. No new mode/template dimension or kernel launch was added.
Implementation identity is `g4-small-b`, core protocol19000+S and timer schema13.
Install the matching full wheel on every rank; a new SDK rebuild is not required
solely for this algorithm change.

CPU checks execute the production tiny-q helper against an independent prefix
reference, with repeated experts, split token shards, all target smallB values,
K1/3/8/32, and multiple S counts. Static digest guards preserve dedup send/q/tail
methods, the receiver, main dispatcher, overlap plans and core configuration.
These are source/CPU checks, not proof of identical dedup machine code or speed.

Target A/B: start from197052c, then sender-only commit, then count candidate.
Use B1/2/4/8/16/32 with M32 and fixed EP/H/K/E/dtype/S/R. Also regress B1/8/32
with M256 and B256/M256 to cover dedup. Compare OFF profiler kernel time; capture
ON stages separately (INIT is outside KERNEL). Include all supported input modes
in correctness checks. Do not compare ON with OFF timings or change S/R per case.

Compilation/device results will be recorded in docs/BUILD_DISPATCH_SPLIT.md.
