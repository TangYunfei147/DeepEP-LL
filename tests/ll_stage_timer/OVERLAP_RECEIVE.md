# Dispatch overlap receive experiment

Base: kernel/ll-g4-relay-priority@781b1aa. Implementation is delivered in four commits.

## 1. Transition instrumentation

Schema11 retains the 4608-byte/core carrier and all schema10 stage IDs. It adds
SEND_COMPLETE_WAIT (existing sender MTE3_S only), TRANSITION_DRAIN (existing
phase PIPE_ALL only), RECV_PREPARE and SEND_SUBMIT_END (a sampled issue boundary,
not DMA completion). OVERLAP_PATH/SERIAL_FALLBACK/SEND_LANES/RECV_LANES are reserved
counters for the partitioned implementation. No synchronization is added.
The analyzer exports these in normal metric CSV and sampled Chrome intervals;
old schemas7/9/10 remain readable. First two intervals are sampled, so never
construct a complete timeline for repeated steps or subtract nested stage sums.
ON capture requires a matching wheel on every rank. OFF is the performance baseline.

## 2. Persistent partition baseline

UB planner: common/ll_overlap_layout.h, conservative per-core shape bound using
maximum receive rows (including skew), no observed routing specialization.
Count calculation aliases only the sender compute pool before payload issue;
count records/descriptors, prefix production, full prefix acquisition, send and
receive DMA sources occupy disjoint retained ranges. Sender and receiver select
1..6 lanes once. Input MTE2_S uses 6/7; receive data events use 0..5; count
MTE3_V uses 7; immediate receive dependencies use 7. No allocator Reset occurs
on the partitioned path. Serial fallback retains original allocation/reset.
The second commit retains transition drains as a structural A/B baseline.
Protocol 14000+S and implementation g4-overlap-recv require a full matched wheel.

## 3. Receive before sender retirement

S submits remote packets and source metadata, then acquires the prefix and starts
receive without a tail MTE3_S, MTE3_V lane drain or transition PIPE_ALL.
R publishes counts/prefix before local payload and likewise enters receive without
retiring local sends. Original waits remain on SERIAL_FALLBACK cores. Sender
events are consumed at final retirement; all DMA sources persist through final
PIPE_ALL. First use of receive lanes bypasses reuse events; later uses remain
protected. This overlaps preparation/MTE2/Scalar, not independent MTE3 engines.
