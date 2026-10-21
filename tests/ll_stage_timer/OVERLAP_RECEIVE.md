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
