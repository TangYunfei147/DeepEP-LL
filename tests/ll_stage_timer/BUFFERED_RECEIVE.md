# Early FIRST_HOP receive during source DIRECT submission

This candidate starts from the c1e4faa receive flag-prefetch implementation.
This independent commit selects `Immediate` in `common/ll_early_receive.h`.
Its only source difference from the Buffered candidate is that policy constant:
selected FIRST_HOPs are consumed and relayed in the callback. The description
below explains the shared mechanism and the Buffered comparison policy.

The `Buffered` policy acts after every two
actually submitted source DIRECT packets, try one 16-entry flag snapshot and
stage at most two ready FIRST_HOP packets into the existing receive lanes.
The sender retains its one-packet input lookahead. No relay, local payload
output or receive metadata output is submitted by the buffered callback.

The FIRST_HOP generator can still own one deferred input at its end. The
cadence therefore records `packets_ + inputPrefetch_.HasPending()` as its target.
`packets_` increments only after the existing packet DataCopy/SetFlag. A callback
runs after Push has taken the next input, and never after the final Flush.
Zero FIRST_HOPs, singleton DIRECTs and empty target groups require no special
rank IDs, expert IDs or shape constants.

## Prefix and event ownership

`TryLoadPrefixOnce` initializes the prefix reader once, reads one epoch and
returns immediately when it is not ready. It never calls the blocking R0 count
poller. A successful epoch read loads the prefix once; Receiver Prepare then
builds the original stable task table once. A later final receive phase skips
that preparation when it has already happened.

The first callback leases one MTE2_V event and one V_S event from TPipe **after
the sender input queue has been initialized**. The lease is retained until
Sender Process returns. Prefix reads and the early flag reduction use that
pair and close every Set/Wait transaction before returning to the sender.
Receiver's early flag-buffer zeroing uses the leased V_S join too. Input
MTE2_S events6/7 are never used for these controls; staged payload events remain
the existing MTE2_S/MTE2_MTE3 lane events0..5. Early callbacks do not change
CTRL[60]: Counts.Init already establishes the source path's mode, prefix copies
are raw integers and flag values are ordinary0/1/1.5 floats. Avoiding a control
transition also avoids changing the interpretation of in-flight V work. These
are finite pipeline dependencies, not PIPE_ALL, send
retirement, cross-core or cross-rank barriers.

TPipe-managed leases, rather than guessed implicit queue event numbers, are
necessary here. The [FetchEventID documentation](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/resource_management/TPipe/FetchEventID.md)
distinguishes a temporary available ID from an occupied AllocEventID lease;
[ReleaseEventID](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/resource_management/TPipe/ReleaseEventID.md)
must pair with allocation. This source-level resource argument does not replace
CANN compilation or hardware event validation.

## Ownership and lifetime

Tasks move `pending -> staged -> consumed`. Only ready `keepFlags` FIRST_HOPs
are selected, in the existing priority order. Their stable IDs enter a small
FIFO; only those selected positions are removed from pending, in descending
original index order. Unready packets and all compact packets remain pending.
The early cursor advances even on unready/compact-only windows. Full lanes stop
all further early probing; no staged lane is recycled while initial TX runs.

The FIFO describes ownership, not DMA completion. At the start of final Run,
ConsumeLoaded waits on each existing payload readiness event and consumes the
staged packets in lane FIFO order. Then ordinary c1e4faa two-snapshot receive
continues. issued/consumed counters are retained. Even if pending is already
empty, staged work is consumed before lane retirement and flag cleanup.

Source masks/weights are published at their original Sender Process tail before
the final FIFO drain. This boundary means this S core has submitted its payload
and source metadata; other cores/ranks can still be sending. There is no new
global send-completion barrier. Relay reads packet headers/member expert/q/weight fields,
not source-mask GM. Source and receiver buffers persist through the existing
final completion. Payload format, q assignment, source packet order, output-row
ownership, original ConsumeLoaded body and combine are unchanged.

## Budget and independent policy comparison

The existing planner chooses its TX/RX lane counts first. It enables the
candidate only for deduplicated S workers on the persistent overlap path and
only when an additional aligned stable-ID FIFO fits. At most six IDs cost32B;
payloads use the original receive lanes. No lane is removed, and total software
allocation remains at most230KiB. The rest of the stated248KiB hardware capacity
is not spent. Serial fallback, R local sends and non-dedup operation retain
their existing schedule. Physical link/port routing is unchanged.

`LlEarlyReceivePolicy::Immediate` is selected in this independent commit:
the same cadence, snapshot, selection and memory budget
call the original ready-packet consumption immediately. Only one policy is
compiled per wheel; this adds no Cartesian product of dispatch modes. The
other candidate selects Buffered. Compare the two policies to
measure the effect of early relay/output MTE3 contention; do not combine their
timings or assume either is faster.

The finite MTE2_V -> V_S join may wait for earlier sender V work and its existing
lane dependencies. The first successful callback also moves prefix loading,
task-table construction and flag initialization earlier. These costs are not
assumed to be hidden; compare OFF latency and the diagnostic SEND_SUBMIT_END.

SEND_TOKENS is closed around callback work and resumed afterward, so its sum
still covers source token submission rather than prefix/receive work. Its calls
now count sending fragments. Other existing receive/flag stages cover the real
operations; no diagnostic boundary introduces an extra synchronization. Timer
ON is explanatory only. Performance comparisons use matching full OFF wheels,
normal uniform random top-k without repeated experts, matched seeds across
candidates and held-out seeds after selection; retain10/100/tail50 sampling.

## Local validation scope

The new native CPU fixture exercises the production one-input prefetch helper,
actual-submission cadence, early-window selector, original pending table and
final two-snapshot scheduler. It checks delayed FIRST_HOPs/terminal dependencies,
FIFO ownership, bounded scans, full-cache stop, compact-only windows and all
tasks consumed once. The layout matrix covers configured S/R splits and input
modes; the target S30/R26 FP8 shape retains6TX/6RX lanes. Source guards preserve
FillTail, ConsumePacket, IssueLoad, ConsumeLoaded and ClearFlags bodies.

CPU mocks do not prove AscendC instruction support, runtime TPipe allocation,
NPU correctness, deadlock freedom on hardware or performance. This revision has
not been compiled with CANN or run on an NPU. The unavailable remote host is not
used. All seven existing dispatch translation units and production O2/target
architecture remain; use the established ON/OFF build and complete-wheel
validation procedure when target access is available.
