# Purpose-specific MTE3 backpressure diagnosis

This experiment asks whether MTE3 completion dependencies explain the late
arrival of S cores at reception and the dispatch tail. It is a diagnostic
intervention, not a candidate optimization. Use the same machine and deployment
environment as the preceding baseline report.

The fixed case is EP16, B=M256, H3584, K8, E384, G4, S30/R26, HT OFF and the LL
MTE runtime. The ST's `fp8` input is already quantized: it passes FP8 values and
INT32 packed scales to the prequantized dispatch entry. Its per-packet input
wait is **MTE2_S**, not a quantization V_S. This test runs dispatch only.

Both wheels use the frozen diagnostic source
`1b9ff6b2448aeb19daf20381219498eaacf3027c`. The business reference remains
`ed1c5531dbb22d4d2a6b5571a2d1bce17a3683b5`; the report records the runner revision
separately. Updating the runner does not silently change the measured kernel.

## What is measured

Schema16/profile7 (`mte3`) retains the 512-byte aligned per-core carrier. It
contains 18 first-occurrence milestones and eight statistics. Each statistic
stores raw sum ticks, maximum single-call ticks, call count, and an empty-control
sum where applicable. All writes to the diagnostic tensor follow WORK_DONE.
There is no diagnostic SyncAll or per-packet GM export. The old schema15 reports
remain readable, and their pinned baseline command remains supported.

All three ON conditions time these **existing** Scalar-visible waits:

| Metric | Exact scope | Interpretation on this case |
|---|---|---|
| TX_INPUT_READY | Original per-packet input MTE2_S; other quantizing entries use their existing V_S | Residual input readiness, potentially including upstream pipeline dependencies; one call per sent packet |
| RX_PREPARE_VS | Existing V_S at the end of receive task-table Build | Vector completion before Scalar consumes the receive table; can include earlier V work delayed by MTE3_V |
| RX_PAYLOAD_MTE2S | Existing MTE2_S before consuming a loaded receive packet | Residual receive payload readiness; one call per received packet |
| RX_FINAL_ALL | Existing final receive PIPE_ALL after flag clearing | Remaining pipelines including output, relay, cleanup and any remaining source work |

These counters are deliberately bounded to four dependency sites. There are no
poll-count, empty-poll, bandwidth, port-utilization or per-packet timeline claims.
The same counters are present in all ON modes, so their shifts can be compared.

The additional first-hit points are SOURCE_SUBMIT_DONE, PREFIX_WAIT_ENTER,
PREFIX_WAIT_EXIT and RECV_PENDING_EMPTY. SOURCE_SUBMIT_DONE follows source packet
and metadata submission; it may already include backpressure. PREFIX_WAIT is
the actual WaitCumSumFlag call, not the earlier mixed prefix/local-send envelope.

## Controlled interventions

| Condition | Binary | Added dependencies |
|---|---|---|
| OFF | Full timer OFF wheel | None; formal 10 warmup / 100 tasks / per-rank tail50 |
| observe | Same full timer ON wheel used by TX and RX | None; timestamps around the existing waits above |
| mte3-tx | Same ON wheel | S cores only: MTE3_S before each reused sending lane and after all source/metadata submissions |
| mte3-rx | Same ON wheel | S cores only: MTE3_S before each reused receiving lane and after receive pending tasks become empty |

Every added MTE3_S immediately repeats the same SetFlag/WaitFlag pair without
issuing another transfer. The first interval is **raw exposure**, the second is
an **empty control**. Both remain in the full task duration. The report preserves
both sums and their signed difference; it never clips negative differences or
subtracts them from an OFF latency.

The synchronization uses the separate MTE3_S event domain and consumes each
event before reusing it. It does not reuse a live sending MTE3_V or receiving
MTE3_MTE2 lane event. All added waits are gated on an attached diagnostic buffer,
the requested mode, dispatch, and `core < send_cores`. OFF and ordinary ON calls
without a capture buffer execute no diagnostic wait or clock read. R cores keep
their original synchronization; global traffic changes may still affect them.

MTE3_S waits for the previously queued MTE3 work **and its producer dependencies**.
It is not a measurement of pure DMA time, network time, peer software consumption
or wire saturation. A Scalar timestamp pair around MTE3_V alone would only expose
instruction submission, so this experiment instead uses a Scalar-visible wait.

## Run on the target

Keep the previously selected 16 devices and MASTER_ADDR, MASTER_PORT, SHMEM_PORT.
The ports must be explicit and distinct. From this branch's project root:

```bash
git pull --ff-only
python tests/ll_control_opt/profile_mte3.py \
  --devices "$ASCEND_RT_VISIBLE_DEVICES" \
  --run-prefix mte3-backpressure-001
```

The driver pins an immutable diagnostic source commit, builds complete OFF and
ON wheels into a dedicated `.tmp/ll-mte3-profile/` cache, installs them into two
isolated venvs, and verifies both before any device launch. It uses the existing
validated project SHMEM SDK and does not silently rebuild an incompatible SDK.
Native, kernel, source and SDK identities are checked again from all 16 running
ranks. Existing environments/results are preserved. Logs and caches stay under
the project. A non-idle device aborts the next case; no process is killed.

Defaults are three requested seeds 0/17/101 (actual measured routes 2/19/103).
OFF runs first for each seed. The ON order rotates observe/TX/RX, TX/RX/observe,
RX/observe/TX. Every ON condition has 10 ordinary warmups, 100 ordinary settling
calls, then five actual attached captures, each preceded by Exp + scalar
all_reduce. Each capture's output is checked after sampling. Its full device
task is obtained from its own profiler record, including final diagnostic export.
The ordinary settling calls do not execute intervention waits. The five measured
intervention calls should therefore be read as repeatable injected trials after
the same baseline prelude, not as an uninstrumented steady-state performance run.

`--plan` prints the commands without building or starting devices.
`--build-only` builds both complete wheels and performs Host import/layout checks
without starting devices. `--run-only` reuses the two fully validated builds.
Use a new run prefix for another run; partial results are not overwritten.
Build logs include build time and failures. Cold OFF/ON native object builds are
separated by the existing timer build-cache configuration; no mode or production
optimization level is removed to make compilation faster.

## Return one file

```text
results/ll-mte3-profile/mte3-backpressure-001/mte3-report.txt
```

The exporter rejects a report larger than 65535 UTF-8 bytes and never truncates
evidence. It writes one file with exclusive creation, below the agreed ten-file
limit. Raw samples and profiler output stay on the target. If export alone needs
to be rerun, use a fresh output name:

```bash
python tests/ll_control_opt/report_mte3.py \
  results/ll-mte3-profile/mte3-backpressure-001 \
  --output results/ll-mte3-profile/mte3-backpressure-001/mte3-report-002.txt
```

## How the evidence answers the question

First compare observe's full task with OFF to quantify instrumentation
perturbation. Then compare TX and RX with observe, which has identical statistics,
capture count and binary. Match each seed/rank/core after averaging captures;
never subtract clocks across ranks or compare independently selected slow cores.
Fixed routing must preserve sent/received packet counts and lane configuration.

If an existing wait on the observe critical path decreases while the corresponding
MTE3_S exposure grows, with little change in total task time, this supports a
transfer of an existing dependency cost. For the prequantized case, inspect
TX_INPUT_READY and RX_PREPARE_VS rather than a nonexistent quantization wait.
RX injection can also move receive payload/lane and final drain costs.

A large forced wait or a large task slowdown by itself does **not** prove that
MTE3 dominates the baseline: drains destroy overlap, alter arrival order and
change contention. If observe itself is strongly perturbed, precise baseline
attribution is also unresolved. The report keeps raw measurements and identifies
these limits; it does not manufacture a main-bottleneck verdict or an optimization
gain. Even strong dependency evidence would not distinguish link latency, remote
memory service and queued local work without a separate experiment.

## Validation status

Local CPU checks cover layouts, event/capture guards, OFF/null-buffer behavior,
entry linkage, parser rejection, report identity/count matching and isolated
build/run orchestration. CPU mocks do not validate Bisheng code generation or
real device pipeline timing. CANN compilation and NPU correctness/performance
for this revision must be established by the target command above.
