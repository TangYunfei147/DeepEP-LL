> 历史指南：旧打点及入口适用于原提交（最后完整旧实现e4d114c）。当前已删除旧插桩/解析；构建与采样以[reference schema7指南](../ll_reference_profile/README.md)为准。下文算法与历史结果保留。

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
immediate events use 7; immediate receive dependencies use 7. No allocator Reset occurs
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

## 4. Cooperative R scheduling

R initializes count buffers once. Before local payload and after each eight-task
quantum it probes counts once; when ready, it publishes the prefix and prepares
receive once. Subsequent callbacks probe at most 16 pending slots and consume
only ready packets, retaining FIRST_HOP priority and descending pending removal.
Empty snapshots return to sending. After all local tasks, the receiver runs to
completion. Count/prefix blocking waits have no dependency on payload consumption.
No atomics/global barriers/new per-packet routing are introduced. The input
lookahead is flushed at quantum boundaries, not remote DMA; this costs some
prefetch continuity and must be compared on the target. S retains continuous
send issue then receive. Source/output/cleanup ownership and wire format stay unchanged.

R SEND_TOKENS now accumulates send batches only; count/receive work between them
is excluded. RECV_PREPARE and SEND_SUBMIT_END are sampled separately.
`overlap_receive.csv/md` pairs same-core/round boundaries and transition waits;
negative recv_minus_submit_ticks proves receive load was submitted before the
last initial-send boundary, never DMA completion. Missing samples stay blank.
A zero transition wait does not prove communication overlap or a speedup.
`relay-complete` still deliberately adds drains: use observe/OFF for this experiment.
The standalone CPU route replay retains packet counts/static ownership but its
legacy round ordering is not a simulation of this cooperative scheduler.

## Target build and validation

Run from the repository root on every target node, after loading that machine's
CANN environment. All ranks need the same commit, dtype, core split and wheel.
No SDK rebuild is forced when its existing identity validation passes.

```bash
git fetch origin kernel/ll-g4-overlap-recv
git switch kernel/ll-g4-overlap-recv
git pull --ff-only origin kernel/ll-g4-overlap-recv
git submodule update --init --recursive --depth 1 -- third-party/shmem
source scripts/ll_timer_env.sh
unset SHMEM_ROOT CATLASS_ROOT
python scripts/build_shmem.py
export DEEPEP_NPU_ARCH=Ascend950
export DEEPEP_LL_SEND_CORES=28 DEEPEP_LL_RECV_CORES=28
export DEEPEP_LL_STAGE_TIMER=ON
export LD_LIBRARY_PATH="$PWD/third-party/shmem/install/shmem/lib:${LD_LIBRARY_PATH:-}"
python -m pip install -r requirements-build.txt
python -m pip install -v --force-reinstall --no-build-isolation --no-deps .
python -m pip install pytest
python -m pytest tests/ll_stage_timer/test_overlap_receive.py tests/ll_stage_timer/test_relay_timing.py -q
```

Set NNODES/NODE_RANK/MASTER_ADDR/MASTER_PORT/SHMEM_PORT and allocated device
visibility for the target. For EP16 use NNODES=1 with 16 local processes.
NNODES>1 with this local count means EP=16*NNODES, not EP16. Use a fresh RUN_NAME
on every run, shared across nodes. Correctness checks remain enabled.

```bash
export RUN_NAME=overlap_observe_001
python tests/elastic/test_ep_low_latency.py \
  --num-processes 16 --num-nodes "$NNODES" --node-rank "$NODE_RANK" \
  --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" --shmem-port "$SHMEM_PORT" \
  --num-tokens 256 --num-max-tokens-per-rank 256 \
  --hidden 3584 --num-topk 8 --num-experts 384 --dtype fp8 \
  --do-cpu-sync 0 --correctness-repeats 5 \
  --ll-stage-timer --ll-timer-iters 5 --ll-timer-mode observe \
  --output-dir "$PWD/results/overlap-recv" --run-name "$RUN_NAME"
python tests/ll_stage_timer/analyze.py "results/overlap-recv/$RUN_NAME" --op dispatch --trace-iteration 4
```

Collect all rank sample directories before multi-node analysis. Test bf16 and
fp16 too, B=1/8/32 with M=32 (non-dedup), B=33/256 with M>=B (dedup), top-k=1/8,
normal/hotspot routes and S/R=1/55,10/46,28/28,55/1 (rebuild for each split).
Inspect OVERLAP_PATH/SERIAL_FALLBACK and lane counters; a fallback result does not
validate overlap. Keep combine enabled for downstream metadata/window reuse checks.

For performance, rebuild each comparison commit with DEEPEP_LL_STAGE_TIMER=OFF
and repeat the command without --ll-stage-timer/--ll-timer-iters/--ll-timer-mode.
Keep the default 10 warmup / 100 samples / tail50 protocol, seed and shapes fixed.
Compare complete dispatch kernel time, not summed nested stages. Use distinct
output names; compare base781b1aa, instrumented baseline db31aab, partition
baseline5cf0049, tail-overlap63819fc and final cooperative commit. Different timer
schemas are not diagnostic speedup pairs; OFF performance can be compared with
matching shape and core split.

Local validation does not compile CANN kernels or run NPUs. Native C++ planner
and guard tests are skipped when no C++ compiler is installed; run them on the
target as shown above before device validation.

## Local check record

2026-10-21: full timer suite plus routing report, group ownership, input prefetch,
relay priority, frequency histogram and protocol/core checks: 485 passed,
49 skipped (no C++ compiler/POSIX). Additional OFF comparison identity check
passes; the complete run's final total is 486 passed / 49 skipped.
EP16 B256/H3584/K8/E384 BF16+FP8 CPU --plan succeeds. These checks do not prove
CANN compilation, event execution on hardware, numerical device correctness or
performance.

## Target compile fix: explicit device callback

The first target build of d4109a8 rejected the plain cooperative lambda as a Host
callable (including its device timer Scope). The callback now uses a named
LocalProgress functor and an explicitly __aicore__ AdvanceLocal method. The UB
planner's alignment lambda also uses the existing Host/AICore-qualified
DispatchDedupAlign helper. Scheduling, barriers, schema11 and protocol14000+S
are unchanged. Rebuild the full wheel after updating; no SDK rebuild is required
for this source fix. Local CPU tests cannot prove Bisheng call-domain compatibility.
