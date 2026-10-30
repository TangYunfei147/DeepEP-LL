> 历史指南：旧打点及入口适用于原提交（最后完整旧实现e4d114c）。当前已删除旧插桩/解析；构建与采样以[reference schema7指南](../ll_reference_profile/README.md)为准。下文算法与历史结果保留。

# Adjacent-pair receive balancing for cb59-stream

`DEEPEP_LL_DISPATCH_IMPL=cb59-pair` is an independent candidate against
`cb59-stream`. Source-side counting and sending remain the original cb59
algorithm. The existing stream count protocol still assigns contiguous
count-record slices and writes all public counts; only payload ownership is
balanced between two adjacent receive cores.

## Scope and identity

| Item | Reference | Candidate |
|---|---|---|
| Selector | `cb59-stream` | `cb59-pair` |
| Native implementation | `cb59-stream-recv` | `cb59-pair-recv` |
| Native selector macro | 2 | 3 |
| Core protocol | 22028 | 23028 |
| Group / cores / raw P | G16, S28/R28, P0 | G16, S28/R28, P0 |
| Specialized scope | EP16, M<=32, 1<=B<=32 | same |
| Other valid shapes | original cb59 receive protocol | original cb59 receive protocol |

The per-case `effective_dispatch_impl` records the actual specialized path or
`cb59` fallback. All seven dtype entries remain available in the same separate
translation units. If a specialized shape exceeds the conservative UB budget,
the Host rejects it; resource failure does not select a different algorithm.
All ranks must use matching complete wheels and the same call sequence.

## Receive algorithm

The 28 R cores form fixed pairs `(0,1), (2,3), ..., (26,27)`. Each core first
finishes the existing stream protocol for its own record slice, including
zero-count records and cross-slice expert-count output. It then:

1. Publishes its exact dense output begin/count and record-slice identity to
   its partner, after ordering its public `sendCounts` slice before readiness.
2. Reads its partner's corresponding header. The two original record slices
   must describe adjacent dense output ranges.
3. Splits the pair's combined dense range at its midpoint. The even core gets
   the first half and the odd core gets the second half; their row counts
   differ by at most one.
4. Uses its local prefix for rows from its own slice. If the assigned range
   intersects the partner's slice, it reads that slice's public prefix.
5. Constructs tasks for the assigned dense rows and runs the unchanged packet
   probe, payload copy, metadata output, pending deletion and flag cleanup.

A midpoint may fall inside one source/expert record. The task builder keeps
`q = output_row - original_record_begin`; it does not restart q at the new
consumer boundary. Public dense output order, packet addresses and combine's
metadata format remain unchanged. Count-output ownership remains with the
original record producer, while packet/output/cleanup ownership follows the
new pair range.

A core with zero rows in its original slice can receive rows from its partner.
Only the final pair range decides whether payload work is empty. With EP16 and
E divisible by 16, the number of active record owners is 16 or 28, so an
inactive record owner always has an inactive partner.

## What the balance does and costs

This balances **within each fixed pair**. It is not a global work queue and
does not move a busy pair's work to an unrelated idle pair. A hotspot at a pair
boundary can still leave different pairs with different loads. The experiment
tests whether local workload variation contributes to the larger-B completion
behavior. The previous summaries did not establish imbalance as the sole
cause, and this candidate does not guarantee an improvement for normal routes
or very small B.

The added costs are one pair-header publication and wait, header validation,
and an optional partner-prefix read. The public-prefix-to-header dependency
uses an MTE3 ordering barrier. There is no additional all-core rendezvous,
extra kernel launch, device-to-host query, or full TPipe Reset on this path.

## Handoff reuse, lifetime and budget

Each directional mailbox has exactly one writer and one consumer. It occupies
one 32-byte cell of the legacy full-prefix-ready matrix, which stream counting
does not use. The consumer clears all eight header words after acquiring the
metadata and any needed prefix, so a later legacy call cannot mistake stale
header payload for old readiness values. Stream's triangular sum matrix is
separate. Existing two-bank selection and final retirement remain intact.

One packed TBuf contains three disjoint regions: immutable outgoing header,
incoming snapshot subsequently reused as an immutable clear source, and the
optional partner prefix. The GM prefix read starts on a 32-byte boundary and
uses a bias to skip leading words; its tail is not rounded past the partner
slice or the public tensor end. Leading words are never used as dependencies.

The shared scratch pool contains at most **16 TBuf objects**: eight persistent
stream buffers, one packed handoff buffer, two task tables, four probe buffers,
and one packet-lane buffer. Every asynchronous source remains allocated and
unchanged until the original final drain. Source-count records stay outside
this scratch pool. Host and device include the packed-buffer bytes and the
pair's conservative row bound before selecting up to eight payload lanes from
the unchanged 190 KiB limit; at least one lane must fit.

## Timing interpretation

Both freshly built implementations use sparse schema16 with the same 512-byte
per-core carrier. For `cb59-pair`, `PREFIX_READY` means **this pair consumer's
assigned range and required prefix are available for reception**. It includes
the pair handoff and is later than stream's local offset-ready boundary by
design. It does not mean that every pair or the full-rank prefix is ready.

Use complete OFF profiler kernel tasks as the performance result. Keep normal
and hotspot cases separate. Use matched same-core ON intervals to locate work,
report ON/OFF perturbation, and do not sum group maxima or subtract separate
percentiles. Initialization points are described in [CB59_SMALL.md](CB59_SMALL.md).

## Target procedure: clean build both experiments before sampling

Use the delivered source revision and an exclusively allocated project checkout
in the existing target Python/CANN environment. SHMEM manifest/hash/field checks
have been removed. The shared build cleans and rebuilds the SDK once, then
cleans project wheel products before each of the eight OFF/schema16 wheels for
small and pair. It preserves sources, downloads and earlier results.

This is the same build block as CB59_SMALL.md: **execute it only once**. If it
already succeeded in this terminal, retain AB_BUILD_DIR/AB_BUILD_OK and proceed
directly to the correctness section below. Sampling never cleans/rebuilds the
SDK. Use a fresh build group/run suffix if these paths already exist.

```bash
source /home/tyf/0env.sh
export DEEPEP_LL_SEND_CORES=28 DEEPEP_LL_RECV_CORES=28
export DEEPEP_LL_SMALL_B_PREFIX_CORES=0
unset SHMEM_ROOT CATLASS_ROOT
export LD_LIBRARY_PATH="$PWD/third-party/shmem/install/shmem/lib:${LD_LIBRARY_PATH:-}"

AB_BUILD_NAME=ll_ab_clean_build_003
AB_BUILD_DIR="diagnostic-runs/$AB_BUILD_NAME"
AB_BUILD_OK=0
if python tests/ll_stage_timer/build_ab.py --experiment all --run-name "$AB_BUILD_NAME"; then
  AB_BUILD_OK=1
  echo 'One fresh SDK and all eight OFF/schema16 wheels are ready.'
else
  echo 'Build stopped; preserve logs. Terminal remains open.'
fi
```

## Check normal and hotspot dispatch/combine

Set `MASTER_ADDR`, `MASTER_PORT` and `SHMEM_PORT` to this job's actual allocated
values. The device list is for the allocated 16-card target. The function uses
the pair candidate's already-built isolated venv and OFF wheel; it does not
modify shared Python or rebuild the SDK. Repeated fresh routes and repeated
combine exercise metadata and two-bank reuse before full profiling.

```bash
check_pair_combine() {
  local node py wheel route
  if [ -z "${MASTER_ADDR:-}" ] || [ -z "${MASTER_PORT:-}" ] || [ -z "${SHMEM_PORT:-}" ]; then
    echo 'Set MASTER_ADDR, MASTER_PORT and SHMEM_PORT for this job first.'
    return 1
  fi
  node="${AB_BUILD_DIR}/cb59-pair/node_000"
  py="$node/venv/bin/python"
  wheel=$("$py" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); w=Path(json.loads(p.read_text())["off"]); print(w if w.is_absolute() else p.parent/w)' "$node/wheel-map.json") || return $?
  "$py" -m pip install --no-index --no-deps --force-reinstall "$wheel" || return $?
  for route in normal hotspot; do
    DEEPEP_LL_DISPATCH_IMPL=cb59-pair "$py" tests/ll_stage_timer/run.py \
      --devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
      --num-nodes 1 --node-rank 0 \
      --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" --shmem-port "$SHMEM_PORT" \
      --num-tokens 1 8 32 --num-max-tokens-per-rank 32 \
      --hidden 3584 --num-topk 8 --num-experts 384 --dtype fp8 --do-cpu-sync 0 \
      --route "$route" --skip-perf-test --warmup 1 --num-iters 1 \
      --correctness-repeats 6 --repeat-combine 2 \
      --output-dir diagnostic-runs/pair_ab_combine_checks \
      --run-name "${route}_003" || return $?
  done
}
PAIR_CHECK_OK=0
if [ "${AB_BUILD_OK:-0}" = 1 ] && check_pair_combine; then
  PAIR_CHECK_OK=1
  echo 'Normal and hotspot dispatch/combine checks passed.'
else
  echo 'Build incomplete or correctness failed; terminal and logs remain available.'
fi
```

## Run the matched OFF/sparse A/B

Keep the same terminal. Both route distributions use identical seeds and the
10-warmup/100-sample/per-task-tail50 OFF protocol. Sparse captures are separate
and report their own complete-task perturbation.

```bash
run_pair_ab() {
  local route impl
  for route in normal hotspot; do
    for impl in cb59-stream cb59-pair; do
      DEEPEP_LL_DISPATCH_IMPL="$impl" \
        python tests/ll_stage_timer/diagnose.py --skip-build \
          --wheel-map "${AB_BUILD_DIR}/${impl}/node_000/wheel-map.json" \
          --run-name "pair_ab_${impl}_${route}_run_003" \
          --bs 1 2 4 8 16 24 32 --m 32 \
          --hidden 3584 --topk 8 --experts 384 --dtype fp8 --cpu-sync 0 --route "$route" \
          --devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
          --num-nodes 1 --node-rank 0 \
          --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
          --shmem-port "$SHMEM_PORT" || return $?
    done
  done
}
if [ "${PAIR_CHECK_OK:-0}" = 1 ] && run_pair_ab; then
  echo 'Both implementations and both route distributions completed.'
else
  echo 'Prerequisite incomplete or profiling stopped; terminal remains open.'
fi
```

Return only the four corresponding `SEND_FIRST.txt` files. Before sending,
this read-only command confirms the actual byte counts and four-file limit:

```bash
python - <<'PY'
from pathlib import Path
reports = [Path(f"diagnostic-runs/pair_ab_{impl}_{route}_run_003/SEND_FIRST.txt")
           for route in ("normal", "hotspot") for impl in ("cb59-stream", "cb59-pair")]
assert len(reports) <= 10
for report in reports:
    data = report.read_bytes()
    data.decode("utf-8")
    print(f"{len(data):6d} bytes  {report}")
    assert len(data) <= 65535, f"Report exceeds the per-file limit: {report}"
PY
```

Raw profiling, traces and logs remain on the target. Compare normal and hotspot
results independently; repeat important boundaries in reversed A/B order before
claiming a stable gain. A separate call sequence should cover legacy fallback
and other dtype/EP combinations before broader deployment.

## Local validation and limits

CPU models cover pair boundaries inside records, one empty payload owner,
uneven counts, inactive pairs and two-bank reuse. The real PairHandoff header
has been exercised in 50 pair cases and 24 interleaved stream cases with CPU
threads and checked memory; task tests verify unique dense-row/slot/cleanup
ownership. Shared source/payload functions retain their cb59 behavior.

These checks do not establish NPU cache visibility, asynchronous DMA timing,
Bisheng code generation, device correctness or performance. No SSH, CANN
compilation or NPU execution was performed for this candidate. Full target
compilation, correctness and measurements are still required.
