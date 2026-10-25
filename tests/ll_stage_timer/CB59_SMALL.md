# Small-direct compile experiment for the historical cb59 timing gap

`cb59-small` is an independent, temporary compile experiment. It accepts only
**1 <= B <= M <= 32**, retains the original valid EP domains and all seven
dispatch dtype entries, and does not compile a large-B/dedup fallback. The
regular `cb59` selector remains the general replay baseline. The original six
generated replay files and the production large-B paths are not removed.

The purpose is to test whether generic control flow and the resulting compiler
code/register layout contribute to the historical 17–22 us versus replay
22–27 us difference. This is a hypothesis, not a measured attribution or a
promise to recover that difference.

## What actually changes

The isolated files under `csrc/kernels/dispatch/low_latency/cb59_small/` keep
the original small-call instructions, source/receive ownership, allocation
sequence, 190 KiB UB budget, packet layout, event IDs and synchronization order.

- Sender and receiver `enableDedup_` become `static constexpr false`. Their
  constant-only branches use `if constexpr`, so dedup/relay bodies are not
  instantiated through those call sites.
- Count `useAllCoreCount_` becomes `static constexpr true`, because the admitted
  B range is strictly below the historical B512 participation threshold. The
  original expert-versus-vector histogram selection is retained: a small B
  can still have enough B*K routes to use the vector path.
- Shape-dependent expressions and actual packet-layout checks remain runtime
  operations. The zero-token sender check is not changed to an unconditional
  return: flattened direct-route ownership differs from token ownership.

This is more than a Host rejection of B>32. The device classes themselves have
compile-time mode constants. It is also deliberately narrower than an overall
initialization rewrite: timer members, tiling ABI, SDK, pages, all UB setup and
the current seven-TU launch scheme remain as in the replay.

Generate with `python scripts/generate_cb59_small.py`; verify with `--check`.
The generator first validates its replay source against the immutable cb59
business-token fixture, then applies a short list of reversible substitutions.
Inverse normalization compares every business token and synchronization call
to that same fixture. Native CPU tests additionally prove that the actual
constant branch expressions do not instantiate dedup/relay for any of the
seven mode IDs. These are not CANN compilation or device-correctness results.

The implementation ID is 4, native protocol is 24028, and G16/S28/R28/P0 are
fixed. Every rank must install the matching complete wheel. An out-of-scope
shape is an error, not a silent switch to another implementation.

## What remains different from the old binary

The historical cb59 used one global entry containing all seven dtype branches
and a 144-byte Host launch structure. The current replay uses seven separate
mode translation units, validates that the selected mode matches tiling, and
has a 152-byte launch structure including an unused legacy expert-timer pointer.
These differences can alter compiler allocation and entry code; source alone
does not establish their cost.

The 32-byte tiling structure, data-layout formulas, magic-rotation function and
quantization implementation are unchanged from cb59. Replay still uses the
old 190 KiB arena. Both old and present OFF timers compile their recording
operations away, so ON recording overhead does not explain an OFF residual.

The user has confirmed that the historical run and current comparison both
use `cpu_sync=0`; this is not an unresolved configuration difference. Both ST
versions implement the same default 10-warmup/100-sample/per-rank-tail50
protocol and maximum-of-rank-means statistic, and the historical values were
confirmed to be complete timer-OFF profiler kernel durations.

SDK revision, page/mapping setup and compiler/runtime environment remain
separate variables. The historical original was not rerun alongside this
comparison under the same current target conditions. This experiment changes
none of those variables. If the small compile is still slower than the
historical values, perform that matched original-binary rerun before
attributing the remainder to any one of them.

## Initialization diagnosis and interpretation

The matching sparse schema16 adds four one-shot points while retaining the
512-byte per-core carrier and the existing ENTRY/INIT_DONE/WORK_DONE points:

| Same-core interval | Included device work |
|---|---|
| ENTRY → TILING_DONE | FFTS setup, timer setup and tiling reads/validation |
| TILING_DONE → PIPE_READY | TPipe construction |
| PIPE_READY → OBJECT_READY | Kernel/member-object construction |
| OBJECT_READY → MAGIC_DONE | Initial Init assignments and original magic read/write/dependencies |
| MAGIC_DONE → INIT_DONE | Count and role-specific initialization |

OBJECT_READY→MAGIC_DONE is not an isolated magic microbenchmark. No new
business barrier is inserted at these points. Compare complete OFF profiler
tasks for performance and matched same-core, same-call ON intervals for
location. Use the individual intervals before aggregation; do not subtract
separate percentiles to reconstruct them.

ENTRY is the first source-level device clock read. Compiler prologue or
argument unpacking before that read, and epilogue after WORK_DONE, are outside
its span. **Do not subtract ON ENTRY→WORK_DONE from an OFF profiler task and
call the difference launch overhead.** Those are different calls with
different instrumentation. A standalone empty entry can measure a task floor,
but its optimized argument use and prologue need not match a full dispatch.

## Target A/B: build both implementations before running

Update this branch and run from the project root in the existing target
Python/CANN environment. Keep the validated SDK. Use fresh run names if these
already exist. These functions return failure without closing the terminal.
Build both `cb59` and `cb59-small` again from this same source revision: four
fresh wheels in total, each implementation's OFF and schema16 sparse variant.
Do not reuse the historical cb59 or schema15 wheel maps for this comparison.

```bash
source /home/tyf/0env.sh
export DEEPEP_LL_SEND_CORES=28 DEEPEP_LL_RECV_CORES=28
export DEEPEP_LL_SMALL_B_PREFIX_CORES=0
unset SHMEM_ROOT CATLASS_ROOT
export LD_LIBRARY_PATH="$PWD/third-party/shmem/install/shmem/lib:${LD_LIBRARY_PATH:-}"

build_small_ab() {
  local impl
  for impl in cb59 cb59-small; do
    DEEPEP_LL_DISPATCH_IMPL="$impl" \
      python tests/ll_stage_timer/diagnose.py --build-only \
        --run-name "small_scope_${impl}_build_001" || return $?
  done
}
SMALL_BUILD_OK=0
if build_small_ab; then
  SMALL_BUILD_OK=1
  echo 'Both complete OFF/ON builds finished.'
else
  echo 'Build stopped; terminal and logs remain available.'
fi
```

Use the same terminal and the job's actual communication address/ports:

```bash
run_small_ab() {
  local impl
  if [ -z "${MASTER_ADDR:-}" ] || [ -z "${MASTER_PORT:-}" ] || [ -z "${SHMEM_PORT:-}" ]; then
    echo 'Set MASTER_ADDR, MASTER_PORT and SHMEM_PORT for this job first.'
    return 1
  fi
  for impl in cb59 cb59-small; do
    DEEPEP_LL_DISPATCH_IMPL="$impl" \
      python tests/ll_stage_timer/diagnose.py --skip-build \
        --wheel-map "diagnostic-runs/small_scope_${impl}_build_001/node_000/wheel-map.json" \
        --run-name "small_scope_${impl}_run_001" \
        --bs 1 2 4 8 16 24 32 --m 32 \
        --hidden 3584 --topk 8 --experts 384 --dtype fp8 --cpu-sync 0 \
        --devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
        --num-nodes 1 --node-rank 0 \
        --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
        --shmem-port "$SHMEM_PORT" || return $?
  done
}
if [ "${SMALL_BUILD_OK:-0}" = 1 ] && run_small_ab; then
  echo 'Both A/B runs finished.'
else
  echo 'Build incomplete or test stopped; terminal remains open.'
fi
```

First return only the two `SEND_FIRST.txt` summaries from those run directories.
The diagnostic exporter enforces the existing individual UTF-8 byte and total
file limits; keep raw profiling and logs on the target. The same route seed,
10/100/tail50 OFF protocol and separate ON captures are used for both builds.
An independent reverse-order run is needed before claiming a stable benefit.
