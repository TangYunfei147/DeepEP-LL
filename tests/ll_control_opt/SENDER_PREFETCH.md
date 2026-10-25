# Prequantized FP8 sender: Vector metadata tail A/B

This candidate follows the MTE3 backpressure experiment. That experiment showed
that source submission, input readiness and MTE3 completion are coupled: forcing
TX completion reduced the observed input wait but increased the full dispatch
task substantially. It did not establish MTE3 as the baseline's main bottleneck.
This A/B therefore asks whether removing an avoidable Scalar dependency while
preserving asynchronous input and sending-lane reuse reduces the complete task.

## Candidate and scope

The baseline is `9027275b31a28608296b1b71c93fa6e2bb456db5`; the candidate is
`0a8e5b786763365da7541e3fd033b267d086aa5e`. Both exact source revisions are pinned
in [commits-sender-prefetch.json](commits-sender-prefetch.json).
The candidate label is `prequant-vector-tail`; the file name retains the sender
prefetch context, but this change does not increase prefetch depth or lane count.

Only asynchronous, deduplicated, prequantized FP8 sending on S cores uses the new
path. The ST's FP8 configuration supplies already-quantized input and packed INT32
scales, so the old per-packet input wait is MTE2_S, not quantization V_S.

The old path waited for input with Scalar, wrote the packet metadata tail from
Scalar, then synchronized those writes to Vector before copying the packet.
The candidate writes that tail with masked Vector Duplicate operations. This
allows input readiness to remain in the MTE2-to-Vector/TQue dependency chain and
removes the corresponding Scalar MTE2_S and S_V sequence on the selected path.
It is not safe to remove those waits while keeping the old Scalar tail writes.

The existing input queues, output-lane completion protection and packet Copy
remain. A Vector pipeline barrier orders the new tail writes before packet Copy.
There are no additional UB buffers, lanes, event IDs, cross-core barriers
or forced MTE3 drains. Non-deduplicated modes, quantizing inputs, R-core local
sending and combine retain their prior paths. The wire layout, core protocol,
overlap stage3 and schema16/profile7 layout remain unchanged.

## Target test

Use the same target and previously selected 16 devices as the MTE3 experiment.
Keep the existing CANN environment and validated project SDK. The fixed case is
EP16, B=M256, H3584, K8, E384, G4, S30/R26, MTE and HT OFF. Both complete wheels
are built with timer OFF and installed in separate environments. No capture or
MTE3 diagnostic intervention is enabled. The suite verifies wheel, native/kernel
library and SDK identities before launch; busy devices stop the next test without
terminating any process.

From the project root, with `ASCEND_RT_VISIBLE_DEVICES`, `MASTER_ADDR`,
`MASTER_PORT` and `SHMEM_PORT` set to the existing deployment values, run:

```bash
run_sender_ab() {
  local run_prefix="${1:-sender-vector-tail-001}"
  git pull --ff-only || return $?
  python tests/ll_control_opt/suite.py \
    --commits tests/ll_control_opt/commits-sender-prefetch.json \
    --devices "$ASCEND_RT_VISIBLE_DEVICES" \
    --num-nodes 1 --node-rank 0 \
    --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
    --shmem-port "$SHMEM_PORT" \
    --resume --send-cores 30 --seeds 0 17 101 \
    --rounds 2 --order alternating --run-prefix "$run_prefix" || return $?
  python tests/ll_control_opt/export_report.py \
    "results/ll-control-opt/$run_prefix" || return $?
}
if run_sender_ab sender-vector-tail-001; then
  printf '%s\n' 'A/B complete; send the single analysis-report.txt printed above.'
else
  printf '%s\n' 'A/B stopped; the terminal, logs and existing results are preserved.'
fi
```

The function returns on failure; it does not exit the interactive shell.
`--resume` reuses only fully verified build artifacts. The command deliberately
does not use `--resume-results`, so successful comparison and report files are
written directly under the selected run prefix. Use a fresh prefix for another
run, for example `sender-vector-tail-002`; existing samples are never overwritten.
The exporter does not automatically select a resumed session. If separately using
`--resume-results`, pass the parent of the `comparison.csv` path printed by that
invocation to the exporter, not the original run root.

There are two revisions, three requested seeds (0/17/101) and two rounds: 12
complete dispatch+combine ST calls. Round1 runs baseline then candidate for each
seed; round2 reverses the order. Every call checks correctness and retains the
existing 10 warmups / 100 collected tasks / last50 per rank protocol, with
Exp + scalar all_reduce for each performance iteration. Three correctness routes
precede measurement, so actual measured routing seeds are 2/19/103. FP8 dispatch
uses BF16 combine.

The suite builds both wheels before device execution and uses the same SDK
dependency/cache identity. It validates the SDK without silently replacing an
incompatible installation. Raw samples, profiler exports and logs stay under the
target project. For a CPU-only command preview, add `--plan` to the suite command
and omit the exporter.

## Return and interpretation

For the command above, send only:

```text
results/ll-control-opt/sender-vector-tail-001/analysis-report.txt
```

The exporter retains all 24 comparison rows and the original numeric text. It
checks the actual UTF-8 size and refuses to write above 65535 bytes; it does not
truncate rows. This is one file, below the ten-file limit. It omits raw samples,
per-rank detail, full hashes and local paths, which remain on the target.

Compare candidate minus baseline within each matching round and seed, separately
for dispatch and combine. Negative differences mean faster. Review all six pairs,
their order and SDK identity fields before drawing a conclusion from their mean.
An improvement would support this scheduling change; it would not by itself prove
that MTE3 backpressure is the dominant baseline cost. A regression or correctness
failure rejects the candidate for this case. These local CPU checks do not replace
target CANN compilation, device correctness or performance validation.

## Separate compile verification

The A/B command builds complete OFF wheels. To additionally check all seven
dispatch modes in fresh OFF/ON object directories and link a complete ON wheel,
load the existing project/CANN build environment and validate the project SDK
with the standard `python scripts/build_shmem.py` first. Then, from this checkout
root, use the following function. It builds without installing the ON wheel or
starting a device. Existing outputs are preserved.

```bash
check_sender_compile() {
  local out=".logs/sender-vector-tail-compile-001"
  if [ -e "$out" ]; then
    printf '%s\n' 'Choose a fresh compile output directory.'
    return 1
  fi
  mkdir -p "$out" .tmp .cache/pip || return $?
  DEEPEP_LL_SEND_CORES=30 DEEPEP_LL_RECV_CORES=26 \
    python tests/ll_stage_timer/benchmark_compile.py \
      --output "$out/off" --timer OFF --jobs 4 || return $?
  DEEPEP_LL_SEND_CORES=30 DEEPEP_LL_RECV_CORES=26 \
    python tests/ll_stage_timer/benchmark_compile.py \
      --output "$out/on" --timer ON --jobs 4 || return $?
  DEEPEP_LL_SEND_CORES=30 DEEPEP_LL_RECV_CORES=26 \
  DEEPEP_NPU_ARCH=Ascend950 DEEPEP_BUILD_HT=OFF DEEPEP_LL_STAGE_TIMER=ON \
  TMPDIR="$PWD/.tmp" PIP_CACHE_DIR="$PWD/.cache/pip" \
    python -m pip wheel -v --no-deps --no-build-isolation \
      --wheel-dir "$out/wheel-on" . > "$out/full-on-wheel.log" 2>&1 || return $?
}
if check_sender_compile; then
  printf '%s\n' 'OFF/ON object compilation and full ON wheel build completed.'
else
  printf '%s\n' 'Compile check stopped; terminal and logs preserved.'
fi
```

The two object-directory timings measure dispatch compilation only. They do not
include complete-wheel packaging, SDK preparation or NPU execution. This check
is separate from the OFF performance comparison and supplies no performance data.

The unchanged MTE3 diagnostic driver still pins its original diagnostic source.
This candidate is evaluated with OFF tasks. Since its selected path no longer
executes Scalar MTE2_S, an absent TX_INPUT_READY event in a future candidate
capture would not mean that input DMA or its dependencies take zero time.
