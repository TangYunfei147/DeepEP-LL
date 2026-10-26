# Rebuild and measure the original cb59 code

## Evidence and comparison limits

The supplied `small_scope_cb59-small_run_003` uses `f8bf1bd`, EP16,
M32/H3584/K8/E384, prequantized FP8, CPU sync disabled and S28/R28.
All seven cases have80/80 rank-round captures and56/56 cores.

| B | Historical original cb59, us | Small run003, us | Difference, us |
|---|---:|---:|---:|
| 1 | 17.5150 | 22.0143 | 4.4993 |
| 2 | 17.5547 | 22.3629 | 4.8082 |
| 4 | 18.0762 | 22.6935 | 4.6173 |
| 8 | 18.1107 | 22.2213 | 4.1106 |
| 16 | 19.1610 | 23.1057 | 3.9447 |
| 24 | 19.9418 | 23.9153 | 3.9735 |
| 32 | 21.9017 | 25.9518 | 4.0501 |

These are separate historical runs, not a matched causal comparison. Removing
generic large-B branches did not recover the historical range. This identifies
neither SDK/page mapping nor entry compilation as the cause.

ON adds2.53–4.20us (10.3–18.2%). `INIT_TO_COUNTS_RECEIVED` includes local count
sending, source-count waits, extraction and Sum/V_S. `PREFIX_LOCAL_TO_READY`
includes total-count exchange, cleanup, prefix correction,28 copies, ready
publication/waits, prefix reload, expert-count output and original PIPE_ALL.
These roughly7us spans are sequential within one core/round. The report takes
independent maxima over R cores and medians over rank-rounds; adding the cells
cannot reconstruct the roughly15us entry-to-prefix-ready cell. Neither is pure
network waiting. OFF task minus an ON span is not launch-overhead measurement.

## Match actual routes

Original ST profiles its declared seed. Before this fix, current ST's default
three correctness checks changed routes to `seed+2` and retained the final
route for performance while recording the original seed. The documented default
run003 command therefore sampled seed2; historical cb59 defaulted to seed0.
The compact file itself lacks an effective seed: this interpretation assumes
the documented command without correctness-repeat overrides.

At equal seeds, CPU checks find identical original/current normal routes at
all seven B values. Seed0-to-seed2 changes maximum-rank receive rows in both
directions, so it is not proof of the entire3.94–4.81us gap.

Current ST now restores the declared seed and matching output/handle after
repeated correctness checks. It records `profile_seed`, `profile_route_sha256`
and fingerprint schema1. The fingerprint covers the complete all-rank route
list serialized as compact ASCII JSON. OFF/ON comparisons check recorded route
identities; missing historical fields are not invented. The original runner
records the same fingerprint from the original route function.

## Original build boundary

The user retains source but needs a rebuild. The runner clones fixed commit
`cb59bff99ecd39d27ef57788c5e6fd0009e4e636` into a new isolated project under
this checkout's diagnostic group. It retains original SHMEM source, default
1GiB huge pages, MTE+UDMA Session initialization (LL payload is MTE), dynamic
AIV1:1 split, G16, single dispatch entry,144-byte launch, all seven modes, O2
and original device build graph. Actual AIV is recorded; a count other than56
is not silently treated as replay's S28/R28.

Only build/loader/preflight SDK validation is removed and clean-build helpers
are supplied. C++ sources remain byte-identical; the exact non-kernel patch list
is saved. No SDK manifest, capability field or library hash gates this build.
This historical single-entry experiment does not merge current seven-mode TUs.

Original/current SDK paths are separate. Cleanup affects only the new clone's
SDK outputs and wheel build products, preserving the parent's SDK, shared
Python, previous wheels and runs. Compiler/linker failure stops without an
automatic SDK upgrade or algorithm change. Original monolithic compilation
may take longer than the current split-TU build.

## Target commands

From the current `kernel/ll-g4-small-b-prefix` checkout, load the existing target
Python/CANN environment. Python build dependencies must already be installed.
Git obtains original source and SHMEM submodules only; CATLASS is not fetched.
Pip installs the locally built wheel
only into the group's isolated venv.

```bash
ORIGINAL_RUN=cb59_original_repro_001
ORIGINAL_BUILD_OK=0
build_original_cb59() {
  git pull --ff-only || return $?
  git fetch origin cb59bff99ecd39d27ef57788c5e6fd0009e4e636 || return $?
  python tests/ll_stage_timer/cb59_original.py build --run-name "$ORIGINAL_RUN"
}
if build_original_cb59; then
  ORIGINAL_BUILD_OK=1
  echo 'Original cb59 OFF wheel ready; no device test has run yet.'
else
  echo 'Build stopped. Preserve logs; terminal remains open.'
fi
```

Use a fresh run name if its directory exists. Add `--plan` for a read-only
preview without directory creation, compilation or NPU execution. Set
`MASTER_ADDR`, `MASTER_PORT` and `SHMEM_PORT` to allocated job settings.
The16-device list is for the user's target, not assistant remote execution.

```bash
run_original_cb59() {
  if [ -z "${MASTER_ADDR:-}" ] || [ -z "${MASTER_PORT:-}" ] || [ -z "${SHMEM_PORT:-}" ]; then
    echo 'Set MASTER_ADDR, MASTER_PORT and SHMEM_PORT for this job first.'
    return 1
  fi
  python tests/ll_stage_timer/cb59_original.py run --run-name "$ORIGINAL_RUN" \
    --devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
    --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" --shmem-port "$SHMEM_PORT"
}
if [ "${ORIGINAL_BUILD_OK:-0}" = 1 ] && run_original_cb59; then
  echo "Return only diagnostic-runs/$ORIGINAL_RUN/SEND_FIRST.txt"
else
  echo 'Build incomplete or measurement stopped. Preserve logs; terminal remains open.'
fi
```

Defaults: seven B values, seeds0/2, two rounds with reversed seed order0,2,2,0.
Each serial run uses original OFF ST,10 warmups/100 samples/per-rank tail50,
maximum of rank means, one Exp plus scalar all-reduce per sample, and correctness
checks. Raw tasks/logs stay on target. Return only `SEND_FIRST.txt`, limited
to60,000 UTF-8 bytes. Regenerate from completed raw results with:

```bash
python tests/ll_stage_timer/cb59_original.py analyze --run-name "$ORIGINAL_RUN"
```

## What the result decides

If original seed0 reaches its historical range again, preserve that successful
build as the current-day baseline. Seed2 contextualizes the small report;
collect a same-day replay with identical route fingerprints before claiming
a stable difference. If original seed0 is also slower now, first inspect recorded
compiler/CANN/Python environment, actual cores and repeat variation. Current
replay-only timestamps cannot explain that historical change by themselves.

After baseline reproduction, proposed single-factor follow-ups are: huge pages
to ordinary pages; MTE+UDMA to MTE-only Session; SDK and required API compatibility;
split mode TUs retaining144-byte launch; finally152-byte wrapper. These are not
applied by this runner. Report technically coupled changes explicitly.

Internal timing should use a separate original diagnostic copy with matching
one-time endpoints and original entry/ABI. Check its ON/OFF perturbation first.
Add raw-count/sum/ready endpoints only if matched results leave those intervals
unexplained; no new barriers or per-poll statistics are needed for initial reproduction.
