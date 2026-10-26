# Source-level LL timer removal and measured-benefit candidate A/B

## Purpose and candidate selection

The user requests a test of whether LL instrumentation affects compiler
optimization or Scalar work even when recording is OFF, followed by testing
optimizations that already have positive target evidence. Empty recording
methods do not establish identical generated code: arguments, object members,
call structure and Host/Python integration may still differ.

This experiment builds three complete wheels from fixed base
`439d4597a52b657858217bac7fc1e965f22507f7`. The controller checkout's production
sources and the user's reproducible small_bs-DeepEP deployment are preserved.

| Wheel | Algorithm | LL instrumentation source | Host dispatch/combine launch sizes |
|---|---|---|---|
| replay-off | cb59 migration | Existing source, recording OFF | 152 / 96 bytes |
| replay-clean | Same cb59 migration | Physically removed | 136 / 88 bytes |
| stream-clean | cb59-stream | Physically removed | 136 / 88 bytes |

The sizes describe Host launch structures, not communication payload bytes.
The same SDK, seven mode translation units, all supported modes, O2, G16,
S28/R28/P0, MTE Session, ordinary pages and benchmark protocol are used.
This is not a recreation of the old cbd04eb single-entry/old-SDK binary.

Only **cb59-stream** enters the formal second-stage candidate list. Its earlier
same-source target A/B is recorded in [CB59_STREAM.md](CB59_STREAM.md): normal
B1/2/4/8/16/24 improved8.69–16.35%, and hotspot B1/2/4/8 improved9.47–19.60%.
Normal B32 improved only0.3609us; hotspot B16/24/32 regressed. These boundary
cases remain in the matrix as guards, not as claimed positive-benefit cases.
The earlier evidence is one target A/B, not a guarantee of a stable gain after
timer removal or under other routes.

That earlier diagnose run measured dispatch-only and used the former default
effective seed2. This matrix explicitly restores seed0 and keeps full
dispatch+combine on both arms. A past gain is the reason to select stream;
it is not a substitute for validating these new matched conditions.

Pair has no returned target OFF A/B; small has not demonstrated a matched
improvement. They are excluded. P1 and count-ready/extract VF had positive
results in a different G4/P1 prefix design; that does not establish their
benefit after moving them into cb59's distributed prefix. They are not silently
combined with this experiment. The source projection nevertheless covers all
five existing selectors so unused source cannot retain a hidden timer path.

## What clean-source removes

The generated independent clone physically removes stage/expert timer headers,
timer arguments from device entry points and Host launch structures, timer
objects/pointers/assignments, probe calls and no-op marker macros. It also removes
Host timer dictionary/layout plumbing, Python attach calls and diagnostic-only
ST helpers/CLI. It does not substitute a dummy timer class or fake native layout.

The business packet/count layouts, expert/token ownership, Scalar operations,
DMA, event ordering, barriers, magic rotation, buffer reuse and output semantics
are retained. Removal operates on specific diagnostic statements and arguments;
an inline marker does not cause a surrounding wait or scope to be deleted.

Source manifests record exact transformations, file hashes, full projected token
checks and independent ordered Scalar/DMA/synchronization-call checks. Separate
clean wrapper symbols plus matching native/Python artifact identities reject
mixed source/package/Host/device launch layouts. Both arms have the same form
of one-time identity check outside profiling. Each wheel is checked against its
own frozen source fingerprint rather than another variant's checkout.

This source/CPU evidence cannot prove Bisheng optimization, device correctness
or performance. The target build and correctness checks remain mandatory before
performance results are accepted.

## Build all wheels before measuring

Use the current branch's project root and load the existing target Python/CANN
environment. Build/runtime dependencies must already exist; the helper does not
download Python packages or change shared Python. It creates a fresh group with
two independent source clones, three isolated venvs and three wheel directories.

Dependency initialization is restricted to SHMEM and its required submodules:
`git submodule update --init --recursive --depth 1 -- third-party/shmem`.
Do not fetch CATLASS or use an unscoped recursive submodule update/clone.

The SDK is cleaned and rebuilt **once** under source-normal/third-party/shmem.
All three wheels explicitly use that same SDK. Each wheel's native build tree
is cleaned before compiling, with compiler ccache reuse disabled. SDK metadata,
hashes and optional fields are not admission gates. The script publishes its
relative wheel map only after every complete wheel has built, installed and
passed Host-only identity checks. No device query or NPU test occurs in build.

```bash
NO_TIMER_RUN=ll_no_timer_ab_001
NO_TIMER_BUILD_OK=0
build_no_timer_ab() {
  git pull --ff-only || return $?
  git fetch origin 439d4597a52b657858217bac7fc1e965f22507f7 || return $?
  python tests/ll_stage_timer/no_timer_ab.py build --run-name "$NO_TIMER_RUN"
}
if build_no_timer_ab; then
  NO_TIMER_BUILD_OK=1
  echo 'All three full wheels and Host checks completed; no NPU work has run.'
else
  echo 'Build stopped. Preserve logs; terminal remains open.'
fi
```

Use a new name if the group exists. Add `--plan` for a read-only command plan.
All source, dependency, cache, log and output paths are derived from the project
and group. Previous groups and the separate fast reference deployment remain
unchanged. A compiler/linker failure stops without attempting device sampling.

## Run the two stages serially

The default seed is0. Stage1 compares replay-off/replay-clean on normal routes.
Stage2 compares replay-clean/stream-clean on normal and hotspot routes. Each
stage/route/seed block uses A-B-B-A; an optional second seed uses B-A-A-B.
The default is12 sequential ST processes, each covering all seven B values and
both dispatch and combine. No build or SDK replacement occurs during sampling.

Set `MASTER_ADDR`, `MASTER_PORT` and `SHMEM_PORT` for the allocated job first.
The16-device example below is for the user's target, not assistant execution.

```bash
run_no_timer_ab() {
  if [ -z "${MASTER_ADDR:-}" ] || [ -z "${MASTER_PORT:-}" ] || [ -z "${SHMEM_PORT:-}" ]; then
    echo 'Set MASTER_ADDR, MASTER_PORT and SHMEM_PORT for this job first.'
    return 1
  fi
  python tests/ll_stage_timer/no_timer_ab.py run --run-name "$NO_TIMER_RUN" \
    --stage all --seeds 0 \
    --devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
    --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" --shmem-port "$SHMEM_PORT"
}
if [ "${NO_TIMER_BUILD_OK:-0}" = 1 ] && run_no_timer_ab; then
  echo "Return only diagnostic-runs/$NO_TIMER_RUN/SEND_FIRST.txt"
else
  echo 'Build incomplete or measurement stopped. Preserve logs; terminal remains open.'
fi
```

To inspect the stages separately, use `--stage timer`, then a second invocation
with `--stage optimization` in the same build group. Neither selected run may
already exist unless explicitly using `--resume` for that same saved plan.
Analysis combines completed stage plans; it never claims a failed
or partially executed block is complete. `--seeds 0 2` is available for a fresh
matrix and doubles the default process count; do not mix or average seeds.

Fixed conditions: EP16, B1/2/4/8/16/24/32, M32/H3584/K8/E384, prequantized FP8
INT32-scale dispatch, **BF16 combine**, CPU sync0, correctness repeats3 and
repeat-combine2. Formal profiling is10 warmups/100 calls/per-rank tail50, with
Exp plus scalar all-reduce before every sample. Actual seed/routes are restored
after correctness checks and fingerprinted. Both arms use dispatch+combine.
Changing numbers to5/50/30 would not reproduce cbd04eb's old consecutive-call
protocol; that protocol is not mixed into this timer-removal comparison.

## Resume after a failed measurement

Update the controller checkout with `git pull --ff-only`, then repeat the same
`run` command with `--resume`. Do not repeat `build`: the existing SDK, three
wheels, venvs and frozen source clones are reused. The saved stage, seeds,
devices, master address and ports must match. This works with groups created
before `--resume` was introduced; no source projection or wheel rebuild is needed.

For `--stage all --seeds 0`, if the first hotspot/replay-clean process failed
after the two normal-route blocks, resume retains those eight validated runs
and launches the four hotspot runs after three fresh Host preflight checks.
If a failure occurred later inside a block, all four members of that incomplete
ABBA block are repeated. A complete block is retained only if its raw samples,
source/ABI identity and all four route fingerprints validate. Damaged completed
results stop recovery rather than silently selecting a new measurement.

From the same project root and target environment, retaining the job's original
`MASTER_ADDR`, `MASTER_PORT` and `SHMEM_PORT`:

```bash
resume_no_timer_ab() {
  git pull --ff-only || return $?
  if [ -z "${MASTER_ADDR:-}" ] || [ -z "${MASTER_PORT:-}" ] || [ -z "${SHMEM_PORT:-}" ]; then
    echo 'Set the same MASTER_ADDR, MASTER_PORT and SHMEM_PORT as the saved job.'
    return 1
  fi
  python tests/ll_stage_timer/no_timer_ab.py run --run-name ll_no_timer_ab_001 \
    --resume --stage all --seeds 0 \
    --devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
    --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" --shmem-port "$SHMEM_PORT"
}
if resume_no_timer_ab; then
  echo 'Return only diagnostic-runs/ll_no_timer_ab_001/SEND_FIRST.txt'
else
  echo 'Resume stopped. Existing and new logs remain; terminal remains open.'
fi
```

New attempts use `runs/resume001/...` and `logs/runs/resume001/...`, incrementing
on later explicit resumes. Old reports/logs are never moved or overwritten.
`run-history/<saved-plan>/resume001.json` preserves the prior status and the
retry decision. The final report identifies the selected attempt paths and the
interruption; superseded samples are excluded from statistics. No automatic
retry occurs. The error's final exit code alone does not diagnose its cause;
inspect the failed process log if the same failure recurs.

A fully completed `--resume` only regenerates analysis. A group lock rejects
concurrent controllers; caught Ctrl-C records `interrupted` and releases it.
An active state or a stale lock after a forced kill requires inspection of
the stopped job before recovery; the script does not erase that lock or stop
other processes automatically.

## Reading the result

Return only `SEND_FIRST.txt`, at most60,000 UTF-8 bytes including a CRLF margin.
The raw task arrays, source proofs, wheels, build logs and local summary JSON
stay in the group. The existing cumulative ten-file send budget applies.

Each table row retains A1/A2/B1/B2. Its primary value is computed separately
for each run: rank tail50 mean, then maximum over16 ranks. The two repeats of
one variant are averaged only after that maximum. Mean/p99 columns first pool
each run's16 rank tails, then average the two per-run statistics; they do not
pool different repetitions or seeds into a new percentile.

Negative B-minus-A means B was faster. Stage1 measures the combined effect of
removing instrumentation source and its parameters; it does not identify a
particular leftover Scalar instruction. Stage2 measures stream receive on the
same stripped baseline. Preserve visible repeat variation and the known hotspot
regression guards. A CPU check or a successful build is not evidence of a speedup.

Offline regeneration after completed runs:

```bash
python tests/ll_stage_timer/no_timer_ab.py analyze --run-name "$NO_TIMER_RUN"
```

Local validation completed1346 CPU regression cases, with100 environment skips
and8 existing POSIX fixture exclusions. A separate12-test local C++ run checked
all five selector identities, seven-mode Host dispatch, launch layouts and
bidirectional mixed-link rejection. These sets are reported separately; no
target CANN build, NPU correctness result or performance gain is claimed.
