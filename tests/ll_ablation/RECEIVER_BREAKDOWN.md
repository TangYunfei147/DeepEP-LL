# Receiver replay breakdown: one wheel, three measured arms

The previous ready replay was about40us. This experiment remeasures a common
baseline and two ablations in one ON wheel, rather than compiling three wheels.
It is restricted to EP16/B=M256/H3584/K8/E384/prequantized FP8/int32 scales/G4.
Production dispatch and the original five-mode suite remain separate.

## One command on the target

Update experiment/ll-g4-ablations, activate the same Python/CANN environment used
to build the existing wheels, then:

```bash
python3 tests/ll_ablation/receiver_suite.py \
  --devices "$ASCEND_RT_VISIBLE_DEVICES" \
  --send-cores 30 --run-prefix recv40-001
```

This builds or reuses **one** ready-breakdown ON wheel, installs it into a dedicated
venv, runs correctness/stabilization and9 diagnostic captures, then automatically
writes the comparison. It does not rebuild or rerun the old five-wheel suite.
All16device IDs must be explicit. Occupancy checking remains in place before the
device run. Use a fresh prefix; existing results are never overwritten.

`--plan` displays commands without building/running. `--build-only` prepares only
the wheel and needs no device list. `--fresh-build` avoids reuse of the existing
worktree. Fresh builds can use `--offline-deps <uploaded-directory>`; validated
cache reuse does not fetch dependencies or rebuild the SDK silently.

## Three arms

| Arm | Flag DMA/Sum/decode | Payload/scale output | Metadata output and cleanup |
|---|---|---|---|
| baseline | Original | Original | Original |
| skip-flags | Replaced by the saved layout/ready snapshot | Original | Original |
| no-payload-output | Original | Suppressed | Original |

The first pass is a complete, real dispatch. It records each task's observed
KeepFlags/Compact state in the unused high bit of the existing token-position
table. This shape's positions are below98304; Host and device reject other shapes.
Slot addressing masks that bit. No new UB tensor or change to the lane count is
introduced. The initial pass still performs all flag checks, payload copies and
relay operations and retains the input-window contents for replay.

Before **every** arm, flags are restored from this known-ready snapshot and all
local AIV synchronize, outside the timed envelope. Each arm resets pending indices,
uses the same priority ordering, loads the same real packets and retains existing
lane dependencies. All arms suppress duplicate relay. Each arm clears flags and
drains its work before finishing. No-payload-output retains local metadata writes;
the already-correct first-pass payload/scales remain in the output tensors.

Every arm's payload and scales are compared word-for-word against the retained
real packets **after its timer ends**, before another arm can overwrite results.
The Host additionally performs the original route/payload/count checks. This
avoids relying solely on the final output of a multi-arm invocation. Verification
and flag restoration warm memory outside timing; this is deliberately a paired
warm-replay experiment, not a cold-receiver measurement.

Order rotates over three launches using word1 of each core's existing512B local
magic slot. Session initialization zeroes this storage, and all cores update it
once per launch. Nine captures cover every first-arm order three times. No global
shared counter or cross-rank clock subtraction is introduced.

## Evidence and report

Mode5 has protocol29500+S. Schema15 uses121metrics with the unchanged5120B/core
carrier. It records three duration envelopes plus explicit operation counters and
per-arm verification errors. Counter encoding is value+1 so executed-zero and
missing instrumentation cannot be confused. The parser checks:

- skip-flags issued zero flag probes;
- no-payload-output issued zero payload/scale bytes;
- task-dependent counts agree across arms;
- each arm's word comparison found zero errors;
- all56cores/all16ranks/all iterations are present and order rotation is balanced.

The main result is:

```text
.ablation/results/recv40-001-receiver-comparison.json
```

It contains compact per-iteration/per-rank summaries and paired differences. Full
per-core data and rank samples are adjacent `.cores.csv` and `.rank_samples.csv`.
The raw schema15 captures remain in the run directory. Send the compact JSON first.

Use the **new baseline** in this report; do not subtract the previous40us from a
new arm. The main durations exclude restoration, alignment and validation, but
include pending reset, receive work, cleanup and drain. Differences include
pipeline effects and instrumentation-counter work, are not additive, and cannot
be applied directly to the93us full dispatch. No-output is a diagnostic only.

## Cache reuse

The command looks for `.ablation/wheels/ready-replay-ON-s30/manifest.json` (using
the requested S). It validates the old wheel, source revision, generated header,
tracked/untracked source changes, dependency pins and worktree ownership. It then
fast-forwards that **isolated** worktree to the current revision, changes only the
generated mode selection and records the new state. Its original absolute source
path, SDK and build directories stay in place, preserving native cache eligibility.

Old wheel files/manifests are retained and remain runnable with the unchanged SDK.
The old worktree's source HEAD/header now describe mode5; the migration is recorded
in `.tmp/receiver-cache-state.json`. A build lock rejects concurrent reuse. User
changes are not reset, stashed or overwritten; use --fresh-build if the donor has
unrelated changes. An unchanged second preparation does not rewrite the header.

The standard SDK identity/hash validation and CMake build/depfile checks still run.
Changed LL/Host objects must rebuild; unrelated cached objects can be reused. A
different Python/toolchain/SDK/flags can still select a new cache. Reusing a
worktree is not proof of a measured full-wheel cache hit; inspect verbose build.log
and build-timing.json. Missing caches naturally require compilation.

The legacy suite remains five modes and does not mix an old ready-replay wheel
with the new arms. The new launcher verifies current native inputs before reusing
an already completed mode5 wheel.
