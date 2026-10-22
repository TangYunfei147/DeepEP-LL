# Receiver replay breakdown: one wheel, three measured arms

The three arms are a conditioned receive-replay diagnostic. They do not divide
the production receive time or the full dispatch time into independent costs.
The audit and proposed controlled follow-up are in
[EXPERIMENT_AUDIT.md](EXPERIMENT_AUDIT.md). The current kernel is unchanged by the
offline audit/export update.

## Correctness fix after recv40-001

The target's initial dispatch.check failed with four FP8 elements at offsets
480..483 (absolute error7 and relative error1.46875). A CPU byte-level regression
reproduces these exact errors when the float Compact marker1.0 overwrites the
compact payload. Replay suppressed relay issue but still performed relay header
and marker preparation. Both are now guarded by the same live-forward condition.
The first real dispatch keeps its original forwarding behavior. This identifies
an unsafe replay write; the precise device path that selected the compact view
is still unexplained. Subsequent payload/scale checks passing does not explain
that layout/type combination or prove the three arms followed identical layouts
and packet order. Do not use the failed run for performance.

After updating Git, rerun the same one-command suite with a new result prefix,
for example `--run-prefix recv40-002`. When an existing completed wheel has stale
native sources, the suite selects a `-rev-<12hex>` artifact directory and retains
the old wheel and manifest. It reuses the same validated ready-replay worktree
and dependency-tracked caches. No deletion or fresh checkout is necessary.
All three arms must pass correctness again; compare against their new paired
baseline because removing replay-only forwarding preparation changes the work.

The previous ready replay was about40us under a different replay implementation.
This experiment remeasures a common
baseline and two ablations in one ON wheel, rather than compiling three wheels.
It is restricted to EP16/B=M256/H3584/K8/E384/prequantized FP8/int32 scales/G4.
Production dispatch and the original five-mode suite remain separate.

## One command on the target

Update experiment/ll-g4-ablations, activate the same Python/CANN environment used
to build the existing wheels, then:

```bash
python3 tests/ll_ablation/receiver_suite.py \
  --devices "$ASCEND_RT_VISIBLE_DEVICES" \
  --send-cores 30 --run-prefix recv40-003
```

This builds or reuses **one** ready-breakdown ON wheel, installs it into a dedicated
venv, runs correctness/stabilization and9 diagnostic captures, then automatically
writes the comparison and partitioned JSON export. It does not rebuild or rerun
the old five-wheel suite. This command retains the current three-arms-per-kernel
design and its limitations; the redesigned experiment below is not implemented.
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

The first pass performs real sends, receives, outputs and relays while delaying
flag cleanup for replay. It also records each task's observed
KeepFlags/Compact state in the unused high bit of the existing token-position
table. This shape's positions are below98304; Host and device reject other shapes.
Slot addressing masks that bit. No new UB tensor or change to the lane count is
introduced. The initial pass still performs all flag checks, payload copies and
relay operations and retains the input-window contents for replay.

Before **every** arm, flags are restored from this known-ready snapshot and all
local AIV synchronize, outside the timed envelope. Each arm resets pending indices,
uses the same priority algorithm and retains existing lane dependencies. However,
the parser does not verify that decoded layouts or the actual processing order
match across arms. All arms suppress duplicate relay. Each arm clears flags and
drains its work before finishing. No-payload-output retains local metadata writes;
payload/scales from the prefill or a preceding arm remain in the output tensors.

Every arm's payload and scales are compared word-for-word against the retained
real packets **after its timer ends**, before another arm can overwrite results.
The Host additionally performs the original payload/count checks and metadata
fingerprinting after the entire three-arm invocation. It does not independently
check the prefill before replay, and per-arm verification does not check metadata.
For no-output, a successful comparison means retained output still matches the
packet reference; it is not a proof that the arm independently produced output.
Both references also depend on the retained packet-to-slot mapping. See the audit
for sentinel, independent reference and metadata checks needed in a follow-up.

Verification and flag restoration change memory state outside timing. The first
arm follows real dispatch; subsequent arms additionally follow a full packet and
output readback plus scalar comparisons. Their cache and pipeline histories are
different. This is not a cold-receiver measurement.

Order rotates over three launches using word1 of each core's existing512B local
magic slot. Session initialization zeroes this storage, and all cores update it
once per launch. Nine captures cover every first-arm order three times. No global
shared counter or cross-rank clock subtraction is introduced.
Only the cyclic orders ABC/BCA/CAB are used, so the predecessor relationship is
fixed. Balanced position counts cannot remove arbitrary arm-by-position or
predecessor interactions; it is not sufficient evidence of causal isolation.

## Evidence and report

Mode5 has protocol29500+S. Schema15 uses121metrics with the unchanged5120B/core
carrier. It records three duration envelopes plus explicit operation counters and
per-arm verification errors. Counter encoding is value+1 so executed-zero and
missing instrumentation cannot be confused. The parser checks:

- skip-flags issued zero flag probes;
- no-payload-output issued zero payload/scale bytes;
- output-byte and flag-batch counts agree where expected;
- each arm's word comparison found zero errors;
- all56cores/all16ranks/all requested iterations are present and order rotation is balanced;
- total output rows match16*256*8, with exactly one probe batch per16ready tasks.

These gates do not check matching KeepFlags/Compact classifications, actual
packet order, per-arm metadata, or independent prefill correctness. Both receive
layouts can yield correct output with different DMA work, so matching payloads
and counts alone cannot establish an otherwise identical ablation.

The main result is:

```text
.ablation/results/recv40-001-receiver-comparison.json
```

It contains per-iteration/per-rank summaries and observed differences. Full
per-core data and rank samples are adjacent `.cores.csv` and `.rank_samples.csv`.
The raw schema15 captures remain in the run directory. The offline exporter can
regroup existing captures and produce bounded JSON parts without a wheel update:

```bash
python3 tests/ll_ablation/receiver_export.py \
  .ablation/results/recv40-002-ready-breakdown \
  --output-dir .ablation/results/recv40-002-export
```

Use a new output directory. Every JSON part is ASCII and no larger than60000
bytes/characters. Keep the part identifiers and send the parts separately. The
updated receiver suite also runs this export after reporting. This offline path
does not start devices or modify the recorded input captures.

Send `000-index.json`, followed by the files listed in its `send_first` field
(summary and rank evidence). Retain the core parts for follow-up questions.
Each file carries a dataset ID, part index and record checksum; the index also
records exact bytes/characters and file SHA256. Core parts retain every measured
core's three raw durations and operation counters without sampling. Optional
sampled timestamps are included only where actually recorded. Very large single
records are split into numbered JSON-text fragments without truncation.

The exporter verifies the complete package before returning. To recheck later:

```bash
python3 tests/ll_ablation/receiver_export.py \
  .ablation/results/recv40-002-export --verify
```

Use the **new baseline** in this report; do not subtract the previous40us from a
new arm. The main durations exclude restoration, alignment and validation, but
include pending reset, receive work, cleanup and drain. Differences include
pipeline effects and instrumentation-counter work, are not additive, and cannot
be applied directly to the93us full dispatch. Different statistics are also in
use: full dispatch takes the maximum of per-rank mean kernel times; the original
replay report takes the mean of per-iteration maximum core/rank durations. The
maxima may belong to different cores/ranks across arms. No-output is a diagnostic
only. Position-stratified and same-core comparisons improve description of the
existing data but cannot repair the experiment's unequal prior states.

Historical captures label the configured seed, while sampling retains the last
correctness round's routes. With default correctness_repeats3, the effective
route seed is configured seed+2. Report that distinction instead of silently
changing the stored seed. An absent correctness configuration makes this seed
derivation uncertain and must be recorded as such.

## Cache reuse

The command looks for `.ablation/wheels/ready-replay-ON-s30/manifest.json` (using
the requested S). It validates the old wheel, source revision, generated header,
tracked/untracked source changes, dependency pins and worktree ownership. It then
fast-forwards that **isolated** worktree to the current revision, changes only the
generated mode selection and records the new state. Its original absolute source
path, SDK and build directories stay in place, preserving native cache eligibility.

Old wheel files/manifests are retained and remain runnable with the unchanged SDK.
Do not delete the donor source directory: the new wheel also uses its SDK path.
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

## Validation history and remaining gaps

Before the marker guard fix, native source c300ce9: on27, all7device entries compiled with mode5 OFF/ON
in27s/43s (O2/dav-3510, jobs4). Mode5 explicitly rejects non-prequantized input
entries; the original modes retain their supported dtypes. Host syntax passed.

A real CMake object-cache check at S30/R26, prequantized FP8 ON, measured32.897s
cold,0.029s unchanged,33.032s after a header edit. Object timestamps confirmed
no-op reuse and necessary dependency rebuild. This is one device object, not
a full-wheel speedup measurement. Git/worktree tests additionally verify cached
objects are preserved, header mtime is unchanged on a no-op, locks reject concurrent
reuse and user edits are rejected; paths containing spaces are covered.

CMake3.31.6 was uploaded to the project-local .tools directory for the check; no
external access or system install on27. Full-wheel preflight there is blocked by
the absence of an installed SHMEM SDK (only locked source headers are available).
No EP16 device run or full-wheel link was performed in that validation environment.
The user subsequently supplied target recv40-002 results after the marker guard
fix. Their existing report gates passed; the additional causal/layout checks
identified above were not collected and cannot be claimed from those results.

Only local AIV synchronization is used, not a global cross-rank quiet barrier.
Rotation balances position counts, but its ability to cancel bias depends on
assumptions the current data do not establish. Replay remains a conditioned
diagnostic with different cache history from live dispatch. A follow-up should
first add A/A controls and per-arm layout/order evidence, then use one measured
arm per invocation with identical preparation. Those changes require new kernel
work and target captures; they are proposals, not part of this offline update.

Historical local regression before the marker guard:74passed/2compiler-discovery skips. NPU operation counts and
per-arm errors are mandatory gates in the target comparison, not inferred from
final output alone.
