# Full-dispatch ablation suite (experimental)

Branch `experiment/ll-g4-ablations`, algorithm base `197052c` (plain q, optimized
receiver destination metadata). This is separate from the earlier standalone
bandwidth probes. **CPU checks and CANN device-object compilation passed. Full-wheel linking and
EP16 device correctness are not yet validated. Run the correctness gates before
interpreting performance.**

The target is B=M256, H3584, K8, E384, prequantized FP8, int32 E8M0 scale packs,
do_cpu_sync0. Other dtype/scale combinations are rejected by the launcher for this
suite. Ordinary routes and weights remain real and reproducible in every mode.

## Experiments

| Mode | Input | Kernel behavior | Timing |
|---|---|---|---|
| baseline / 0 | Original rank/token-identifying payload | Full normal dispatch | OFF profiler kernel |
| fixed-payload / 1 | Zero FP8 and unity scales | Full normal dispatch, all real copies | OFF profiler kernel |
| preload / 2 | Same zero/unity payload | Initialize UB once, skip per-packet payload/scales GM input; keep route/q, pack, send, receive and relay | OFF profiler kernel |
| prepack / 3 | Same zero/unity payload | Also initialize packed output lanes once and skip per-packet payload Copy; still construct real wire tail and marker | OFF profiler kernel |
| ready-replay / 4 | Original payload | Full dispatch prefill without clearing packet flags; then replay receiver with all data already ready, suppress duplicate relay, finally clear flags | ON device-cycle envelope only |

First compare baseline and fixed-payload to identify content effects. Then compare
fixed-payload vs preload (input path), and preload vs prepack (packing path).
Preload/prepack initialization remains inside the kernel; differences are actual
changed-program costs, not ideal zero-cost hardware estimates. They do not skip
routing, q construction, counting, prefix calculation or communication.

Mode4 uses retained real incoming packets and the original task-to-slot table.
The first pass completes normal input/output/relay dependencies; no flags are
invented. All local AIV synchronize before replay. Replay resets pending indices,
reuses the built task table/prefix, rereads payload and writes local outputs. It
does not issue a second relay. Cleanup happens only after replay. The dedicated
`ABLATION_READY_RECV` duration includes pending reset and replay processing/drain,
but excludes first-pass setup and the local synchronization before its timer.
Ordinary stage metrics describe only the first pass. A cold receiver, task-building
cost or live-network relay cost cannot be inferred from this warm replay result.
Do not subtract its full profiler task (which contains both passes) from baseline.

Not implemented here: pre-generated full route descriptors, cross-rank frozen
sender replay, or a no-send full LL kernel. Those need separate state/identity
design. “No-send” is already available in the standalone micro suite, and must
not be achieved by deleting sends while leaving live receivers waiting.

## Isolation and correctness

Every mode has protocol `29000+100*mode+S`, compiled into Host/device and checked
on load. All ranks agree on mode, split, timer settings and wheel SHA before SHMEM
creation. Schema14 adds one replay metric; per-core timer allocation remains5120B.
The generic historical compare.py is not used; use this directory's compare.py.

Baseline/replay keep full payload correctness checks. Fixed/preloaded/prepacked
validate exact zero payload, unity scale bits, expected output counts/shapes and
cross-mode fingerprints for all valid destination rows, source weights, count
outputs and (when dedup enabled) token types/source masks/valid relay entries and
weights. Uninitialized reserved relay tail bytes are not hashed. Fingerprints are required for **all ranks**.
The baseline is necessary: constant payload alone cannot validate row identity.
Three correctness rounds change routes and reuse both protocol windows before
sampling. Never use --skip-check or combine with these diagnostic wheels.

## Prepare environment

Clone through Git into a new directory and load the target's existing CANN/Python
environment. Do not install onto the current production venv. From checkout root:

```bash
python -m venv --system-site-packages .venv-ablation-build
source .venv-ablation-build/bin/activate
python -m pip install -r requirements-build.txt
python -c 'import torch, torch_npu; print(torch.__version__, torch_npu.__version__)'
```

The build script creates a detached worktree for each mode under `.ablation/sources`,
initializes its locked recursive submodules and builds its own in-project SHMEM.
No SDK upgrade, system dependency install or driver change is performed. Every
variant retains all7 production dispatch translation units and O2/Ascend950.
This deliberately costs more initial build time to isolate variants. Already
existing complete variants are reused only after validation. With --resume,
incomplete directories remain untouched and a new isolated attempt is created.
For a new source revision use a fresh suite checkout.

### Network-limited hosts (including27): upload only

Do not run online dependency downloads there. On a networked machine, prepare a
complete checkout at the same locked submodule revisions, including recursive
dependencies. Export its Git objects without credentials/configuration:

```bash
python tests/ll_ablation/deps.py pack --checkout /path/to/complete/checkout \
  --bundle .ablation/offline-deps
```

The exporter rejects missing/dirty/wrong-revision/shallow dependencies. If needed,
unshallow on the networked machine before export. Upload the entire offline-deps
directory via SSH/SCP; do not put Git credentials or token files in the upload.
Then append `--offline-deps .ablation/offline-deps` to build.py or suite.py:

```bash
python tests/ll_ablation/build.py --mode preload --timer OFF --send-cores 30 \
  --prepare-sdk --offline-deps .ablation/offline-deps
```

Offline import checks SHA256 and every receiving gitlink, uses only local bundle
fetches, and rejects incomplete trees instead of falling back to the network.
Pip wheel building uses `PIP_NO_INDEX=1`. Reuse the existing target Python/CANN
environment; missing Python build packages must also be supplied through an
uploaded wheelhouse compatible with the target's Python/architecture.
SHMEM Ascend950 also needs nlohmann-json3.11.3 outside its Git submodule tree.
Prepare its single_include/nlohmann/json.hpp under SHMEM's3rdparty/json on the
networked machine; the exporter validates its version and includes the header
with a SHA256. Missing JSON is rejected, never downloaded on27. Offline builds
also restrict Git to the local file protocol.

## Compile first, launch later (recommended when the NPU is busy)

```bash
# Reuse already completed wheels, build every missing variant; no NPU checks/runs.
python3 tests/ll_ablation/suite.py --build-only --resume --send-cores 30

# Once all wheels are ready and the devices are idle, install and run all modes.
python3 tests/ll_ablation/suite.py \
  --devices "$ASCEND_RT_VISIBLE_DEVICES" --send-cores 30 --run-prefix bs256-ab-001
```

With --build, the order is now **all builds -> all installs -> serial experiments**.
--build-only performs only the first phase and needs neither devices nor run-prefix.
--resume reuses complete manifests only after identity/SHA256 checks. Missing
variants are built at the existing wheels' source commit, so a launcher-only update
need not rebuild the baseline or mix source revisions. Mixed revisions are rejected.
Interrupted builds without a valid manifest are preserved. --resume builds that
mode in a uniquely named source worktree and wheels/<mode>/attempt-* directory,
then publishes the canonical manifest only on success. Installation/run scripts
follow its relative wheel/source paths. Completed wheels are not recompiled.
Results are never skipped or overwritten: use a new prefix if a previous run already
created result directories. Occupancy checking remains unchanged before each run.

## One-command single-host EP16 suite

Use the **actual target device IDs and the same S/R as the88us baseline**. The
example S30/R26 is only appropriate if that is the configuration being compared.
The newly authorized26/27 hosts expose8 devices each, so do not run this16-device
command there. This command is for the user's16-device target host.

```bash
python tests/ll_ablation/suite.py --build --plan \
  --devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --send-cores 30 --run-prefix b256-ab-001

# After reviewing the plan, execute the same command without --plan.
python tests/ll_ablation/suite.py --build \
  --devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --send-cores 30 --run-prefix b256-ab-001
```

NPU occupancy is checked before each mode. Busy/unknown status stops the suite;
it never kills other jobs. No subsequent mode runs after a build, correctness or
runtime error. Completed directories remain available. Do not rerun --build into
existing directories; use individual commands to continue with existing wheels.

For normal modes:10warmups/100samples/tail50, with the existing Exp+scalar
all_reduce conditioning and profiler kernel statistics. Ready replay uses the
same stabilization and5 diagnostic captures with timer ON, no profiler comparison.

## Individual build / install / run

```bash
python tests/ll_ablation/build.py --mode preload --timer OFF \
  --send-cores 30 --prepare-sdk

python -m venv --system-site-packages .ablation/envs/preload-OFF-s30
source .ablation/envs/preload-OFF-s30/bin/activate
python tests/ll_ablation/install.py .ablation/wheels/preload-OFF-s30/manifest.json
python tests/ll_ablation/run.py --mode preload \
  --manifest .ablation/wheels/preload-OFF-s30/manifest.json \
  --devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --run-name b256-ab-002-preload
```

The installed interpreter must match the wheel's Python ABI. run.py checks mode,
protocol, timer setting and wheel digest, then delegates the original LL launcher.
Use a fresh run-name; existing results are rejected. Change shape or multi-host
settings in a copy of b256.json via --config. Original launcher restrictions
(16 contiguous local ranks, supported EP, explicit addresses/ports) still apply;
do not spoof EP16 with2 devices or use standalone peer mappings across hosts.

## Compare

```bash
python tests/ll_ablation/compare.py \
  --baseline .ablation/results/b256-ab-001-baseline \
  --fixed-payload .ablation/results/b256-ab-001-fixed-payload \
  --preload .ablation/results/b256-ab-001-preload \
  --prepack .ablation/results/b256-ab-001-prepack \
  --ready-replay .ablation/results/b256-ab-001-ready-replay \
  --output .ablation/b256-ab-001-comparison.json
```

Comparison rejects mismatched revision/SR/world/shape/routes/sampling or metadata.
For multiple hosts collect per-rank metadata and timer captures into one run tree
without overwriting different files. Normal latency uses max-rank mean profiler
kernel duration. Replay reports max-core duration per rank/iteration, retaining
ticks and frequency-based microseconds; no cross-rank clock subtraction.

88->70us requires18us reduction. If fixed-payload->prepack recovers much less,
payload handling alone cannot explain that target. If replay is short while the
live path is long, investigate data arrival/routing/communication overlap. These
are directional tests, not independent additive savings or a promise of70us.

## Build source guard

Builds use committed HEAD in isolated worktrees. Untracked files and modifications
inside a submodule at the pinned commit are left untouched and excluded, with an
explicit message and manifest record. They no longer block builds. Changes to
tracked files (staged or unstaged) or to a submodule commit still stop the build
and list exact paths, because those changes would not be included in HEAD.
No reset, clean, automatic stash or source overwrite is performed. Both --devices
and --device are accepted by suite.py; quote the visibility variable.

## Receiver40us follow-up

Use [RECEIVER_BREAKDOWN.md](RECEIVER_BREAKDOWN.md) and receiver_suite.py for the
new paired baseline/skip-flags/no-payload-output experiment. It builds one ON
wheel and preferentially reuses the old ready-replay build worktree/SDK/cache.
The original suite.py still runs only its five original modes.
