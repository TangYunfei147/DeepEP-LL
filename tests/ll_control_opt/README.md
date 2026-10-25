# Control-path candidates: commit-by-commit A/B

> Current run: follow [LL_ONLY_FIX.md](LL_ONLY_FIX.md). Use
> `commits-topology-mte.json`, `DEEPEP_BUILD_HT=OFF` and the new prefix
> `b256-topology-mte-001`: three new-SDK candidates,18 ST calls.
> The earlier mixed-session/four-item experiment below is retained for provenance;
> do not reuse its samples as the corrected LL-only session results.

The earlier topology/SDK experiment is documented in
[TOPOLOGY_EXPERIMENTS.md](TOPOLOGY_EXPERIMENTS.md): a new-SDK baseline with the
measured `c1e4faa` kernel, an explicit old-SDK comparison, and two independent
receive-scheduling candidates. Select `commits-topology.json` explicitly.
The delivery branch's production kernel no longer defaults to `868ddf8`.

`commits.json` remains the historical default manifest for independent candidates based on
`868ddf8`: revert source-mask accumulation, reduce the flag batch from16 to8,
and prefetch flags across batches. Each candidate is compared with the declared
reference, not with the preceding candidate. The actual revisions and labels are
recorded in that manifest. Each wheel contains one committed revision, timer OFF,
logical G4 and S30/R26 by default. All wheels finish building before any device
run starts. Dispatch and combine are both tested.

| Default label | Commit | Independent change from868ddf8 |
|---|---|---|
| baseline | `868ddf8` | Original16-packet probe, source-mask accumulation retained |
| source-mask-revert | `85d05f4` | Restore original source-mask UB writes only |
| flag-batch8 | `1c29599` | Probe8packets, retain16-packet storage and the original lane budget |
| flag-prefetch16 | `c1e4faa` | Pair two16-packet snapshots; overlap the second flag probe with first-batch consumption |

Every candidate in this historical manifest has868ddf8 as its direct Git parent.
The delivery branch retains these commits so previous runs can be resumed with
the same manifest. The suite builds the exact manifest revision, not the
delivery branch's HEAD. Historical details and validation limits are in
[RECEIVE_CANDIDATES.md](RECEIVE_CANDIDATES.md).

The earlier cumulative experiment remains available through
`--commits tests/ll_control_opt/commits-cumulative.json`:

| Label | Commit | Cumulative change |
|---|---|---|
| baseline | `197052c` | Production source corresponding to ablation mode0 |
| source-mask | `0a44d45` | Register accumulation, one source-mask UB store per token |
| pending-suffix | `4783fed` | Remove a fully ready pending suffix without UB self-copies |
| flag-vf | `868ddf8` | Fuse eight-block flag reduction and ready/layout masks in one VF |
| metadata-batch | `d00e87f` | Batch type/destination output in spare UB, retaining the lane count |

In that historical manifest, the last row includes all four optimizations. These are production candidates,
not skip-flag or no-payload ablations. The generic flag path remains for other
block counts. Metadata batching requires dedup, B>32, overlap enabled and enough
unused conservative UB budget; otherwise it uses the original per-packet path.
At35..39rows it adds608..640bytes/core, never beyond230KiB and never by reducing
send/receive lanes. FIRST_HOP forwarding and payload issue order are unchanged;
type/destination publication moves to the existing receive completion boundary.
The new independent candidates do not inherit that metadata-batch commit.

## Explicit comparison references

Schema2 allows sibling commits. A top-level `comparison_base` supplies the
default reference; each entry may override it. A reference can be a Git revision
or a manifest label, and must itself be measured in the same manifest. The first
entry is still labelled `baseline` and references itself. For example:

```json
{
  "schema": 2,
  "comparison_base": "868ddf84cb77c51dcf15416e8f6053b4cf851858",
  "commits": [
    {"label": "baseline", "revision": "868ddf84cb77c51dcf15416e8f6053b4cf851858"},
    {"label": "candidate", "revision": "<candidate-commit>", "comparison_base": "baseline"}
  ]
}
```

Use the checked-in manifest for execution; the example contains a placeholder.
Legacy schema1 manifests retain their cumulative ancestor-order validation and
compare each candidate with the preceding manifest entry. Adding
`comparison_base` to a schema1 manifest is rejected instead of silently ignoring it.

The main metric is **`summary.csv` → `mean_kernel_us`**, separately for dispatch
and combine: the arithmetic mean of all ranks' last50 kernel samples. This is not
max-rank mean, host elapsed time, or the receiver replay timer. The underlying
`tests/elastic/test_ep_low_latency.py` keeps its defaults:10 warmups,100 measured
iterations, last50 per rank, Exp + scalar all_reduce, correctness enabled. FP8 is
the dispatch configuration; its combine input/output remains BF16.

## Multiple uniform-routing seeds and SDK comparisons

For the complete Git/SDK preparation, topology assumptions, CPU accounting and
current experiment commands, start with [TOPOLOGY_EXPERIMENTS.md](TOPOLOGY_EXPERIMENTS.md).

The old default `commits.json` and historical `commits-cumulative.json` remain
unchanged so their saved runs remain resumable. Use the separate topology/SDK
manifest explicitly:

```bash
python3 tests/ll_control_opt/suite.py \
  --commits tests/ll_control_opt/commits-topology.json \
  --resume --send-cores 30 --seeds 0 17 101 --rounds 2 --order alternating \
  --run-prefix b256-topology-001
```

This manifest compares the old SDK with the new SDK using the same `c1e4faa`
kernel, and compares the new kernel candidates with their explicit new-SDK
reference. Follow its recorded `comparison_base`; adjacent manifest rows need
not be parent/child changes. Every result row records `dependency_id`,
`shmem_revision`, SDK manifest/library hashes and the corresponding reference
fields. An old-SDK row minus a new-SDK reference is an old-minus-new difference,
not a new-minus-old speedup.

`--seeds` defaults to `0`. Routing remains the ST's uniform `normal` route; each
requested seed is passed explicitly through `--seed`. The original three
correctness routes are preserved, so performance uses the last routing table:
**requested seed + 2**. The example therefore measures routing seeds2,19,103.
`seed`/`requested_seed` and `measured_route_seed` are both saved. Warmup10,
collection100, per-rank tail50, FP8 dispatch and BF16 combine remain unchanged.

For every `(round, seed)` the suite runs all candidates and compares only with
the same-seed reference from that round. Odd rounds use manifest order; even
rounds reverse the candidate order within each seed. Wheels are still built and
installed only once per commit, regardless of the number of seeds. There is no
automatic pooling or subtraction across seeds.

The single default seed0 retains `<label>/round-NNN`. Other seed selections use
`<label>/seed-S/round-NNN`. Single-node `--resume-results` includes seed in each
result key; old plans without a seed field or explicit `--seed` are seed0.
Keep the same ordered seed list when resuming; a different seed selection needs
a new run prefix. Increasing rounds remains supported.

Each new ST invocation records a new `*.node-N.build.json` sidecar atomically
before launch. It binds the tested source/dependency revision to the validated
native/kernel/SDK hashes. Resume reads this saved metadata and does not require
the old wheel or SDK binaries to remain installed. Historical complete results
without the sidecar retain `sdk_identity_source=legacy-unavailable` and null
actual SDK hashes; the current SDK is never used to fill in missing historical
provenance. A result for one source revision cannot be reused for the SDK-update
revision, even when the kernel code is identical.

## One command on each participating machine

Load the existing Python/CANN environment first, for example the site's
`source /home/tyf/0env.sh`. Every machine must have the same branch, commit manifest,
S/R split and absolute project path, as required by the existing ST run-directory
broadcast. There is no automatic SSH, dependency download, or system installation.

Set the deployment variables already used by the existing ST command:

```bash
export NNODES=1                   # use the real host count:1,2,4,8
export NODE_RANK=0                # distinct per machine:0..NNODES-1
export MASTER_ADDR=192.0.2.1      # replace with the reachable first host
export MASTER_PORT=29600
export SHMEM_PORT=29601
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

python3 tests/ll_control_opt/suite.py \
  --resume --send-cores 30 --rounds 4 --order alternating --run-prefix b256-independent-001
```

`--devices "$ASCEND_RT_VISIBLE_DEVICES"` is optional when that variable is set.
Exactly16 distinct local device IDs are required. All four visibility variables
are set to this list. Before every ST launch, all selected devices must be
reported idle by `npu-smi info`; otherwise the suite stops and does not terminate
any processes. Unrecognized occupancy output is also treated as busy.

Each commit executes the user's ST shape and endpoint arguments:

```bash
python tests/elastic/test_ep_low_latency.py \
  --num-processes 16 --num-nodes "$NNODES" --node-rank "$NODE_RANK" \
  --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" --shmem-port "$SHMEM_PORT" \
  --num-tokens 256 --num-max-tokens-per-rank 256 \
  --hidden 3584 --num-topk 8 --num-experts 384 --dtype fp8 --do-cpu-sync 0 \
  --output-dir "$PWD/results/$CASE_NAME" --run-name "$RUN_NAME"
```

There is no `--dispatch-only`, timer capture, reduced warmup, or shortened sample count.
The plan prints every exact argument before execution:

```bash
python3 tests/ll_control_opt/suite.py \
  --resume --send-cores 30 --rounds 4 --order alternating --run-prefix b256-independent-001 --plan
```

In a multi-machine job, launch the same command and run prefix on every machine
with the appropriate `NODE_RANK`. A CPU-only readiness rendezvous waits until all
machines have built/installed their wheels, passed local binary checks and found
their selected devices idle. It first waits for preparation to finish everywhere,
then checks device occupancy and exchanges the idle status immediately before
launch; an early-finishing host does not rely on an occupancy check performed
while another host was still compiling. It compares the ordered commit list, current
commit/round, execution order/position, shape, split and run prefix before allowing the ST to start.
The default wait limit is3600 seconds (`--ready-timeout`); no NPU is initialized
by this rendezvous. Existing ST distributed synchronization and connection
timeouts then apply normally. Runtime commit identity is supplied through
`DEEPEP_LL_CONTROL_REVISION`; the ST capability exchange checks it across ranks.

The readiness listener uses `--control-port` on the first host. If omitted, its
deterministic default is the first port after `MASTER_PORT` excluding
`SHMEM_PORT`; `--plan` shows the chosen port. For the example above it is29602.
This port must be reachable from the other machines. An occupied port fails;
the script never probes alternate ports or kills its owner. To select one:

```bash
python3 tests/ll_control_opt/suite.py \
  --resume --send-cores 30 --rounds 4 --order alternating \
  --run-prefix b256-independent-001 --control-port 29602
```

Single-machine runs create no readiness socket and require no additional port.
A reported idle-check failure is relayed to every prepared host and prevents
the ST launch. If a host exits before connecting, the other hosts stop at the
readiness timeout; the suite does not terminate unrelated processes.

## Build everything first, then launch together

This lets compilation finish separately before the device allocation is available:

```bash
# All machines; does not require device/network launch variables or use an NPU.
python3 tests/ll_control_opt/suite.py \
  --build-only --resume --send-cores 30

# After all machines finish building, execute this on each machine.
python3 tests/ll_control_opt/suite.py \
  --run-only --send-cores 30 --rounds 4 --order alternating --run-prefix b256-independent-001
```

Candidate environments are project-local venvs with `--system-site-packages`.
This exposes the selected Python's base installation, not packages installed
only in an already activated outer venv. Use a verified base/Conda Python that
provides the matching Torch/CANN dependencies, or explicitly install the matching
Torch/torch_npu packages into the candidate environments before running them.
Wheel installation uses
`--no-index --no-deps`. No packages are fetched from pip. Installed native,
actually loaded `libdeepep_kernels.so` (from `/proc/self/maps`), and loaded SDK
library hashes must match each machine's local wheel manifest before the NPU
launch. Across machines the source revision/configuration must match;
binary hashes are not assumed reproducible across toolchain installations.

## Native/SDK cache and offline dependencies

Default build workspaces are
`.tmp/ll-control-opt/source-s30-deps-<dependency-id>`, where the ID hashes the
selected commit's dependency lock and Git pins. Old/new SDK identities use
separate source and SDK directories. The former `.tmp/ll-control-opt/source-s30`
is reused only when its committed dependency identity matches; otherwise it is
preserved. A matching workspace switches
between the selected commits, including siblings or an earlier baseline, using
detached Git checkouts. All commits use the
same build/SDK directories, letting CMake depfiles reuse unchanged Host/HT/device
objects. Every new wheel runs `scripts/build_shmem.py` for SDK identity/hash
validation and enters the normal `pip wheel --no-build-isolation --no-deps`
build. It never skips native dependency checks based only on a source hash.
Cold preparation and individual build/package wall times are recorded separately
in each attempt's `timings.json`; compilation speed is not a performance result.

Dependency preparation selects only `third-party/shmem` and its actual nested
Git dependencies. CATLASS and its test/DSL repositories are optional MegaMoE
dependencies; this core LL suite does not copy or require them. The main
checkout remains unchanged. Existing SHMEM source and SDK in the build worktree
are reused first; otherwise the builder uses the main checkout or matching
same-repository worktrees referenced by `.ablation/wheels/*/manifest.json` and
previous control-suite manifests. Cached kernel sources/HEADs are never changed.
Pinned source identity, tracked edits and the standard complete SDK validation
remain mandatory. A manifest's presence alone never qualifies an SDK.

An existing SDK does not require the `nlohmann/json.hpp` source asset: that file
is required, at version3.11.3, only when preparing a cold SDK. Neither an invalid
SDK nor an incomplete existing install is silently overwritten or rebuilt.
If dependencies were
uploaded as an existing `tests/ll_ablation/deps.py` bundle, use:

```bash
python3 tests/ll_control_opt/suite.py \
  --build-only --resume --send-cores 30 --offline-deps .tmp/uploaded-deps
```

There is no Git/network fallback. If neither the checkout nor registered local
caches contain pinned SHMEM, initialize/upload it first. Extra CATLASS entries in
an offline bundle are ignored; required SHMEM pins and hashes are still checked.
An uploaded dependency bundle carries
source/header assets, not necessarily an SDK; first SDK preparation may therefore
need a local compile. Existing SDK identity mismatches fail rather than trigger a
destructive rebuild.

An existing **clean, disposable production build worktree of the same Git
repository** can be explicitly reused. This option authorizes moving that
worktree's HEAD through the selected commits (it may detach an attached branch;
the branch ref itself remains unchanged):

```bash
python3 tests/ll_control_opt/suite.py \
  --build-only --resume --send-cores 30 \
  --build-workspace .tmp/existing-production-worktree
```

It must belong to the same Git repository, have the same dependency-lock content
and pinned Git dependencies, and have no tracked edits or untracked source files.
The previous HEAD need not be an ancestor of the next candidate. The primary
checkout is never accepted as the build workspace. Generated ablation headers
are not silently restored or deleted; an ablation worktree with such edits is
rejected as a native build workspace. Copying only its pinned SHMEM source/SDK
is permitted without changing that worktree. No `git reset`, `git clean`, or
overwrite of old wheels occurs.

`--resume` reuses a finished wheel only after checking its revision, split,
timer/group, wheel hash and SDK hashes. An incomplete build attempt is retained
and a new build-attempt subdirectory is created. Performance-result continuation
is selected separately with `--resume-results`, described below.
For the former CATLASS preparation failure, update the scripts and repeat the
original command with `--resume`; no performance run had started, so the same
run prefix and the existing unfinished build worktree can be retained.
Locks prevent concurrent suite/cache mutation. `--resume-results` can recover a
suite lock only when its recorded local process is confirmed dead; an active,
unverifiable, malformed or foreign-host lock is retained and rejected. It never
terminates a process. The adjacent `.guard` file is a persistent operating-system
lock target; do not delete it to bypass an active owner. Build locks retain
their explicit recovery policy.

## Continue an interrupted performance run

This continuation feature is for a single machine (`--num-nodes 1 --node-rank 0`).
Update the scripts, then repeat the interrupted command with
**the same `--output-dir` and `--run-prefix`**, adding
`--resume-results`. Existing candidate wheels do not need rebuilding for this
Python-only change. For the independent receive experiment:

```bash
python tests/ll_control_opt/suite.py \
  --run-only --resume-results --send-cores 30 \
  --rounds 4 --order alternating \
  --devices "$ASCEND_RT_VISIBLE_DEVICES" \
  --num-nodes 1 --node-rank 0 \
  --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
  --shmem-port "$SHMEM_PORT" \
  --output-dir "$PWD/results/ll-control-opt" \
  --run-prefix b256-rx868-001
```

Use your interrupted run's actual prefix/path if different. Keep its commit
manifest, shape, sampling, S/R split, host count and execution order. Round count
can stay the same or increase; it cannot shrink. Network endpoint ports may be
changed. New saved plans also check the selected devices. Legacy plans
did not save device visibility: continuation validates the available identity
and records that the legacy device mapping is unavailable.

Node0 validates each `(candidate, round)` independently using both `summary.json`
and `summary.csv`: correctness must pass, dispatch and combine must both be
complete, all ranks must provide100valid raw samples, and the last50pooled mean
must match. A missing `.done.json` does not discard a complete valid summary;
the old process may have been killed after saving it. A `.done.json` alone is
never enough. No partial iteration samples are appended to a new measurement.

Completed items are skipped. A missing, partial or invalid item is rerun in full under a
fresh `round-NNN.attempt-<id>` directory. Original directories, logs and summaries
remain untouched. Multiple valid completed attempts for one item are rejected
as ambiguous, rather than selecting whichever happens to be faster or newer.
Only needed candidates require wheel verification/installation and idle-device
checks. If every requested item is complete, continuation performs no build,
installation, binary preflight or NPU occupancy check.

Each continuation writes a separate session under:

```text
<output-dir>/<run-prefix>/sessions/<session-id>/
  suite-plan.node-0.json
  comparison.csv
  comparison.json
```

The script prints the new comparison path. Existing top-level comparison files
are preserved. The new comparison includes reused and newly measured items,
with `result_status` and `source_directory` on each row. Only complete rounds
are compared; references still come from the same numbered round. A resumed
round can span separate measurement sessions, so its ordering and timing gaps
must be considered when interpreting small differences. Original per-run raw
data remain the source of every reported mean.

If a complete attempt survives but its creation-session plan is unreadable, its
data still undergo full summary validation. The report marks its session as
`unknown`, device mapping unavailable, and cross-session comparison unknown.
The damaged plan remains on disk and is listed in `ignored_session_plans`.
An unreadable original plan prevents continuation because experiment identity
cannot be established. New plans and comparison files are published atomically.

## Results and interpretation

By default node0 writes:

```text
results/ll-control-opt/b256-independent-001/
  suite-plan.node-0.json
  baseline/round-001/summary.csv
  baseline/round-001/summary.json
  <candidate-label>/round-001/summary.csv
  comparison.csv
  comparison.json
```

The source ST preserves original profiler exports, raw samples and correctness
results. Nonzero nodes do not search for node0's aggregate summary; they retain
local completion/occupancy records and profiler files. The comparison refuses
missing ranks, fewer than100 raw samples, a different tail, mixed shapes/routes,
wrong source/core identity, timer ON, and inconsistent JSON/CSV/raw-tail means.
Each ST stdout/stderr stream is also retained in
`<label>/round-001.node-N.log` while remaining visible in the terminal.

Builds explicitly use Release, Ascend950, timer OFF and disable unrelated
MegaMoE/fused experimental build options. Inherited `RANK`, `WORLD_SIZE`,
`LOCAL_RANK` and `LOCAL_WORLD_SIZE` are removed so an old launcher environment
cannot override the suite's explicit16-process/node topology.

`comparison.csv` contains one row per operation/commit/round:

| Column | Meaning |
|---|---|
| `mean_kernel_us` | User-requested `summary.csv` mean |
| `baseline_label/revision`, `baseline_delta_us/pct` | Current minus the manifest's first baseline entry in the same round |
| `reference_label/revision`, `reference_delta_us/pct` | Current minus the explicitly resolved reference in the same round |
| `parent_delta_us/pct` | Legacy cumulative manifests only; empty for independent candidates |
| `previous_manifest_label/revision`, `previous_manifest_delta_us/pct` | Descriptive difference from the preceding manifest row; not a causal estimate and not necessarily the previous execution |
| `execution_position` | Actual1-based position within that round |

Negative deltas mean lower latency. Independent candidates have no inferred
parent-to-child performance attribution: `reference_delta` is the primary
candidate-versus-control comparison. Rows remain in manifest order regardless
of execution order, and reference lookup uses the manifest labels/revisions.

`--order alternating` is the default: odd rounds run baseline,A,B,C and even
rounds run C,B,A,baseline. `--order manifest` repeats the fixed manifest order.
Building always follows manifest order and reuses each completed wheel; reversing
the performance execution does not trigger additional builds. The default is
still one round; the commands above explicitly request four rounds so each
direction occurs twice. All rounds retain the ST's10/100/last50 sampling definition.
Alternation reduces execution-position imbalance but does not eliminate temporal
drift or predecessor effects; a single sweep does not establish a stable gain.
Incomplete rounds are rejected rather than compared with a baseline from another round.

CPU plan/synthetic-cache tests do not demonstrate target-device correctness,
CANN linkage, or performance. Those require the generated full wheel/ST runs.

Historical cumulative delivery validation:76CPU tests passed on27, including real
C++ semantic mocks and the multi-node readiness handshake exercised on loopback.
Final candidate source d00e87f compiled and linked complete CANN9.3.0/O2/S30R26
OFF and ON wheels. A real no-op OFF wheel rebuild took1.578s without changing any
native object; editing a device header rebuilt the7dependent LL objects in38.216s.
These checks did not run NPU kernels or establish speedups. Full evidence and
log paths are recorded in [BUILD_DISPATCH_SPLIT.md](../../docs/BUILD_DISPATCH_SPLIT.md).

The independent-manifest update passed58local CPU tests: explicit-reference
validation, forward/reverse planning, same-round comparisons, sibling/parent
worktree switching, repeated cache reuse, dirty/dependency/corrupt-wheel rejection,
and readiness handshake/error propagation. The Windows sandbox blocked local Git
IPC; those tests ran outside it in a short project-local temporary directory.
This update has no new Linux/CANN or NPU validation; historical cumulative wheel
results do not validate the independent kernels.

For a new SDK, first prepare/upload its pinned source and prepare that SDK in a
project-local checkout with the standard `python scripts/build_shmem.py` workflow.
If that checkout already contains an old SDK, the standard preparer refuses a
silent replacement. Use a fresh checkout/install area, or explicitly choose
`--rebuild` only after stopping users of that particular SDK. The suite never
adds `--rebuild` itself. It preserves old suite SDK caches, including `source-s30`,
and never copies a known other-revision SDK into a new dependency cache.

Single-machine continuation validation:127local regression checks passed across
the suite, dependency cache, result continuation and lock recovery. After the
final provenance fix, all49result-continuation checks passed (these batches
overlap). Coverage includes actual local Git/filesystem writes, a child exiting
without cleanup, competing lock owners, old summaries without done markers,
interrupted partial rounds, repeated continuation, round extension, all-results
reuse with no package/device operations, corrupt summaries and preserved original
artifacts. Device/build effects in continuation tests are explicit recording
fakes. No SSH, CANN compilation or NPU sampling was performed for this script update.
