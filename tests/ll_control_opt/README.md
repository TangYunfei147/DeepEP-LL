# Control-path candidates: commit-by-commit A/B

`commits.json` is the ordered source manifest: baseline followed by four cumulative
optimization commits. Each wheel contains one committed revision, timer OFF,
logical G4 and S30/R26 by default. All wheels finish building before any device
run starts. Dispatch and combine are both tested.

| Label | Commit | Cumulative change |
|---|---|---|
| baseline | `197052c` | Production source corresponding to ablation mode0 |
| source-mask | `0a44d45` | Register accumulation, one source-mask UB store per token |
| pending-suffix | `4783fed` | Remove a fully ready pending suffix without UB self-copies |
| flag-vf | `868ddf8` | Fuse eight-block flag reduction and ready/layout masks in one VF |
| metadata-batch | `d00e87f` | Batch type/destination output in spare UB, retaining the lane count |

The last row includes all four optimizations. These are production candidates,
not skip-flag or no-payload ablations. The generic flag path remains for other
block counts. Metadata batching requires dedup, B>32, overlap enabled and enough
unused conservative UB budget; otherwise it uses the original per-packet path.
At35..39rows it adds608..640bytes/core, never beyond230KiB and never by reducing
send/receive lanes. FIRST_HOP forwarding and payload issue order are unchanged;
type/destination publication moves to the existing receive completion boundary.

The main metric is **`summary.csv` → `mean_kernel_us`**, separately for dispatch
and combine: the arithmetic mean of all ranks' last50 kernel samples. This is not
max-rank mean, host elapsed time, or the receiver replay timer. The underlying
`tests/elastic/test_ep_low_latency.py` keeps its defaults:10 warmups,100 measured
iterations, last50 per rank, Exp + scalar all_reduce, correctness enabled. FP8 is
the dispatch configuration; its combine input/output remains BF16.

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
  --resume --send-cores 30 --run-prefix b256-control-001
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
  --resume --send-cores 30 --run-prefix b256-control-001 --plan
```

In a multi-machine job, launch the same command and run prefix on every machine
with the appropriate `NODE_RANK`. A CPU-only readiness rendezvous waits until all
machines have built/installed their wheels, passed local binary checks and found
their selected devices idle. It first waits for preparation to finish everywhere,
then checks device occupancy and exchanges the idle status immediately before
launch; an early-finishing host does not rely on an occupancy check performed
while another host was still compiling. It compares the ordered commit list, current
commit/round, shape, split and run prefix before allowing the ST to start.
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
  --resume --send-cores 30 --run-prefix b256-control-001 --control-port 29602
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
  --run-only --send-cores 30 --run-prefix b256-control-001
```

All Python environments use project-local venvs with `--system-site-packages` and
the existing environment's Torch/CANN dependencies. Wheel installation uses
`--no-index --no-deps`. No packages are fetched from pip. Installed native,
actually loaded `libdeepep_kernels.so` (from `/proc/self/maps`), and loaded SDK
library hashes must match each machine's local wheel manifest before the NPU
launch. Across machines the source revision/configuration must match;
binary hashes are not assumed reproducible across toolchain installations.

## Native/SDK cache and offline dependencies

The default build workspace is `.tmp/ll-control-opt/source-s30`. It advances from
baseline to the final candidate with detached Git checkouts. All commits use the
same build/SDK directories, letting CMake depfiles reuse unchanged Host/HT/device
objects. Every new wheel runs `scripts/build_shmem.py` for SDK identity/hash
validation and enters the normal `pip wheel --no-build-isolation --no-deps`
build. It never skips native dependency checks based only on a source hash.
Cold preparation and individual build/package wall times are recorded separately
in each attempt's `timings.json`; compilation speed is not a performance result.

Default dependency preparation copies the local checkout's pinned initialized
repositories using local Git, plus its existing validated SDK and required
nlohmann header. The source checkout remains unchanged. If dependencies were
uploaded as an existing `tests/ll_ablation/deps.py` bundle, use:

```bash
python3 tests/ll_control_opt/suite.py \
  --build-only --resume --send-cores 30 --offline-deps .tmp/uploaded-deps
```

There is no Git/network fallback. If the default checkout lacks initialized
dependencies, initialize/upload them first. An uploaded dependency bundle carries
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

It must be at an ancestor of the first missing commit, have the same pinned
dependencies, and have no tracked edits or untracked source files. The primary
checkout is never accepted as the build workspace. Generated ablation headers
are not silently restored or deleted; an ablation worktree with such edits is
rejected. Cache reuse across this production/ablation boundary is deliberately
not automatic. No `git reset`, `git clean`, or overwrite of old wheels occurs.

`--resume` reuses a finished wheel only after checking its revision, split,
timer/group, wheel hash and SDK hashes. An incomplete attempt is retained and a
new attempt subdirectory is created. It does **not** skip, merge, or overwrite
old performance results. After a failed device run, use a fresh `--run-prefix`.
Locks prevent concurrent suite/cache mutation; if a process was killed, inspect
the recorded PID before manually removing its stale `.lock` file.

## Results and interpretation

By default node0 writes:

```text
results/ll-control-opt/b256-control-001/
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
| `baseline_delta_us/pct` | Current minus the first baseline commit |
| `parent_delta_us/pct` | Current minus the preceding optimization commit |

Negative deltas mean lower latency. Improvements are cumulative; a difference
between adjacent commits estimates that commit under this fixed sampling setup.
Use `--rounds 3` to repeat full commit sweeps without changing the ST's100/50
sampling definition. Results remain per round; fixed execution order can still
have temporal drift, so a single sweep does not establish a stable percent gain.

CPU plan/synthetic-cache tests do not demonstrate target-device correctness,
CANN linkage, or performance. Those require the generated full wheel/ST runs.
