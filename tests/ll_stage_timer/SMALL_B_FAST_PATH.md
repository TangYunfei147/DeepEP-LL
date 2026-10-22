# M<=32 / B<=32 direct dispatch candidate

Branch `kernel/ll-g4-small-b`, based on197052c (plain q table, receiver-header
optimization). Activation requires BOTH `!enableDedup` (M<=32) and actual B<=32.
B<=32 with M>32 follows the existing dedup path. B>32 non-dedup is unchanged.

Sender candidate:

- Return before setup for a zero-route shard. With K>=1, an empty route shard
  cannot own source-token weights; count publication remains in Counts, before
  sender Process. No send lane/event was created, so Retire has no lanes to wait.
- B1: send q=0 directly, without VF histogram or q-table construction. The legacy
  convention sets q=0 for every route index<K, including repeated expert IDs.
- B2/4: build only the owned q entries from their short original Scalar prefixes.
  Preserve full uint32 q and first-token behavior. B8/16/32 keep the VF seed path.
- Initialize/use no more send lanes than owned packets. Existing overlap pool
  budgets remain conservative; lane event IDs and per-core packet order are kept.

No input-buffer clearing was removed; quantization padding remains protected.
S/R, count participation, R0 prefix, wire flags, receive/output order and combine
are unchanged. No new mode/template dimension or kernel launch was added.
Implementation identity is `g4-small-b`, core protocol19000+S and timer schema13.
Install the matching full wheel on every rank; a new SDK rebuild is not required
solely for this algorithm change.

CPU checks execute the production tiny-q helper against an independent prefix
reference, with repeated experts, split token shards, all target smallB values,
K1/3/8/32, and multiple S counts. Static digest guards preserve dedup send/q/tail
methods, the receiver, main dispatcher, overlap plans and core configuration.
These are source/CPU checks, not proof of identical dedup machine code or speed.

Target A/B: start from197052c, then sender-only commit, then count candidate.
Use B1/2/4/8/16/32 with M32 and fixed EP/H/K/E/dtype/S/R. Also regress B1/8/32
with M256 and B256/M256 to cover dedup. Compare OFF profiler kernel time; capture
ON stages separately (INIT is outside KERNEL). Include all supported input modes
in correctness checks. Do not compare ON with OFF timings or change S/R per case.

Compilation/device results will be recorded in docs/BUILD_DISPATCH_SPLIT.md.

When isolating sender-only20fb05b, also cherry-pick e3e773e for its updated
wheel/checkout identity validation. The count algorithm commit is e12e478. The
latest branch already includes both algorithms and the validation correction.

Count candidate: for the same small non-dedup gate, feed the complete original
padded route IDs directly to CountFrequency with the owned expert range. The VF
range predicate excludes all other experts. This avoids Compare/And/GatherMask
setup and the per-expert reduction fallback. Histograms, count-record placement,
zero-expert readiness and R-only count ownership keep their original storage and
publication rules. There is no new allocation or cross-core synchronization.

## Target commands

Fetch `kernel/ll-g4-small-b` into a clean dedicated checkout. Load the target's
existing Python/CANN environment, activate a project-local venv if needed, and
use the standard pinned dependency/SDK build workflow. Keep the same S/R as the
baseline; defaults are S28/R28, not an automatic selection.

```bash
git submodule update --init --recursive --depth 1
unset SHMEM_ROOT CATLASS_ROOT
export DEEPEP_NPU_ARCH=Ascend950
export DEEPEP_LL_STAGE_TIMER=OFF
python scripts/build_shmem.py
python -m pip install -r requirements-build.txt
python -m pip install -v --force-reinstall --no-build-isolation --no-deps .
```

Do not force SDK rebuild solely for this branch. Preserve the target's current
device selection and topology. On the production target with16 local ranks,
set TARGET_DEVICES to its16 explicit device indices, then:

```bash
ARGS=(
  --devices "$TARGET_DEVICES" --num-processes 16
  --num-nodes 1 --node-rank 0 --master-addr 127.0.0.1
  --master-port 29500 --shmem-port 29501
  --num-tokens 1 2 4 8 16 32 --num-max-tokens-per-rank 32
  --hidden 3584 --num-topk 8 --num-experts 384 --dtype fp8 --do-cpu-sync 0
  --dispatch-only --warmup 10 --num-iters 100
  --output-dir results/small-b --run-name candidate-off-001
)
python tests/ll_stage_timer/run.py "${ARGS[@]}" --plan
python tests/ll_stage_timer/run.py "${ARGS[@]}"
```

Run baseline197052c with its own matching wheel/launcher and a distinct result
name. For ON diagnostics, rebuild/install the full ON wheel on every rank and
add `--ll-stage-timer --ll-timer-iters 5 --ll-timer-mode observe`. Keep OFF formal
performance separate. Regress M256/B1/8/32/256 and all supported dtypes. Native
EP16 must use16 real ranks; the newly authorized dual-card validation visibility
is not expanded or used to simulate an EP16 device job by these instructions.
