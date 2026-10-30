> 历史指南：旧打点及入口适用于原提交（最后完整旧实现e4d114c）。当前已删除旧插桩/解析；构建与采样以[reference schema7指南](../ll_reference_profile/README.md)为准。下文算法与历史结果保留。

# P1 count ready/extraction VF candidate

The VF is the independent change originally committed as `807025f` against
`578d32b`. For current A/B use baseline
`e19aa4d3d91d8bc44e141dff25449c9fdb7cbf63` and the current branch candidate:
both now contain the same SDK identity compatibility correction. Their
production kernel difference remains the P1 VF change.
Keep S30/R26 and `DEEPEP_LL_SMALL_B_PREFIX_CORES=1` in both builds and runs.
The global P default remains unchanged; this experiment sets P explicitly.

Only the existing EP16 small-direct path with effective P1 uses the fused VF.
One VF checks all valid ready words and extracts their uint32 count words from
the same immutable MTE2 snapshot. A failed poll may overwrite private count
scratch but cannot clear flags, publish a prefix or start reception. The
existing MTE2-to-V and final V-to-Scalar synchronization remain. There is one
VF call and one V-to-Scalar completion per poll, replacing the two GatherMask
steps, reduction and intervening Scalar/Vector exchanges.

Tail lanes are masked, ready must have exactly the float1.0 bit pattern, and
count words are copied without float conversion. Existing buffers are reused:
no new UB allocation, lane reduction, prefix layout, packet format, generation,
cleanup ordering or combine change. Other P/domain/dedup paths stay unchanged.
The current sparse timer retains the same milestone boundaries; it does not
invent separate durations for the two now-fused operations.

The UB gather indices are in elements, with a common aligned record base for
flags and counts, as specified by the
[Ascend C register gather API](https://asc.gitcode.com/api/SIMD-API/c_api/reg/reg_load/asc_gather.html).
The production code uses the installed CANN public `c_api/asc_simd.h` API.

## Matched baseline/candidate test

Use the allocated 16-card target with its existing Python/CANN environment.
This is not a command to run NPU work on the compile-only validation host.
The mainline SDK from the preceding P experiment is unchanged and can be reused
after validation. Do not rebuild it merely for this VF change.

The reported SDK with HNS1825 and `hns_1825_pi_on_chip=1` uses the correct
upstream revision. Both current A/B revisions accept this known capability
metadata (strict integer0/1, HNS only). However, its old manifest records only
34 interface headers and libshmem.so: zero implementation files and no auxiliary
libraries. That separate incomplete-coverage issue still requires explicit SDK
rebuilding; hashes are not fabricated or silently appended to old manifests.

For that specific old manifest, first stop jobs using this project SDK, then
run the following once in the existing target Python/CANN environment. It
prepares the SDK for this MTE test branch and does not change SHMEM sources:

```bash
prepare_count_sdk() {
  source /home/tyf/0env.sh || return $?
  git fetch origin || return $?
  git switch --detach e19aa4d || return $?
  git submodule update --init --recursive --depth 1 -- third-party/shmem || return $?
  unset SHMEM_ROOT CATLASS_ROOT
  export DEEPEP_NPU_ARCH=Ascend950
  python scripts/build_shmem.py --rebuild
}
if prepare_count_sdk; then
  echo 'SDK rebuilt with a complete identity; continue below.'
else
  echo 'SDK preparation failed; keep its output and do not start the wheel builds.'
fi
```

Run from the project root with a clean checkout. Preserve existing results and
use a fresh suffix if any of the following build/run directories already exist.
The reported `1b44449` target revision was not available locally when preparing
this candidate; use the known parent below for an attributable code A/B rather
than assuming old target wheels match it.

```bash
build_count_ab() {
source /home/tyf/0env.sh || return $?
git fetch origin || return $?
git switch kernel/ll-g4-small-b-prefix || return $?
git merge --ff-only origin/kernel/ll-g4-small-b-prefix || return $?
git diff --quiet && git diff --cached --quiet || { echo 'Preserve/commit tracked changes first'; return 1; }

BASELINE_REV=e19aa4d3d91d8bc44e141dff25449c9fdb7cbf63
CANDIDATE_REV=$(git rev-parse HEAD)
export DEEPEP_LL_SEND_CORES=30 DEEPEP_LL_RECV_CORES=26
export DEEPEP_LL_SMALL_B_PREFIX_CORES=1
unset SHMEM_ROOT CATLASS_ROOT
export LD_LIBRARY_PATH="$PWD/third-party/shmem/install/shmem/lib:${LD_LIBRARY_PATH:-}"

# Build and Host-verify all four wheels before any device sampling.
git switch --detach "$BASELINE_REV" || return $?
python tests/ll_stage_timer/diagnose.py --build-only \
  --run-name count_vf_base_build_003 || return $?
git switch --detach "$CANDIDATE_REV" || return $?
python tests/ll_stage_timer/diagnose.py --build-only \
  --run-name count_vf_candidate_build_003 || return $?
}
if build_count_ab; then
  echo 'Both builds passed; proceed to the run block below.'
else
  echo 'Build stopped. Keep the logs; the current terminal remains open.'
fi
```

Set MASTER_ADDR, MASTER_PORT and SHMEM_PORT to the actual addresses/ports
assigned to this job. Run baseline and candidate sequentially:

```bash
: "${MASTER_ADDR:?Set the allocated master address}"
: "${MASTER_PORT:?Set the allocated master port}"
: "${SHMEM_PORT:?Set the allocated SHMEM port}"

run_count_ab() {
  label=$1
  revision=$2
  git switch --detach "$revision" || return $?
  python tests/ll_stage_timer/diagnose.py --skip-build \
    --wheel-map "diagnostic-runs/count_vf_${label}_build_003/node_000/wheel-map.json" \
    --run-name "count_vf_${label}_run_003" \
    --bs 1 2 4 8 16 24 32 --m 32 \
    --hidden 3584 --topk 8 --experts 384 --dtype fp8 --cpu-sync 0 \
    --devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
    --num-nodes 1 --node-rank 0 \
    --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
    --shmem-port "$SHMEM_PORT"
}
if run_count_ab base "$BASELINE_REV" && run_count_ab candidate "$CANDIDATE_REV"; then
  git switch kernel/ll-g4-small-b-prefix
else
  echo 'Run stopped. Keep the logs; the current terminal remains open.'
fi
```

Do not terminate the interactive shell on a failed child command. The functions
above return failures to the guarded caller. Preserve any earlier logs under
`diagnostic-runs/count_vf_{base,candidate}_build_001/node_000/logs/` or `_002`:
`venv.log`, `off-build.log`, `sparse-build.log`, and the install/identity logs.
Build subprocess output is redirected there; `Command failed (-11/-6)` can
identify a child signal, while a direct terminal crash may precede log creation.
Use fresh output names for a later retry and preserve the failed artifacts.

Each source checkout must match its wheel: `--skip-build` intentionally rejects
an old wheel under candidate source. Do not remove this check to reuse a wheel.
Both cases keep the same uniform-random route seeds and profiler protocol:
10 warmups/100 captures/last50 OFF tasks, followed by sparse diagnostic capture.
First send only these two files:

- `diagnostic-runs/count_vf_base_run_003/SEND_FIRST.txt`
- `diagnostic-runs/count_vf_candidate_run_003/SEND_FIRST.txt`

Use full OFF task time as the result. ENTRY→PREFIX_READY, first receive probe
and active ON/OFF perturbation explain changes, but diagnostic spans cannot be
added or substituted for OFF latency. No performance improvement is assumed.
If the first comparison is promising, confirm in reverse order with fresh run
names and paired independent seeds. Keep all summaries/supplements within ten
sends; original captures remain on the target.

## Validation

Actual VF semantics are checked with native C++ register mocks, including
nonmultiples of64, stale/unready snapshots, all-ready retries, exact ready bits,
large integer counts, masked tail/guard regions and source immutability. These
tests do not substitute for NPU correctness. CANN OFF/ON full wheel evidence
and limitations are recorded in `docs/BUILD_DISPATCH_SPLIT.md`.

On27, the standalone VF probe, all seven production modes and complete OFF/ON
wheels passed CANN compilation and linking. Final installed source/P/timer and
actual SDK path/hash checks passed. Linux Host/C++ regression:142passed. The
initial Linux test-fixture compile error (missing `<iterator>`) was fixed before
the final run and final wheel fingerprints; production VF code was unchanged.
No NPU was executed, so performance and device correctness remain to be tested.
