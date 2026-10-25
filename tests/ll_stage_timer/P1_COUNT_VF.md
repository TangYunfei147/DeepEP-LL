# P1 count ready/extraction VF candidate

This is one independent change against `578d32b37330fd579b2c7376d651b7ffadeff9e8`.
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

Run from the project root with a clean checkout. Preserve existing results and
use a fresh suffix if any of the following build/run directories already exist.
The reported `1b44449` target revision was not available locally when preparing
this candidate; use the known parent below for an attributable code A/B rather
than assuming old target wheels match it.

```bash
source /home/tyf/0env.sh
git fetch origin
git switch kernel/ll-g4-small-b-prefix
git merge --ff-only origin/kernel/ll-g4-small-b-prefix
test -z "$(git status --porcelain)" || { echo 'Preserve/commit local changes first'; exit 1; }

BASELINE_REV=578d32b37330fd579b2c7376d651b7ffadeff9e8
CANDIDATE_REV=$(git rev-parse HEAD)
export DEEPEP_LL_SEND_CORES=30 DEEPEP_LL_RECV_CORES=26
export DEEPEP_LL_SMALL_B_PREFIX_CORES=1
unset SHMEM_ROOT CATLASS_ROOT
export LD_LIBRARY_PATH="$PWD/third-party/shmem/install/shmem/lib:${LD_LIBRARY_PATH:-}"

# Build and Host-verify all four wheels before any device sampling.
git switch --detach "$BASELINE_REV"
python tests/ll_stage_timer/diagnose.py --build-only \
  --run-name count_vf_base_build_001 || exit $?
git switch --detach "$CANDIDATE_REV"
python tests/ll_stage_timer/diagnose.py --build-only \
  --run-name count_vf_candidate_build_001 || exit $?
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
    --wheel-map "diagnostic-runs/count_vf_${label}_build_001/node_000/wheel-map.json" \
    --run-name "count_vf_${label}_run_001" \
    --bs 1 2 4 8 16 24 32 --m 32 \
    --hidden 3584 --topk 8 --experts 384 --dtype fp8 --cpu-sync 0 \
    --devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
    --num-nodes 1 --node-rank 0 \
    --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
    --shmem-port "$SHMEM_PORT"
}
run_count_ab base "$BASELINE_REV" || exit $?
run_count_ab candidate "$CANDIDATE_REV" || exit $?
git switch kernel/ll-g4-small-b-prefix
```

Each source checkout must match its wheel: `--skip-build` intentionally rejects
an old wheel under candidate source. Do not remove this check to reuse a wheel.
Both cases keep the same uniform-random route seeds and profiler protocol:
10 warmups/100 captures/last50 OFF tasks, followed by sparse diagnostic capture.
First send only these two files:

- `diagnostic-runs/count_vf_base_run_001/SEND_FIRST.txt`
- `diagnostic-runs/count_vf_candidate_run_001/SEND_FIRST.txt`

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
