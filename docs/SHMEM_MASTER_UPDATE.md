# SHMEM official master update, 2026-10-25

The submodule now pins official `master` at
`8e3fc610f91ae9e5e8cf13748cfb74826b272c2a` (upstream commit date 2026-10-23).
The official remote HEAD was checked on 2026-10-25. Gitlink, branch/ref metadata
and six public-header hashes agree. No private patch was applied to SHMEM.

This is an SDK transition: rebuild the SDK and every OFF/ON wheel. An old
installed SDK is rejected even when some public headers happen to match.
The manifest and native cache cover all installed include/src/lib inputs,
including device implementation `.hpp` and bootstrap/helper libraries.

## Compatibility changes

- Shared jetty uses `init_attr.runtime_config.udma_shared_jetty`.
- The old dev peer-mask extension is absent. Explicit UDMA users must opt into
  `DEEPEP_SHMEM_DENSE_QPS=1`; there is no silent dense fallback.
- LL ST explicitly requests an MTE-only Session and ordinary pages. It does not
  initialize UDMA QPs. Public ElasticBuffer retains an explicit `transport`
  choice; the new LL path does not infer transport from token count or topology.
- Mainline UDMA connection polling has a fixed 120000 ms limit. The old
  `SHMEM_UDMA_CONNECT_TIMEOUT_MS=600000` setting no longer changes it.
- Private HT address helpers now distinguish local symmetric pointers, mapped
  peer pointers and registered remote addresses. They pass the real local/peer
  MR bases and reject out-of-range addresses.
- Old HT Dispatch writes directly into ordinary framework peer output, whose
  registration is not established by this interface. It is explicitly rejected
  before Notify/launch. LL MTE remains available; symmetric control/workspace
  and HT Combine are separate paths. Compilation and CPU address tests do not
  establish full HT device compatibility.

The topology's four Clos ports cannot be selected by the current LL MTE
`aclshmem_ptr`/DataCopy API. QP count is not port count. This update makes no
claim about striping traffic over four ports or multiplying measured bandwidth.

## Target preparation

Run from the project root in the target's existing Python/CANN environment.
Use a fresh checkout if other jobs use the old SDK; otherwise stop its users
before the explicit rebuild. Do not overwrite their SDK in place while running.

```bash
git fetch origin
git switch kernel/ll-g4-small-b-prefix
git merge --ff-only origin/kernel/ll-g4-small-b-prefix
git submodule update --init --recursive --depth 1
unset SHMEM_ROOT CATLASS_ROOT
export DEEPEP_NPU_ARCH=Ascend950
python scripts/build_shmem.py --rebuild
export LD_LIBRARY_PATH="$PWD/third-party/shmem/install/shmem/lib:${LD_LIBRARY_PATH:-}"
```

The SDK builder uses upstream `-clean`, project-local temporary/cache/log
directories, and records the installed SDK only after success. It never resets
modified submodule sources. Build dependencies are listed in
`requirements-build.txt`; prepare them in the project environment before
`diagnose.py --build-only`, which does not download dependencies.

For an offline first SDK build, preload the official nlohmann/json v3.11.3
source at `third-party/shmem/3rdparty/json` (including
`single_include/nlohmann/json.hpp` and its license). The source is an upstream
build dependency, not a tracked vendor modification. Do not copy an old
compiled SHMEM library under a new manifest.

## Optional vendor metadata and incomplete manifests (2026-10-26)

`build.hns_1825_pi_on_chip` is optional vendor metadata. This branch accepts
every JSON value without coercion or a fixed enum: absent, null, bool, integer,
float, string, array or object. It does not require `rdma_backend` merely
because this field exists. An explicitly supplied backend must still be
`HNS_1825`, and a consumer requiring RDMA must still have that backend recorded.
The optional field itself never grants RDMA capability.

MTE reuse compares the common revision/CANN/target/source-patch inputs while
leaving the optional field untouched. SDK recording, the wheel build metadata
and native cache retain the complete `build` object, including its original
JSON types. Changing the recorded value isolates the native cache even when
the SDK file hashes are unchanged. Other unknown build fields, mismatched
source revision, modified inputs and incomplete file manifests remain errors.

The error `manifest must cover all installed include/src/lib inputs` is
independent of this optional field. It means some currently installed input
files have no recorded hash: the manifest may be from an older partial recorder,
or files may have been added after the identity was produced. The check now
reports the total, per-directory counts and up to eight example paths. It does
not modify the manifest or certify unrecorded binaries.

For a read-only diagnostic, select the same SDK as the build:

```bash
python scripts/record_shmem_sdk.py \
  --sdk "${SHMEM_ROOT:-$PWD/third-party/shmem/install/shmem}" --check
```

If only this optional field differed and full identity/file validation passes,
reuse the SDK with the updated scripts; rebuilding it is unnecessary. If the
manifest is incomplete, use an explicit build from the pinned source. The
commands below rebuild the **project's** SDK, not an externally selected SDK.
Stop users of that project SDK first, or use a separate checkout if it is in use.
Do not delete its manifest and rehash the existing library as a shortcut.

```bash
prepare_checked_sdk() {
  git submodule update --init --recursive --depth 1 || return $?
  unset SHMEM_ROOT CATLASS_ROOT
  export DEEPEP_NPU_ARCH=Ascend950
  python scripts/build_shmem.py --rebuild || return $?
  export LD_LIBRARY_PATH="$PWD/third-party/shmem/install/shmem/lib:${LD_LIBRARY_PATH:-}"
  python scripts/record_shmem_sdk.py \
    --sdk third-party/shmem/install/shmem --check || return $?
}
build_small_ab_after_sdk() {
  local impl
  export DEEPEP_LL_SEND_CORES=28 DEEPEP_LL_RECV_CORES=28
  export DEEPEP_LL_SMALL_B_PREFIX_CORES=0
  for impl in cb59 cb59-small; do
    DEEPEP_LL_DISPATCH_IMPL="$impl" \
      python tests/ll_stage_timer/diagnose.py --build-only \
        --run-name "small_scope_${impl}_build_002" || return $?
  done
}
if prepare_checked_sdk && build_small_ab_after_sdk; then
  echo 'SDK checked and both complete OFF/ON builds finished.'
else
  echo 'Preparation or build stopped; preserve the logs. Terminal remains open.'
fi
```

Load the existing target Python/CANN environment first, and update this branch
before using these commands. The `_002` names preserve the failed `_001`
attempt; use new names again if `_002` already exists. Subsequent run commands
must point to the successful `_002/node_000/wheel-map.json` paths. Both A/B
variants must use that same validated SDK and the same source revision.

Use [the P experiment](../tests/ll_stage_timer/PARALLEL_PREFIX.md) to build all
variants before sampling. Fix this SDK revision across the P=26/6/1 comparison;
an old-SDK versus new-SDK comparison is a separate experiment.

## Validation boundary

The official SDK and six complete P26/P6/P1 OFF/ON wheels compiled and linked
on27 with CANN9.3.0. All six pass Host identity and actual loaded-SDK path/hash
checks; the production SDK source remains clean. Linux Host/C++ follow-up has
162 passing tests. No NPU was run, so communication correctness and performance
remain untested. Full evidence, cold compile times and artifact locations are
recorded in [BUILD_DISPATCH_SPLIT.md](BUILD_DISPATCH_SPLIT.md).
