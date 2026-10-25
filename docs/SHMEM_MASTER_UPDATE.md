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

Use [the P experiment](../tests/ll_stage_timer/PARALLEL_PREFIX.md) to build all
variants before sampling. Fix this SDK revision across the P=26/6/1 comparison;
an old-SDK versus new-SDK comparison is a separate experiment.

## Validation boundary

CPU tests cover the production Session against mainline declarations, address
translation with distinct mappings, fail-fast HT guards, SDK identity and
rebuild/cache behavior. They do not replace CANN compilation or communication
correctness. Current evidence and the remote-validation blocker are recorded
in [BUILD_DISPATCH_SPLIT.md](BUILD_DISPATCH_SPLIT.md).
