# SHMEM source and clean-build policy

## Current policy, 2026-10-26

The user requires the small A/B and paired-receive experiments to delete SHMEM
validation and rebuild from clean products on every build invocation. This
supersedes the earlier manifest/reuse instructions in this project.

- `build_ab.py` builds SHMEM once per experiment group, then builds every OFF
  and sparse wheel from clean project build products.
- `build_shmem.py` always removes the project submodule's `build`, `install`
  and `ci` outputs before invoking its build script. Its old `--rebuild`,
  `--if-needed` and `--allow-local-changes` arguments are compatibility aliases;
  none enables SDK reuse. It builds checked-out source without resetting edits.
- No build, CMake or native-loading step verifies `deepep-sdk.json`, SDK hashes,
  source revision, capability fields or `hns_1825_pi_on_chip`. The retired SDK
  validator/recorder/reuse utilities are removed. Missing or malformed optional
  metadata does not block compilation. Library resolution, compilation and
  loading still fail normally when required files or interfaces are unavailable.
- Project implementation, S/R/G, ABI/schema, source and all-rank checks remain.
  These ensure the correct experimental wheel is tested; they do not validate
  the SDK. Compilation errors stop the sequence and retain the failed logs.

Cleanup is implemented in `scripts/clean_build_artifacts.py`. It permits only
`build/` and `third-party/shmem/{build,install,ci}` beneath the script's project
root, checks resolved paths, and refuses symlink/junction escape. It does not
remove source files, `.git`, `3rdparty` downloads, old `diagnostic-runs`, saved
wheels, logs or performance captures. Use a checkout whose SDK/build products
are not currently used by another job. External SHMEM_ROOT directories are
never deleted by these scripts.

## Build both A/B groups once

Update this branch and initialize its submodules through Git. From the project
root, load the existing Python/CANN environment and run:

```bash
source /home/tyf/0env.sh
export DEEPEP_LL_SEND_CORES=28 DEEPEP_LL_RECV_CORES=28
export DEEPEP_LL_SMALL_B_PREFIX_CORES=0
unset SHMEM_ROOT CATLASS_ROOT
AB_BUILD_NAME=ll_ab_clean_build_003
AB_BUILD_DIR="diagnostic-runs/$AB_BUILD_NAME"
AB_BUILD_OK=0
if python tests/ll_stage_timer/build_ab.py --experiment all --run-name "$AB_BUILD_NAME"; then
  AB_BUILD_OK=1
  echo 'One SDK and all eight OFF/sparse wheels are ready.'
else
  echo 'Build stopped; preserve logs. Terminal remains open.'
fi
```

`--experiment small` selects cb59/cb59-small; `pair` selects
cb59-stream/cb59-pair. `all` builds both groups against the same fresh SDK.
For a read-only preview, add `--plan`; no SDK build, cleanup, NPU import or
output directory creation occurs. If the group already exists, use a new name.

Successful maps are at:

```text
diagnostic-runs/<group>/cb59/node_000/wheel-map.json
diagnostic-runs/<group>/cb59-small/node_000/wheel-map.json
diagnostic-runs/<group>/cb59-stream/node_000/wheel-map.json
diagnostic-runs/<group>/cb59-pair/node_000/wheel-map.json
```

The group `wheel-maps.json` is published only after all requested wheels pass
project Host identity checks. SDK output is in `group/logs/sdk-build.log`;
per-wheel cleanup, compile and identity logs are under each implementation's
`node_000/logs`. A failed SDK build never starts wheel compilation or devices.

Then follow [small A/B](../tests/ll_stage_timer/CB59_SMALL.md) and
[pair A/B](../tests/ll_stage_timer/CB59_PAIR.md) from their run/check sections,
retaining the same AB_BUILD_DIR and AB_BUILD_OK. Do not repeat the shared build
block or rebuild SHMEM between those runs. `diagnose.py --skip-build` consumes
these completed wheel maps without SDK checks or cleanup. This separates the
fresh build phase from the subsequent serial sampling phase.

## Source transition background

The submodule was updated on 2026-10-25 to official master commit
`8e3fc610f91ae9e5e8cf13748cfb74826b272c2a`. Git still selects the checked-in
source; the builder no longer uses a manifest or source identity as admission.

That transition changed shared-jetty configuration to
`init_attr.runtime_config.udma_shared_jetty` and removed the old dev peer-mask
extension. LL uses the explicit MTE Session and ordinary pages; UDMA transport
choices remain separate. The full SDK needs its installed device sources and
libraries for compilation, regardless of whether a manifest exists.

Earlier SDK/complete-wheel builds on27 with CANN9.3.0 and their Host checks are
historical evidence in BUILD_DISPATCH_SPLIT.md. They are not evidence of a new
build or device test for this clean-build workflow. The assistant performed
only local CPU workflow/path tests for this policy change.
