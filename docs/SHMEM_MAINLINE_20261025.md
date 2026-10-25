# SHMEM mainline migration,2026-10-25

This update follows the user's explicit request to replace the old SHMEM pin.
The official repository's advertised HEAD was checked twice on2026-10-25:

- Repository: https://gitcode.com/cann/shmem.git
- Mainline: `refs/heads/master`, not the former `dev` tracking branch.
- Locked commit: `8e3fc610f91ae9e5e8cf13748cfb74826b272c2a`.
- Commit timestamp:2026-10-23T18:49:15+08:00; upstream VERSION:1.7.0.
- Previous pin: `4dd80e2129da42bd773982c2bb3758af1ed0bba8`.

The gitlink, .gitmodules branch and dependency lock agree on the new pin.
The six locked public-header LF hashes were recomputed from that exact source.
No upstream SHMEM source is patched. Other experimental worktrees retain their
own historical pins; this migration belongs to kernel/ll-g4-control-opt.

## Necessary compatibility changes

The new SDK removed `aclshmemx_set_shared_jetty`. Session now selects shared
UDMA jetty through `attr.runtime_config.udma_shared_jetty`, while preserving the
header-provided `option_attr.version` by assigning option fields individually.
LL MTE does not enable this UDMA setting.

The legacy peer-QP-mask setter was also removed and has no equivalent public
replacement in this pin. A request for that HT sparse policy fails explicitly
before initialization. The existing explicit `DEEPEP_SHMEM_DENSE_QPS=1` HT mode
remains an opt-in; the update does not silently enable it. LL MTE is independent
of this missing HT feature. This is compile/API adaptation, not HT qualification.

The internal UDMA SQE builder now translates a LOCAL symmetric pointer through
local heap and remote MR bases. Seven HT call sites already held a raw remote VA
and could not simply append the new argument. The project now has an explicitly
named raw-remote builder, with SQE/SGE size and field-offset assertions and the
upstream flag helpers. Exchanged framework-output VAs remain raw. FM completion
flags and Combine workspace pointers are translated once with the new SDK's
remote-MR helper. Public put/get APIs keep their local-symmetric contract.

## SDK identity and rebuilding

Wheel configuration now rejects mismatched SDK revisions, headers or libraries
instead of warning and continuing. SDK manifests cover all installed `.h` and
`.hpp` under include/src, and all shared libraries under lib. A manifest that
omits installed build inputs is rejected. A4dd80e SDK cannot build a new-pin wheel.

Explicit `--rebuild` archives the old project-owned build/install directories to
`.cache/shmem-rebuild/<unique-id>/` before rebuilding. It never silently rebuilds
on an identity mismatch. Do this only when no job is using that SDK:

```bash
git submodule update --init --recursive --depth 1 -- third-party/shmem
unset SHMEM_ROOT CATLASS_ROOT
python scripts/build_shmem.py --rebuild
```

For a fresh source with no existing build/install, omit `--rebuild`. The existing
Python/CANN environment and production Ascend950 target must be used. All
participating ranks need matching newly built wheels and SDKs.

The control suite separates native build workspaces by the selected commit's
dependency identity. Historical source-s30 caches and old SDK/wheel artifacts
remain available for the explicit old-SDK baseline. New SDK data and old data
must never be merged under the same candidate revision or resumed run identity.

## Known upstream behavior changes

The old pin exposed `SHMEM_UDMA_CONNECT_TIMEOUT_MS`. This master pin removes that
reader and uses a fixed120000ms UDMA channel-status polling timeout. The old
600000environment value no longer extends this path. It has not been restored
through an unrequested SDK patch. Current single-host LL MTE does not initialize
that UDMA transport path; HT/UDMA and future cross-host tests must account for it.

The new SDK provides QP-specific/multi-link UDMA interfaces. Updating the SDK
does not convert the LL kernel's MTE DataCopy path into QP-specific UDMA, and a
QP index is not a physical NIC index. The confirmed3fullmesh+4Clos+1inter-host
ports inform experiment topology; actual per-port utilization needs telemetry.

## Validation limits

Local CPU validation passed50SDK/identity/raw-WQE checks, including a C++ test
against field construction extracted from this exact SDK and2,048configurations
with distinct local-heap, mapped-peer and remote-MR bases. The production Session
also passed39C++ lifecycle/fault-injection checks, including missing sparse API,
MTE independence and preservation of the new versioned runtime configuration.

These are CPU semantic checks. No SSH connection, CANN full wheel build, NPU
correctness test or performance run was performed during this migration.
Compatibility of all seven dispatch entries and the other device targets still
requires the target's normal full build. Historical CANN builds for4dd80e do not
validate this SDK. Logs are retained in the project's .tmp validation directories.
