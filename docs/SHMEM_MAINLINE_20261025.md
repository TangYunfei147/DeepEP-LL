# SHMEM mainline migration,2026-10-25

Runtime correction: the first migration commit `c705381` still constructed the
general Elastic MTE+UDMA session even when the LL kernel used MTE. That triggered
the removed HT sparse-QP API before dispatch. Use the explicit LL-only fix
`ed1c553` and [LL_ONLY_FIX.md](../tests/ll_control_opt/LL_ONLY_FIX.md), with the
new three-entry manifest and HT compilation disabled. Updating only the root
checkout while selecting `c705381` from the old manifest retains the old error.

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
The corrected explicit LL-only MTE session does not enable this UDMA setting.

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
through an unrequested SDK patch. The corrected LL-only session does not initialize
that UDMA transport path; HT/UDMA and future cross-host tests must account for it.

The new SDK provides QP-specific/multi-link UDMA interfaces. Updating the SDK
does not convert the LL kernel's MTE DataCopy path into QP-specific UDMA, and a
QP index is not a physical NIC index. The confirmed3fullmesh+4Clos+1inter-host
ports inform experiment topology; actual per-port utilization needs telemetry.

## Optional official primitive diagnostics

The pinned source includes `examples/udma_qp_demo` for route/correctness checks,
`examples/shmem_perftest/udma_perftest` for UDMA microbenchmarks and
`examples/shmem_perftest/mte_perftest` for MTE microbenchmarks. Prepare an isolated
SDK checkout under the project before running the official example build:

```bash
bash scripts/build.sh -examples -soc_type Ascend950 -multi_link
```

This prepares additional CATLASS/json dependencies and replaces that checkout's
install directory. Do not run it against the SDK used by an existing LL job.
Keep the checkout, environment, outputs, logs and caches inside the project;
load the target's existing CANN environment and explicit device visibility.
These commands have been checked against source, not executed on the target.

From the isolated SDK, examples of route checks are:

```bash
# PE0..3 bind to device0..3; each sends to the next PE in a ring.
bash examples/udma_qp_demo/run.sh \
  -pes 4 -device_ids 0,1,2,3 -peer_stride 1 \
  -qp_count 4 -op put -elems 1048576 \
  -qp_net_layers 0:0,1:0,2:1,3:1
# Two PEs bind to device0 and4, crossing the logical G4 boundary.
bash examples/udma_qp_demo/run.sh \
  -pes 2 -device_ids 0,4 -peer_stride 1 \
  -qp_count 4 -op put -elems 1048576 \
  -qp_net_layers 0:1,1:1,2:1,3:1
```

Read `[QP-ROUTE]`'s actual layer rather than assuming a request selected a
particular port. These are logical-layer requests that must be supported by
the target topology; the demo has no bandwidth statistic. It does not change
the policy of another process or the LL MTE path.

For the UDMA perftest, `-b` is QP count and `-e` is the byte exponent **per QP**.
To hold total bytes at1MiB, use `(b,e)=(1,20),(2,19),(4,18),(8,17)`.
`--metric bw` includes quiet/completion; `--metric lat` measures submission
without completion and only supports `-b 1 -t put`. `--batch 0/1/N` selects
fully asynchronous, per-operation or batched completion. This perftest has no
`qp_net_layers` argument and does not inherit the route demo's policy.
Its script reuses CSV names: run the binary in a fresh per-case output directory
or retain each result before the next case. Preserve `Bandwidth/GB/s(1000)`,
`Bandwidth/GiB/s(1024)`, `CoreMaxTime/us` and `SingleCoreTime/us` separately.

`build/bin/root_info_generate <physical-device-id>` can reveal EIDs, planes,
ports and logical layers. Capture its output in the project only; never install
generated topology into `/etc/hccl_rootinfo` or change the live SHMEM topology.
UDMA debug records can associate a peer/QP with layer, plane and EID indices,
but an EID can cover multiple ports. Physical utilization requires per-port
byte counters over a measured interval and the corresponding port peak rates.
Primitive results cannot be reported as LL operator improvements.

## Validation limits

Local CPU validation passed50SDK/identity/raw-WQE checks, including a C++ test
against field construction extracted from this exact SDK and2,048configurations
with distinct local-heap, mapped-peer and remote-MR bases. The production Session
also passed39C++ lifecycle/fault-injection checks, including missing sparse API,
MTE independence and preservation of the new versioned runtime configuration.

Those initial migration checks were CPU-only. The subsequent explicit LL-only
fix was compiled in the user-authorized isolated27directory: a fresh SDK and
six complete LL OFF/ON wheels passed, including all seven dispatch entries.
Host-only imports verified HT disabled and the loaded SDK identity. See
[LL_ONLY_FIX.md](../tests/ll_control_opt/LL_ONLY_FIX.md) for exact revisions and
logs. No NPU correctness or performance run was performed. HT is excluded from
this performance build and remains unqualified with this SDK.
