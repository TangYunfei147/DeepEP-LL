# LL kernel configuration and branch

`kernel/ll-configurable-groups` is based on upstream-aligned `main` and contains
LL dispatch/combine, required Host/runtime/build changes, and CPU protocol tests.
`experiments/ll-four-layer-probes` merges that branch and adds experimental tooling.

## Logical rank group

Edit `DEEPEP_LL_RANKS_PER_GROUP` in
`csrc/kernels/dispatch/low_latency/common/ep_memory_server_dedup_protocol.h`.
The default is 16; the constant accepts integers from 1 through 16. Rebuild the complete wheel
and any generated metadata snapshot after changing it. This is a compile-time
constant, not a runtime environment variable. It is independent of physical
cards per host and of the number of AIV cores. Rank `r` belongs to group `r / G`
and has group-local index `r % G`. Groups comprise contiguous global ranks.
A physical host may contain several groups; a group may span hosts only when
all required peer MTE mappings are actually supported by the deployment.

The native attribute `low_latency_ranks_per_group` reports the compiled value.
Host/device mismatch is rejected at module initialization. All ranks must use
the same build. The old misleading `low_latency_ranks_per_host` is removed.
The 512-byte ready-row capacity and eight notification cores are independent
protocol capacities. EP must remain at least 16 and divisible by the group size;
the regular experiment launcher retains its EP16/32/64/128/256 support list.
For that list only G=1/2/4/8/16 divide EP. Other constants do not automatically
enable additional end-to-end EP configurations;
configuring G=8 does not enable EP8. Existing layout and buffer limits remain.

## Build and validate

Load the current machine's CANN environment, then from the repository root:

```bash
git submodule update --init --recursive --depth 1
unset SHMEM_ROOT CATLASS_ROOT
python scripts/build_shmem.py
export DEEPEP_NPU_ARCH=Ascend950
export LD_LIBRARY_PATH="$PWD/third-party/shmem/install/shmem/lib:${LD_LIBRARY_PATH:-}"
python -m pip install -r requirements-build.txt
python -m pip install -v --force-reinstall --no-build-isolation --no-deps .
python -m pytest tests/elastic/test_ll16_protocol.py tests/elastic/test_ll_core_config.py
```

Use project-local caches, logs, temporary files and a project-local Python
environment. In the experiment branch, source `scripts/project_env.sh` first.
Its regular test launcher accepts physical `--num-processes` independently;
with torchrun, set the actual `LOCAL_WORLD_SIZE`. `run_ll16.sh` remains an
explicit convenience launcher for sixteen processes per physical host.
CPU plan output uses null for unknown compiled group size; actual execution
reads it from the installed native module. Profiling remains 10 warmups,
100 captures, Exp plus scalar all_reduce per iteration, last 50 per task.

A5 validation uses devices 0/1 only. CPU protocol tests exercise G=1..16,
divisible EP16..256 (including non-power-of-two layout cases) and AIV16..72, including repeated ready clearing. They do not prove
NPU ordering or cross-host communication. Full LL correctness and performance
must be validated by the user on the target deployment.
