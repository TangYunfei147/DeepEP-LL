# A5 scalar exploration, 2026-10-22

User target observation: 197052c is stably ~1% faster than c038f44; af094ce
(receiver + packed q) returns to baseline. Keep 197052c as this branch's base.

Measured on SSH 177.22.26.1 / rack05os6, Ascend950DT, CANN9.3.0 from the user's
0env.sh. Python torch2.11.0 / torch_npu2.11.0.post2. Both newly authorized hosts
were idle at initial inspection; every launch is preceded by an occupancy check.
No unrelated task was stopped. Scalar execution used device0 only.

Production source remains identical to 197052c. Standalone test source cf0e574,
runner 934c4b4; latest runner strengthens occupancy parsing without changing any
operation. 8 configurations x 10 modes x 100 recorded rounds, 10 warmups; all
checksums passed. Values below are medians of the per-core tail50 medians,
56 AIV and 32 repeats. Raw ticks/repeat are reported, not event time.

| Operation | count1 ticks/repeat | count32 ticks/repeat |
|---|---:|---:|
| Empty loop/drain | 7.875 | 7.875 |
| Three-field Scalar metadata copy | 54.219 | 1280.594 |
| Four-word Vector copy + dependencies | 54.688 | 55.563 |
| Direct header reference | 8.000 | 8.000 |
| Scalar pending indices | 15.438 | 96.000 |
| Vector pending indices + dependencies | 51.188 | 51.391 |
| Plain 72-route q construction | 2227.438 | 2227.500 |
| Packed 72-route q construction | 2358.852 | 2358.688 |
| 8192B route DMA to each core | 860.594 | 859.586 |
| Same IDs preloaded in UB | 8.313 | 8.313 |

At the previously confirmed 1GHz timer frequency, ticks are ns. Retain raw ticks
when comparing machines. This timing window includes the described dependencies
and drain, so it is not a single-instruction latency measurement.

Findings:

- A per-packet tiny vector copy offers no clear steady-state benefit over the
  three Scalar fields. Removing the copy entirely is the more useful direction.
- Vector metadata copying benefits a batch of 32 records (~1.281us vs .056us),
  but only if that batch already exists. Do not accumulate packets and delay
  first relay just to obtain this microbenchmark ratio.
- One Scalar pending index is cheaper than a vector launch. With 32 indices,
  the vector version wins. This rejects a blanket “vectorization is always free”
  assumption; current production already vectorizes the initialization.
- Packed q adds ~131ns per 72-route construction (~5.9%). This supports a setup
  tradeoff, but does not explain the entire target 1% regression by itself:
  FillTail savings, decoding, code generation and pipeline effects remain outside
  this microkernel. Do not add packed q back based on setup tests alone.
- 1-core repeated route DMA is ~155ticks versus ~860ticks at 56 cores. All cores
  read the same synthetic 8KiB source, so contention/cache behavior is specific
  to this pattern. This is a useful route-input ablation, not evidence that all
  production preparation can be eliminated by constant top-k IDs.

Original CSV and identity on 177.22.26.1:
`/home/tyf/Q4_remote_test/DeepEP-LL-scalar-probes/tests/ll_micro/results/scalar-001/`
Initial one-core sanity run: `results/scalar-c1-r32-n32.csv`.
Build logs: `/home/tyf/Q4_remote_test/.logs/scalar-build-001.log` through `003.log`.
First build failed because CreateVecIndex<uint32_t> is unsupported on this
toolchain. Switching the probe to int32 indices fixed it; subsequent A5 build and
checksum validation passed. No architecture, SDK or optimization-level change.

These results are warm standalone measurements without production Exp/all_reduce
conditioning. They are not target OFF dispatch results or an end-to-end bandwidth
comparison. See README for target transfer modes and their topology limitations.

## Transfer smoke: paused for external NPU work

The standalone scalar and transfer executables compiled successfully at cf0e574.
Transfer run `transfer-smoke-001` completed 4KiB/1core/1lane tx modes0/1/2,
100 samples and before/after destination checks. The next launch was refused by
the occupancy guard. Read-only inspection confirmed external python3.11 tasks
on devices0/1/2/3/4/7. No process was stopped and no further NPU job was launched.

Tail50 event medians: empty7.7885us; UB/control7.5295us; preloaded-UB tx7.5480us.
These small events are dominated by launch/runtime variability; the non-monotonic
ordering precludes subtracting them into a reliable transport latency. Do not
report the 4KiB effective rate as link bandwidth. Load-only/rx/relay and the larger
bandwidth matrix remain **device-unvalidated**, pending an idle target machine.
Raw results: `tests/ll_micro/results/transfer-smoke-001/` on177.22.26.1.

Local verification: required dispatch/source/ST contracts43passed/2compiler skips;
occupancy fail-closed and per-core tail50 tests5passed. Initial local pytest
missing basetemp parent was fixed before the passing run. Production wheel/ON-OFF
linking was not repeated because production sources/build graph did not change.

## Stage audit follow-up on177.22.27.1

26 remained occupied, so no job was launched there. 27 was inspected idle, then
production-target O2/dav-3510 compilation and a device0 checksum run succeeded.
First fixed-mask pilot e785b3d is retained in stages-001 but is not used as the
main mask result. The corrected f984751 loads mask bits from runtime UB inputs
in both versions, avoiding constant-folded mask arithmetic. It passed all14modes,
10warmups/100samples at56AIV/32repeats/count32; tail50 per-core medians are below.

| Operation | Original ticks/repeat | Candidate ticks/repeat | Interpretation |
|---|---:|---:|---|
| 32token masks,3first-hop bits/token |1397.984|875.078|register accumulation removes repeated UB mask read/write|
| 32DIRECT descriptors plus consumption |1852.594|1505.563|read immutable q at use instead of descriptor roundtrip|

At1GHz, savings are .523us/32tokens and .347us/32DIRECT tasks. This is roughly
16.3ns/token with3first hops and10.8ns/DIRECT task in this synthetic workload,
not a prediction for full dispatch. Production first-hop counts, register pressure
and overlapping input prefetch differ. No production candidate was applied yet.
The same corrected binary measured plain-q2232.563ticks vs packed-q2362.438ticks
per72-route construction, consistent in direction with the earlier26 result.

Raw location on27:
`/home/tyf/Q4_remote_test/DeepEP-LL-scalar-probes/tests/ll_micro/results/stages-002/`.
Build/run logs: `.logs/stage-build-002.log` and `.logs/stage-run-002.log` under the
authorized root. All14mode checksums passed; no NPU correctness claim is made for
any fused receiver VF or modified production dispatch.

The complete source-stage audit, priorities and protocol requirements are in
[STAGE_OPPORTUNITIES.md](STAGE_OPPORTUNITIES.md). Runner regression5passed.
