# A3_docker compile validation environment

User authorization: 2026-10-21. Compile-only validation; no NPU execution or wheel installation.

- SSH alias: `A3_docker` (use the user's SSH configuration; do not copy credentials).
- Authorized working root: `/home/tyf/Q4_remote_test`.
- Checkout: `DeepEP-LL-g4-overlap-recv` inside that root.
- CANN: `source /usr/local/Ascend/ascend-toolkit/set_env.sh`.
- Python: `source /root/miniforge3/etc/profile.d/conda.sh; conda activate py311_tyf`.
- Observed CANN 9.3.0, Python 3.11.17, aarch64, torch 2.11.0 / torch-npu 2.11.0.post2.
- Production compile target stays `DEEPEP_NPU_ARCH=Ascend950` / `dav-3510`; S28/R28.
- Four visibility variables stay 0,1; this does not authorize device execution.
- Keep cache, temporary files, SHMEM/ASCEND logs and outputs inside the authorized project/root.

## Reproduction

Inspect Git status and existing tasks first. Source the environment above and then
`scripts/ll_timer_env.sh` from this checkout. The source under validation is
`d2cc3e3c5d6a542e74e5ca924dc76247ebbd2f11`.

```bash
git submodule update --init --recursive --depth 1 -- third-party/shmem
unset SHMEM_ROOT CATLASS_ROOT
python -m pip install --index-url https://pypi.org/simple \
  --target "$PWD/.cache/build-python" -r requirements-build.txt
export PYTHONPATH="$PWD/.cache/build-python${PYTHONPATH:+:$PYTHONPATH}"
export DEEPEP_NPU_ARCH=Ascend950
export DEEPEP_LL_SEND_CORES=28 DEEPEP_LL_RECV_CORES=28
export DEEPEP_LL_STAGE_TIMER=ON MAX_JOBS=4 CMAKE_BUILD_PARALLEL_LEVEL=4
python scripts/build_shmem.py
python -m pytest tests/ll_stage_timer/test_overlap_receive.py \
  tests/ll_stage_timer/test_relay_timing.py tests/elastic/test_ll_input_prefetch.py \
  tests/elastic/test_ll_group_local_send.py -q --tb=short
python -m pip wheel -v --no-build-isolation --no-deps \
  --wheel-dir "$PWD/.artifacts/wheels" .
bash tests/ll_stage_timer/compile_off.sh
```

The existing conda build packages were newer than the repository pins. Install the
pinned versions in the project-local directory above, leaving conda unchanged.
The default package index failed to provide setuptools79.0.1; explicit official
PyPI succeeded. All recursive submodule revisions were fetched successfully.

## Results

SHMEM build: passed. CPU/native C++ checks: 98 passed in 7.13s.
Timer ON full wheel: passed (Host + device library + wheel packaging).
Timer OFF production dispatch and combine translation units: both passed.
Final script status: `VALIDATION COMPLETE`, exit code 0.

Wheel: `.artifacts/wheels/ascend_deepep-0.1.0.dev5-cp311-cp311-linux_aarch64.whl`
(about 2.5 MiB; this is an aarch64 artifact, not an x86 wheel).
OFF objects: `.tmp/ll-timer-off-Wu6RUx0S/`.
The SHMEM macro-redefinition warnings remain non-fatal. No additional source
changes were needed after d2cc3e3. This validates the explicit AICore callback
fix with the actual CANN compiler in both timer configurations.
Logs: `.logs/a3-compile/build-003.log` and `exit-code-003` in the remote checkout.
Earlier preparation attempts remain in `build.log` and `build-002.log`.
No device correctness or performance claim follows from compilation.
