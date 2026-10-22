# A5 small-B build environment and offline transfer

User instruction: use the Python/NPU environment from `/home/tyf/0env.sh`.
27 has restricted external networking; prefer preparing materials locally and
uploading them. Never replace this with another Python environment silently.

Observed on177.22.27.1:

```bash
# Initialize conda for this non-interactive SSH shell, then let0env select env.
source /root/miniconda3/etc/profile.d/conda.sh
source /home/tyf/0env.sh
command -v python
# /root/miniconda3/envs/py311_tyf/bin/python
```

All CANN builds, native CPU tests, SDK builds and wheel builds used this Python.
Shell-only preflight/Git checks do not select a separate build Python. The SDK
metadata reports CANN9.2.0 although the environment installation path is named
cann-9.3.0; preserve the actual build metadata when reproducing results.

Source is transferred as Git bundles, fetched/fast-forwarded only after checking
the destination checkout. Dependency wheels are downloaded on the local machine
for Linux aarch64/Python3.11, uploaded into the authorized project, and checked
against SHA256SUMS before installation. Example local preparation:

```bash
python -m pip download --dest .cache/wheelhouse --only-binary=:all: \
  --platform manylinux2014_aarch64 --python-version 311 --implementation cp \
  --abi cp311 cmake==3.31.6 -r requirements-build.txt
```

On27, after loading0env, install from the uploaded wheel directory with
`--no-index --find-links <uploaded-directory> --target .cache/build-python`.
Prepend this project target to PYTHONPATH and its bin directory to PATH only in
the current build shell. This preserves the base py311_tyf environment. Pinned
build packages are required: build1.2.2.post1/setuptools79.0.1/wheel0.47.0/
pybind113.0.4. Additional build dependencies are retained in the uploaded wheelhouse.

The validated upload is `.cache/offline-build-001.tar`, unpacked into
`.cache/wheelhouse-001/` under `/home/tyf/Q4_remote_test/DeepEP-LL-g4-small-b`.
SHMEM JSON source was already present after the earlier attempt, so it was reused.
Do not restart recursive LLVM/Triton downloads merely to compile the native LL
candidate. Preserve pinned submodule revisions and explicit SDK identity checks.

Final result: SDK build and complete ON wheel compile/link/package passed. Wheel
was not installed and no NPU communication was launched. Logs, wheel hash and
seven-mode cold ON/OFF object results are recorded in BUILD_DISPATCH_SPLIT.md.
