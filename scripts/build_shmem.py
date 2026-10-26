# Copyright (c) 2026, Lu Lu
# Modified by huangxiaolan 2026

"""Clean this project's SHMEM outputs and compile on every invocation.

SDK manifests, source revisions, hashes and local source edits are not admission
checks. Git submodule update selects the source; this script always builds the
checked-out source and never reuses an installed SDK. Pip/wheel does not invoke
this dependency build implicitly.
"""

import argparse
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import tempfile

try:
    from scripts.clean_build_artifacts import assert_local_path, clean_artifacts
except ModuleNotFoundError:
    from clean_build_artifacts import assert_local_path, clean_artifacts

ROOT = Path(__file__).resolve().parents[1]


def cann_version(expected):
    cann = os.environ.get("ASCEND_HOME_PATH")
    if not cann or not (Path(cann) / "compiler/version.info").is_file():
        raise RuntimeError("Source the CANN set_env.sh first (ASCEND_HOME_PATH is missing or invalid)")
    version = dict(
        line.split("=", 1) for line in (Path(cann) / "compiler/version.info").read_text().splitlines() if "=" in line
    )
    if version.get("Version") not in {expected, "9.3.0"}:
        raise RuntimeError(f"This build requires CANN {expected}")
    return version


def _revision_for_log(source):
    try:
        return subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL, timeout=10,
        ).strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _output_path(root, name, default):
    value = os.environ.get(name)
    return assert_local_path(root, Path(value) if value else default)


def prepare(root=ROOT, *, rebuild=False, if_needed=True, allow_local_changes=False):
    """Compatibility arguments are accepted; every call is a clean build."""
    if platform.system() != "Linux":
        raise RuntimeError("SHMEM compilation requires Linux and the CANN SDK")
    root = assert_local_path(Path(root), Path(root))
    lock = json.loads((root / "dependencies.lock.json").read_text(encoding="utf-8"))
    source = assert_local_path(root, root / "third-party/shmem")
    script = assert_local_path(root, source / "scripts/build.sh")
    if not script.is_file():
        raise RuntimeError("Initialize dependencies first: git submodule update --init --recursive --depth 1")
    # The upstream native build may download dependencies here; preserve them
    # and reject redirects without treating this source cache as a clean target.
    assert_local_path(root, source / "3rdparty")
    version = cann_version(lock["cann_version"])
    jobs = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", "4")
    if not jobs.isdecimal() or not 1 <= int(jobs) <= 64:
        raise ValueError("CMAKE_BUILD_PARALLEL_LEVEL must be in [1, 64]")
    git_executable = shutil.which("git")
    if git_executable is None:
        raise RuntimeError("Git is required to prepare SHMEM dependencies")
    # A grouped run owns its cache/log locations. Honor only explicit project-
    # local overrides, never the shell's ambient system TMP/TEMP directories.
    temporary_root = _output_path(root, "TMPDIR", root / ".cache/tmp")
    cache = _output_path(root, "XDG_CACHE_HOME", root / ".cache/shmem-build")
    ascend_cache = _output_path(root, "ASCEND_CACHE_PATH", cache / "ascend")
    default_logs = root / ".logs/shmem-build"
    ascend_work = _output_path(root, "ASCEND_WORK_PATH", default_logs / "ascend")
    ascend_logs = _output_path(root, "ASCEND_PROCESS_LOG_PATH", default_logs / "ascend")
    shmem_logs = _output_path(root, "SHMEM_LOG_PATH", default_logs / "shmem")
    revision = _revision_for_log(source)
    print(f"[SHMEM] clean build; source={revision} CANN={version['Version']} target={lock['target']}", flush=True)
    # No installed SDK, source/manifest identity validation or recording gate.
    # Remove only verified build outputs, never the source submodule.
    clean_artifacts(root, sdk=True)
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="deepep-shmem-git-", dir=temporary_root) as git_dir:
        wrapper = Path(git_dir) / "git"
        wrapper.write_text(
            "#!/bin/sh\nunset LD_LIBRARY_PATH LD_PRELOAD\n"
            f'exec {shlex.quote(str(Path(git_executable).resolve()))} "$@"\n'
        )
        wrapper.chmod(0o700)
        for path in (cache, ascend_cache, ascend_work, ascend_logs, shmem_logs):
            path.mkdir(parents=True, exist_ok=True)
        build_env = dict(os.environ, CMAKE_BUILD_PARALLEL_LEVEL=jobs,
                         TMPDIR=str(temporary_root), TMP=str(temporary_root), TEMP=str(temporary_root),
                         XDG_CACHE_HOME=str(cache), ASCEND_CACHE_PATH=str(ascend_cache),
                         ASCEND_WORK_PATH=str(ascend_work),
                         ASCEND_PROCESS_LOG_PATH=str(ascend_logs),
                         SHMEM_LOG_PATH=str(shmem_logs), CCACHE_DISABLE="1")
        build_env["PATH"] = git_dir + os.pathsep + build_env.get("PATH", os.defpath)
        # Safe Python cleanup already removed the exact upstream output roots.
        # -e propagates any configure/build failure instead of producing success
        # after an intermediate error. No docs/examples/package mode is enabled.
        subprocess.run(["bash", "-e", "scripts/build.sh", "-soc_type", lock["target"]],
                       cwd=source, env=build_env, check=True)
    sdk = source / "install/shmem"
    print(f"[SHMEM] Build command completed: {sdk}", flush=True)
    print("[SHMEM] Add third-party/shmem/install/shmem/lib to LD_LIBRARY_PATH before loading the wheel.")
    return sdk


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="Compatibility alias: every invocation already cleans and rebuilds")
    parser.add_argument("--if-needed", action="store_true", help="Compatibility alias: every invocation cleans and rebuilds; no reuse")
    parser.add_argument("--allow-local-changes", action="store_true", help="Compatibility alias: checked-out local source edits are always built")
    args = parser.parse_args()
    prepare(rebuild=args.rebuild, if_needed=args.if_needed, allow_local_changes=args.allow_local_changes)
