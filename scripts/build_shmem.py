# Copyright (c) 2026, Lu Lu
# Modified by huangxiaolan 2026

"""Build or safely reuse the pinned NDR HNS_1825 SDK; pip never invokes this script."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import tempfile

from record_shmem_sdk import record
import shmem_barrier_patch

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


def source_patches(source):
    """Fingerprint tracked edits and untracked inputs without modifying Git."""

    def git(*args):
        return subprocess.check_output(["git", "-C", str(source), *args])

    if not git("status", "--porcelain", "--untracked-files=normal").strip():
        return []
    digest = hashlib.sha256(git("diff", "HEAD", "--binary", "--no-ext-diff", "--no-textconv"))
    for name in sorted(git("ls-files", "--others", "--exclude-standard", "-z").split(b"\0")):
        if not name:
            continue
        path = source / os.fsdecode(name)
        if path.is_symlink() or not path.resolve().is_relative_to(source.resolve()):
            raise RuntimeError(f"Untracked SHMEM input must be a local regular file: {path}")
        digest.update(name + b"\0" + hashlib.sha256(path.read_bytes()).digest())
    if shmem_barrier_patch.is_managed_only(source):
        return [shmem_barrier_patch.patch_identity()]
    _, expected = shmem_barrier_patch.expected_source(source)
    managed = ([shmem_barrier_patch.patch_identity()]
               if (source / shmem_barrier_patch.TARGET).read_text(encoding="utf-8") == expected else [])
    return managed + [{"kind": "local-worktree", "sha256": digest.hexdigest()}]


def prepare(root=ROOT, *, rebuild=False, if_needed=True, allow_local_changes=False, rdma_backend="HNS_1825"):
    if platform.system() != "Linux":
        raise RuntimeError("SHMEM compilation requires Linux and the pinned CANN SDK")
    root = Path(root).resolve()
    # None from older launchers means use the branch default, not a MTE SDK.
    rdma_backend = rdma_backend or "HNS_1825"
    if rdma_backend != "HNS_1825":
        raise ValueError("NDR requires the HNS_1825 SDK")
    lock = json.loads((root / "dependencies.lock.json").read_text(encoding="utf-8"))
    source = root / "third-party/shmem"
    if not (source / ".git").exists() or not (source / "scripts/build.sh").is_file():
        raise RuntimeError("Initialize dependencies first: git submodule update --init --recursive")
    if source.resolve() != source or source.parent.resolve() != source.parent:
        raise RuntimeError("SHMEM source must not be a symlink to another checkout")

    def git(*args):
        return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()

    if Path(git("rev-parse", "--show-toplevel")).resolve() != source:
        raise RuntimeError("third-party/shmem must be its own initialized Git submodule")
    revision = git("rev-parse", "HEAD")
    if revision != lock["shmem"]["revision"]:
        raise RuntimeError("SHMEM revision differs from dependencies.lock.json; do not build a moving branch")
    shmem_barrier_patch.apply(source)
    patches = source_patches(source)
    if any(patch != shmem_barrier_patch.patch_identity() for patch in patches) and not allow_local_changes:
        raise RuntimeError("SHMEM has local changes; pass --allow-local-changes to build them explicitly "
                           "(cluster launcher: --allow-local-shmem)")

    version = cann_version(lock["cann_version"])
    rdma_build = {"rdma_backend": rdma_backend} if rdma_backend else {}
    jobs = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", "4")
    if not jobs.isdecimal() or not 1 <= int(jobs) <= 64:
        raise ValueError("CMAKE_BUILD_PARALLEL_LEVEL must be in [1, 64]")

    # 上游脚本会清空 build/install，并生成 ci 打包产物；禁止跳转到其他目录。
    for name in ("build", "install", "ci", "3rdparty"):
        path = source / name
        if path.resolve() != path:
            raise RuntimeError(f"Refusing SHMEM build through a symlink: {path}")
    manifest = source / "install/shmem/deepep-sdk.json"
    if not rebuild and manifest.is_file():
        identity = json.loads(manifest.read_text(encoding="utf-8"))
        expected_build = {
            "cann_version": version["Version"],
            "cann_timestamp": version.get("timestamp", "unknown"),
            "target": lock["target"],
            "source_patches": patches,
            **rdma_build,
        }
        if identity.get("revision") != revision or identity.get("build") != expected_build:
            raise RuntimeError(
                "Existing SHMEM build identity changed; stop SDK users and rebuild explicitly "
                "with --rebuild (cluster launcher: --rebuild-shmem)"
            )
        # 复用也校验全部头文件/库哈希，不把存在 manifest 当成构建成功。
        import sys

        sys.path.insert(0, str(ROOT))
        from build_support import validate_shmem_sdk

        validate_shmem_sdk(manifest.parent, lock["shmem"])
        print(f"[SHMEM] Reusing validated SDK: {manifest.parent}", flush=True)
        return manifest
    if not rebuild and any((source / name).exists() for name in ("build", "install")):
        raise FileExistsError(
            "SHMEM build/install already exists; reuse its SDK, or stop users of that SDK "
            "and pass --rebuild to regenerate it (then rebuild dependent wheels)"
        )
    print(f"[SHMEM] source={revision} CANN={version['Version']} target={lock['target']}", flush=True)
    if patches:
        print(f"[SHMEM] Building local source modifications: {json.dumps(patches)}", flush=True)
    git_executable = shutil.which("git")
    if git_executable is None:
        raise RuntimeError("Git is required to prepare SHMEM dependencies")
    # 上游脚本会调用 Git 下载 json 等依赖；只隔离 Git 的库路径，保留编译器所需的 CANN 环境。
    temporary_root = root / ".cache" / "tmp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="deepep-shmem-git-", dir=temporary_root) as git_dir:
        wrapper = Path(git_dir) / "git"
        wrapper.write_text(
            "#!/bin/sh\nunset LD_LIBRARY_PATH LD_PRELOAD\n"
            f'exec {shlex.quote(str(Path(git_executable).resolve()))} "$@"\n'
        )
        wrapper.chmod(0o700)
        build_env = dict(os.environ, CMAKE_BUILD_PARALLEL_LEVEL=jobs)
        build_env["PATH"] = git_dir + os.pathsep + build_env.get("PATH", os.defpath)
        # -e 防止上游中间命令失败后继续打包，随后误登记旧库为成功产物。
        subprocess.run(
            ["bash", "-e", "scripts/build.sh", "-soc_type", lock["target"]]
            + (["-enable_rdma", "-rdma_backend", rdma_backend] if rdma_backend else []),
            cwd=source,
            env=build_env,
            check=True,
        )
    sdk = source / "install/shmem"
    if git("rev-parse", "HEAD") != revision or source_patches(source) != patches:
        raise RuntimeError("SHMEM source changed during compilation; SDK identity was not recorded")
    manifest = record(
        sdk,
        revision,
        build_info={
            "cann_version": version["Version"],
            "cann_timestamp": version.get("timestamp", "unknown"),
            "target": lock["target"],
            "source_patches": patches,
            **rdma_build,
        },
    )
    print(f"[SHMEM] SDK identity: {manifest}", flush=True)
    print("[SHMEM] Add third-party/shmem/install/shmem/lib to LD_LIBRARY_PATH before loading the wheel.")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="Regenerate this submodule's build/install artifacts")
    parser.add_argument("--if-needed", action="store_true", help="Compatibility alias: verified SDK reuse is now the default")
    parser.add_argument("--rdma-backend", choices=["HNS_1825"], default="HNS_1825", help="Compatibility option: NDR always builds HNS_1825 with the native SDK QP limit")
    parser.add_argument(
        "--allow-local-changes",
        action="store_true",
        help="Build local edits on the pinned commit and record their source fingerprint",
    )
    args = parser.parse_args()
    prepare(rebuild=args.rebuild, if_needed=args.if_needed, allow_local_changes=args.allow_local_changes,
            rdma_backend=args.rdma_backend)
