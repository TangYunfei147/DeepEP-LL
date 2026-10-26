#!/usr/bin/env python3
"""Safely remove only this project's wheel or SHMEM build-output directories."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import stat

ROOT = Path(__file__).resolve().parents[1]
SDK_OUTPUTS = ("build", "install", "ci")


def _absolute(path: Path, base: Path | None = None) -> Path:
    path = Path(path)
    if not path.is_absolute():
        if base is None:
            raise ValueError("The project root must be an absolute path")
        path = base / path
    if ".." in path.parts:
        raise ValueError(f"Refusing a build-output path containing '..': {path}")
    return Path(os.path.abspath(os.fspath(path)))


def _reject_link(path: Path) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & reparse:
        raise ValueError(f"Refusing symlink/junction/reparse build path: {path}")
    if hasattr(path, "is_junction") and path.is_junction():
        raise ValueError(f"Refusing junction build path: {path}")


def assert_local_path(root: Path, path: Path) -> Path:
    """Check a project-relative location before writing or deleting through it."""
    root = _absolute(root)
    path = _absolute(path, root)
    for ancestor in (*reversed(root.parents), root):
        _reject_link(ancestor)
    if not root.is_dir():
        raise ValueError(f"Project root must be an existing directory: {root}")
    if not path.is_relative_to(root):
        raise ValueError(f"Build path is outside the project root: {path}")
    current = root
    for part in path.relative_to(root).parts:
        current = current / part
        _reject_link(current)
    resolved_root = root.resolve(strict=True)
    resolved_path = path.resolve(strict=False)
    if resolved_path != path or resolved_root != root or not resolved_path.is_relative_to(resolved_root):
        raise ValueError(f"Build path resolves outside its intended project location: {path}")
    return path


def artifact_roots(root: Path) -> tuple[Path, ...]:
    root = assert_local_path(root, root)
    return (root / "build", *(root / "third-party/shmem" / name for name in SDK_OUTPUTS))


def _check_tree(path: Path) -> None:
    """Preflight the whole subtree without following directory links."""
    _reject_link(path)
    if not path.exists():
        return
    if not path.is_dir():
        raise ValueError(f"Build-output target is not a directory: {path}")
    pending = [path]
    while pending:
        parent = pending.pop()
        with os.scandir(parent) as entries:
            for entry in entries:
                child = Path(entry.path)
                _reject_link(child)
                if entry.is_dir(follow_symlinks=False):
                    pending.append(child)


def validate_directory(root: Path, path: Path, *, permitted_root: Path | None = None) -> Path:
    """Validate against both the fixed and the caller-selected output allowlist."""
    root = assert_local_path(root, root)
    target = assert_local_path(root, path)
    allowed = artifact_roots(root)
    permitted = target if permitted_root is None else assert_local_path(root, permitted_root)
    if permitted not in allowed:
        raise ValueError(f"Not an allowed build-output root: {permitted}")
    if not target.is_relative_to(permitted):
        raise ValueError(f"Build target is outside the permitted output root: {target}")
    if target == root:
        raise ValueError("Refusing to remove the project root")
    _check_tree(target)
    return target


def clean_directory(root: Path, path: Path, *, permitted_root: Path | None = None) -> bool:
    """Remove one checked directory; never traverse a symlink or junction.

    Native builders may delete a fingerprint child with permitted_root=root/build.
    SDK callers remove only the exact three known output roots. Saved wheels,
    diagnostic runs, logs, dependency downloads and source trees are excluded.
    """
    target = validate_directory(root, path, permitted_root=permitted_root)
    if not target.exists():
        return False
    shutil.rmtree(target)
    return True


def clean_artifacts(root: Path = ROOT, *, wheel: bool = False, sdk: bool = False) -> list[Path]:
    root = assert_local_path(root, root)
    targets = ([root / "build"] if wheel else []) + (
        [root / "third-party/shmem" / name for name in SDK_OUTPUTS] if sdk else [])
    # Reject a bad later target before deleting any earlier valid directory.
    for target in targets:
        validate_directory(root, target)
    for target in targets:
        removed = clean_directory(root, target)
        print(f"[clean] {'removed' if removed else 'absent'} {target}", flush=True)
    return targets


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", action="store_true", help="Remove only this project's build/ tree")
    parser.add_argument("--sdk", action="store_true", help="Remove only SHMEM build/, install/ and ci/ outputs")
    args = parser.parse_args(argv)
    if not (args.wheel or args.sdk):
        parser.error("choose --wheel, --sdk, or both")
    clean_artifacts(wheel=args.wheel, sdk=args.sdk)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
