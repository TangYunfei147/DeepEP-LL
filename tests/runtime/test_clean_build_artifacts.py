"""Destructive cleanup is confined to explicit, non-redirected output roots."""
from pathlib import Path
import os
import shutil
import stat
import subprocess
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import clean_build_artifacts as clean


def create(root, relative, value="keep"):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)
    return path


def symlink(link, target):
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"this environment cannot create a symlink: {error}")


def test_allowlisted_outputs_only_and_relocation_with_spaces(tmp_path, monkeypatch):
    root = tmp_path / "relocated project with spaces"
    preserved = ["third-party/shmem/src/file.cpp", "third-party/shmem/scripts/build.sh",
                 "third-party/shmem/3rdparty/json/header.h", "third-party/shmem/.git/config",
                 "diagnostic-runs/old/wheels/off.whl", ".logs/build.log", "dist/old.whl"]
    deleted = ["build/native/object.o", "third-party/shmem/build/cache.txt",
               "third-party/shmem/install/shmem/lib.so", "third-party/shmem/ci/release/package.run"]
    for relative in preserved + deleted:
        create(root, relative)
    monkeypatch.chdir(tmp_path)
    clean.clean_artifacts(root, wheel=True, sdk=True)
    assert all((root / relative).exists() for relative in preserved)
    assert all(not (root / relative).exists() for relative in deleted)
    assert clean.clean_directory(root, root / "build") is False


def test_native_subdirectory_cleanup_preserves_other_builds(tmp_path):
    old = create(tmp_path, "build/selected/temp.o")
    other = create(tmp_path, "build/other/keep.o")
    assert clean.clean_directory(tmp_path, old.parent, permitted_root=tmp_path / "build")
    assert not old.exists() and other.exists()


@pytest.mark.parametrize("relative", [".", ".git", "third-party", "third-party/shmem", "third-party/shmem/src",
                                      "third-party/shmem/3rdparty", "diagnostic-runs", ".logs", "dist"])
def test_non_artifact_targets_rejected(tmp_path, relative):
    target = tmp_path / relative
    target.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError, match="allowed build-output"):
        clean.clean_directory(tmp_path, target)
    assert target.exists()


def test_external_or_parent_traversal_rejected(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    external = create(tmp_path, "external/file.txt")
    for target in (external.parent, root / "build/../../external"):
        with pytest.raises(ValueError):
            clean.clean_directory(root, target, permitted_root=root / "build")
    assert external.exists()


def test_output_root_symlink_escape_rejected_without_touching_target(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    outside = create(tmp_path, "external/file.txt")
    symlink(root / "build", outside.parent)
    with pytest.raises(ValueError, match="symlink|junction|reparse"):
        clean.clean_directory(root, root / "build")
    assert outside.exists() and (root / "build").is_symlink()


def test_ancestor_symlink_rejected(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    outside = create(tmp_path, "external/shmem/build/file.txt")
    symlink(root / "third-party", tmp_path / "external")
    with pytest.raises(ValueError, match="symlink|junction|reparse"):
        clean.clean_directory(root, root / "third-party/shmem/build")
    assert outside.exists()


def test_child_symlink_and_broken_link_rejected_before_any_deletion(tmp_path):
    root = tmp_path / "project"
    keep = create(root, "build/first.bin")
    symlink(root / "build/missing-link", tmp_path / "missing")
    with pytest.raises(ValueError, match="symlink|junction|reparse"):
        clean.clean_directory(root, root / "build")
    assert keep.exists()


def test_all_sdk_targets_prevalidated_before_deleting_first(tmp_path):
    keep = create(tmp_path, "third-party/shmem/build/keep.o")
    outside = create(tmp_path, "outside/keep.so")
    symlink(tmp_path / "third-party/shmem/install", outside.parent)
    with pytest.raises(ValueError):
        clean.clean_artifacts(tmp_path, sdk=True)
    assert keep.exists() and outside.exists()


def test_windows_reparse_attribute_rejected_without_platform_dependency(tmp_path, monkeypatch):
    target = tmp_path / "build"
    target.mkdir()
    original = Path.lstat
    def lstat(path):
        if path == target:
            return types.SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
        return original(path)
    monkeypatch.setattr(Path, "lstat", lstat)
    with pytest.raises(ValueError, match="reparse"):
        clean.validate_directory(tmp_path, target)
    assert target.exists()


@pytest.mark.parametrize("location", ["build", "third-party", "build/broken-child"])
def test_symlink_lstat_rejection_without_symlink_creation_privilege(tmp_path, monkeypatch, location):
    target = tmp_path / location
    target.mkdir(parents=True)
    sentinel = create(tmp_path, "external/keep.txt")
    original = Path.lstat
    def lstat(path):
        if path == target:
            return types.SimpleNamespace(st_mode=stat.S_IFLNK, st_file_attributes=0)
        return original(path)
    monkeypatch.setattr(Path, "lstat", lstat)
    selected = tmp_path / "third-party/shmem/build" if location == "third-party" else tmp_path / "build"
    with pytest.raises(ValueError, match="symlink"):
        clean.clean_directory(tmp_path, selected)
    assert sentinel.exists()


@pytest.mark.skipif(os.name != "nt", reason="actual Windows junction test")
def test_actual_windows_junction_escape_is_rejected(tmp_path):
    shell = shutil.which("powershell") or shutil.which("pwsh")
    if not shell:
        pytest.skip("PowerShell unavailable for native junction creation")
    root = tmp_path / "project"
    root.mkdir()
    outside = create(tmp_path, "external/keep.txt")
    link = root / "build"
    environment = dict(os.environ, DEEPEP_TEST_JUNCTION_LINK=str(link),
                       DEEPEP_TEST_JUNCTION_TARGET=str(outside.parent))
    subprocess.run([shell, "-NoProfile", "-NonInteractive", "-Command",
        "New-Item -ItemType Junction -Path $env:DEEPEP_TEST_JUNCTION_LINK "
        "-Target $env:DEEPEP_TEST_JUNCTION_TARGET -ErrorAction Stop | Out-Null"],
        env=environment, check=True, capture_output=True, text=True)
    with pytest.raises(ValueError, match="junction|reparse"):
        clean.clean_directory(root, link)
    assert outside.exists()


def test_cli_resolves_script_root_instead_of_working_directory(tmp_path):
    moved = tmp_path / "moved project"
    script = moved / "scripts/clean_build_artifacts.py"
    script.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "scripts/clean_build_artifacts.py", script)
    deleted = create(moved, "build/selected.o")
    preserved = create(tmp_path, "build/unrelated.o")
    subprocess.run([sys.executable, str(script), "--wheel"], cwd=tmp_path, check=True)
    assert not deleted.exists() and preserved.exists()


def test_regular_file_is_not_recursively_removed(tmp_path):
    target = create(tmp_path, "build")
    with pytest.raises(ValueError, match="not a directory"):
        clean.clean_directory(tmp_path, target)
    assert target.exists()
