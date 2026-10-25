"""Branch-switch leftovers must not change the identity of tracked tests."""
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def source(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("fingerprint_build", ROOT / "build_support.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    root = tmp_path / "checkout with spaces"
    root.mkdir()
    for name in ("VERSION", "setup.py", "pyproject.toml", "build_support.py", "CMakeLists.txt",
                 "dependencies.lock.json", "MANIFEST.in", "requirements-dev.txt", "requirements-build.txt",
                 ".gitmodules", "experiments/ep_gmm_fused/tests/requirements.txt",
                 "third-party/deepseek-deepep.LICENSE", "third-party/cann.LICENSE"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n")
    tracked = root / "tests/tracked.py"
    tracked.parent.mkdir()
    tracked.write_text("print('tracked')\n")
    monkeypatch.setattr(build, "ROOT", root)
    return build, root, tracked


def initialize_git(root):
    if not shutil.which("git"):
        pytest.skip("git required for checkout identity tests")
    env = dict(os.environ)
    env.pop("LD_LIBRARY_PATH", None)
    env.pop("LD_PRELOAD", None)
    subprocess.run(["git", "init", "--quiet", str(root)], env=env, check=True)
    subprocess.run(["git", "-C", str(root), "add", "tests/tracked.py"], env=env, check=True)


def test_untracked_old_probe_directories_do_not_change_checkout_fingerprint(source):
    build, root, tracked = source
    initialize_git(root)
    before = build.source_fingerprint()
    (root / "core").write_bytes(b"preserved diagnostic data")
    for directory in ("communication", "metadata", "native_ll", "network_probe"):
        path = root / "tests/ll_probe" / directory / "old.py"
        path.parent.mkdir(parents=True)
        path.write_text("print('old branch')\n")
    assert build.source_fingerprint() == before
    assert (root / "core").read_bytes() == b"preserved diagnostic data"
    assert len(list((root / "tests/ll_probe").rglob("old.py"))) == 4
    tracked.write_text("print('modified tracked test')\n")
    assert build.source_fingerprint() != before


def test_new_test_enters_fingerprint_once_added_to_git(source):
    build, root, _ = source
    initialize_git(root)
    before = build.source_fingerprint()
    path = root / "tests/new.py"
    path.write_text("print('new')\n")
    assert build.source_fingerprint() == before
    subprocess.run(["git", "-C", str(root), "add", "tests/new.py"], check=True)
    assert build.source_fingerprint() != before


@pytest.mark.parametrize("directory", ["csrc", "deep_ep", "scripts"])
def test_untracked_build_and_runtime_inputs_are_still_hashed(source, directory):
    build, root, _ = source
    initialize_git(root)
    before = build.source_fingerprint()
    path = root / directory / "new.h"
    path.parent.mkdir()
    path.write_text("new build input\n")
    assert build.source_fingerprint() != before


def test_source_archive_needs_no_git_and_hashes_its_delivered_tests(source, monkeypatch):
    build, root, _ = source
    monkeypatch.setattr(build.subprocess, "check_output", lambda *a, **k: pytest.fail("archive invoked Git"))
    before = build.source_fingerprint()
    (root / "tests/archive.py").write_text("print('archive input')\n")
    assert build.source_fingerprint() != before


def test_checkout_git_failure_is_not_silently_ignored(source, monkeypatch):
    build, root, _ = source
    initialize_git(root)

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(128, args[0])

    monkeypatch.setattr(build.subprocess, "check_output", fail)
    with pytest.raises(subprocess.CalledProcessError):
        build.source_fingerprint()
