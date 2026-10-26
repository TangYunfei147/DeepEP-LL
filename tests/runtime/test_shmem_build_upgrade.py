"""Every SHMEM invocation cleans only build outputs, then propagates compiler errors."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import build_shmem


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "project with spaces"
    source = root / "third-party/shmem"
    (source / "scripts").mkdir(parents=True)
    (source / ".git").mkdir()
    (source / "scripts/build.sh").write_text("vendor build placeholder")
    (source / "source.cpp").write_text("user local edits")
    (source / "3rdparty/json").mkdir(parents=True)
    (source / "3rdparty/json/source.hpp").write_text("keep dependency source")
    (root / "diagnostic-runs/old/wheels").mkdir(parents=True)
    (root / "diagnostic-runs/old/wheels/keep.whl").write_text("keep result")
    (root / "dependencies.lock.json").write_text(json.dumps(dict(
        shmem=dict(revision="different-locked-revision"), cann_version="9.3.0", target="Ascend950")))
    monkeypatch.setattr(build_shmem.platform, "system", lambda: "Linux")
    monkeypatch.setattr(build_shmem, "cann_version", lambda _: dict(Version="9.3.0", timestamp="test"))
    monkeypatch.setattr(build_shmem.subprocess, "check_output", lambda args, **kw: "arbitrary-local-revision")
    monkeypatch.setattr(build_shmem.shutil, "which", lambda name: sys.executable)
    monkeypatch.setenv("CMAKE_BUILD_PARALLEL_LEVEL", "4")
    for key in ("TMPDIR", "XDG_CACHE_HOME", "ASCEND_CACHE_PATH", "ASCEND_WORK_PATH",
                "ASCEND_PROCESS_LOG_PATH", "SHMEM_LOG_PATH"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)  # Neither cleanup nor build depends on caller cwd.
    return root, source


def populate_old_outputs(source):
    for name in ("build", "install", "ci"):
        (source / name).mkdir(exist_ok=True)
        (source / name / "old.bin").write_text("stale generated data")
    manifest = source / "install/shmem/deepep-sdk.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("not even valid JSON; must not be read")


@pytest.mark.parametrize("options", [dict(), dict(rebuild=True), dict(if_needed=True),
                                     dict(if_needed=False), dict(allow_local_changes=True)])
def test_every_call_cleans_and_builds_regardless_of_old_options(project, monkeypatch, options):
    root, source = project
    commands = []
    def run(argv, *, cwd, env, check):
        commands.append(argv)
        assert cwd == source and check is True
        assert argv == ["bash", "-e", "scripts/build.sh", "-soc_type", "Ascend950"]
        assert all(not (source / name).exists() for name in ("build", "install", "ci"))
        assert env["CCACHE_DISABLE"] == "1"
        for key in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "ASCEND_CACHE_PATH",
                    "ASCEND_WORK_PATH", "ASCEND_PROCESS_LOG_PATH", "SHMEM_LOG_PATH"):
            assert Path(env[key]).resolve().is_relative_to(root.resolve())
        (source / "install/shmem/lib").mkdir(parents=True)
        (source / "install/shmem/lib/libshmem.so").write_text("fresh SDK")
    monkeypatch.setattr(build_shmem.subprocess, "run", run)
    for _ in range(2):
        populate_old_outputs(source)
        assert build_shmem.prepare(root, **options) == source / "install/shmem"
        assert not (source / "install/shmem/deepep-sdk.json").exists()
        assert (source / "source.cpp").read_text() == "user local edits"
        assert (source / "3rdparty/json/source.hpp").exists()
        assert (source / ".git").is_dir()
        assert (root / "diagnostic-runs/old/wheels/keep.whl").exists()
    assert len(commands) == 2


@pytest.mark.parametrize("error", [OSError("no git metadata"), subprocess.CalledProcessError(128, "git"),
                                   subprocess.TimeoutExpired("git", 10)])
def test_git_revision_is_only_best_effort_log(project, monkeypatch, error):
    root, source = project
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(build_shmem.subprocess, "check_output", fail)
    called = []
    monkeypatch.setattr(build_shmem.subprocess, "run", lambda *args, **kwargs: called.append(args[0]))
    assert build_shmem.prepare(root) == source / "install/shmem"
    assert len(called) == 1


def test_failed_vendor_build_stops_without_manifest_or_retry(project, monkeypatch):
    root, source = project
    populate_old_outputs(source)
    attempts = []
    def fail(argv, **kwargs):
        attempts.append(argv)
        assert all(not (source / name).exists() for name in ("build", "install", "ci"))
        (source / "build").mkdir()
        (source / "build/error.log").write_text("compiler failed")
        raise subprocess.CalledProcessError(7, argv)
    monkeypatch.setattr(build_shmem.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError) as error:
        build_shmem.prepare(root)
    assert error.value.returncode == 7 and len(attempts) == 1
    assert (source / "build/error.log").read_text() == "compiler failed"
    assert not (source / "install/shmem/deepep-sdk.json").exists()


def test_failed_prerequisite_does_not_clean_existing_outputs(project, monkeypatch):
    root, source = project
    populate_old_outputs(source)
    def fail(_):
        raise RuntimeError("load CANN")
    monkeypatch.setattr(build_shmem, "cann_version", fail)
    monkeypatch.setattr(build_shmem.subprocess, "run", lambda *a, **k: pytest.fail("must not build"))
    with pytest.raises(RuntimeError, match="CANN"):
        build_shmem.prepare(root)
    assert (source / "build/old.bin").exists()
    assert (source / "install/shmem/deepep-sdk.json").exists()


def test_sdk_identity_and_source_edits_are_not_build_admission_checks():
    source = (ROOT / "scripts/build_shmem.py").read_text(encoding="utf-8")
    for removed in ("record_shmem_sdk", "shmem_sdk_reuse", "validate_shmem_sdk", "source_patches",
                    'git("status"', 'lock["shmem"]', "deepep-sdk.json"):
        assert removed not in source


def test_linux_and_parallelism_checks_remain_before_cleanup(project, monkeypatch):
    root, source = project
    populate_old_outputs(source)
    monkeypatch.setattr(build_shmem.platform, "system", lambda: "Windows")
    with pytest.raises(RuntimeError, match="Linux"):
        build_shmem.prepare(root)
    monkeypatch.setattr(build_shmem.platform, "system", lambda: "Linux")
    monkeypatch.setenv("CMAKE_BUILD_PARALLEL_LEVEL", "0")
    with pytest.raises(ValueError, match="PARALLEL"):
        build_shmem.prepare(root)
    assert (source / "build/old.bin").exists()


def test_grouped_run_cache_and_log_overrides_are_preserved(project, monkeypatch):
    root, _ = project
    expected = {}
    for key in ("TMPDIR", "XDG_CACHE_HOME", "ASCEND_CACHE_PATH", "ASCEND_WORK_PATH",
                "ASCEND_PROCESS_LOG_PATH", "SHMEM_LOG_PATH"):
        expected[key] = str(root / "diagnostic-runs/group 001" / key.lower())
        monkeypatch.setenv(key, expected[key])
    monkeypatch.setenv("TMP", str(root.parent / "system-temp-ignored"))
    monkeypatch.setenv("TEMP", str(root.parent / "system-temp-ignored"))
    def run(argv, *, cwd, env, check):
        for key, value in expected.items():
            assert env[key] == value and Path(value).is_dir()
        assert env["TMP"] == expected["TMPDIR"] and env["TEMP"] == expected["TMPDIR"]
    monkeypatch.setattr(build_shmem.subprocess, "run", run)
    build_shmem.prepare(root)


@pytest.mark.parametrize("key", ["TMPDIR", "XDG_CACHE_HOME", "ASCEND_CACHE_PATH", "ASCEND_WORK_PATH",
                                 "ASCEND_PROCESS_LOG_PATH", "SHMEM_LOG_PATH"])
def test_outside_override_rejected_before_cleaning(project, monkeypatch, key):
    root, source = project
    populate_old_outputs(source)
    monkeypatch.setenv(key, str(root.parent / "external-output"))
    monkeypatch.setattr(build_shmem.subprocess, "run", lambda *a, **k: pytest.fail("must not build"))
    with pytest.raises(ValueError, match="outside"):
        build_shmem.prepare(root)
    assert (source / "build/old.bin").exists()
