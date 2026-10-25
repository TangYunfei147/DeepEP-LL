"""SHMEM upgrades explicitly discard the old native cache before recording an SDK."""
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
    source = tmp_path / "third-party/shmem"
    (source / "scripts").mkdir(parents=True)
    (source / ".git").mkdir()
    (source / "scripts/build.sh").write_text("vendor build placeholder")
    (tmp_path / "dependencies.lock.json").write_text(json.dumps(dict(
        shmem=dict(revision="new-mainline"), cann_version="9.3.0", target="Ascend950")))
    monkeypatch.setattr(build_shmem.platform, "system", lambda: "Linux")
    monkeypatch.setattr(build_shmem, "source_patches", lambda _: [])
    monkeypatch.setattr(build_shmem, "cann_version", lambda _: dict(Version="9.3.0", timestamp="test"))
    monkeypatch.setattr(build_shmem.subprocess, "check_output", lambda args, **kw:
                        str(source) if args[-1] == "--show-toplevel" else "new-mainline")
    monkeypatch.setattr(build_shmem.shutil, "which", lambda name: sys.executable)
    monkeypatch.setenv("CMAKE_BUILD_PARALLEL_LEVEL", "4")
    return tmp_path, source


@pytest.mark.parametrize("rebuild", [False, True])
def test_new_build_and_explicit_rebuild_have_local_outputs(project, monkeypatch, rebuild):
    root, source = project
    if rebuild:
        (source / "build").mkdir()
        (source / "build/CMakeCache.txt").write_text("old revision cache")
        old = source / "install/shmem/deepep-sdk.json"
        old.parent.mkdir(parents=True)
        old.write_text(json.dumps(dict(revision="old-revision")))
    commands, records = [], []
    def run(argv, *, cwd, env, check):
        commands.append(argv)
        assert cwd == source and check is True
        assert ("-clean" in argv) is rebuild
        assert argv[:5] == ["bash", "-e", "scripts/build.sh", "-soc_type", "Ascend950"]
        assert not any(mode in argv for mode in ("-full", "-gendoc", "-onlygendoc", "-examples", "-package"))
        for key in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "ASCEND_CACHE_PATH",
                    "ASCEND_WORK_PATH", "ASCEND_PROCESS_LOG_PATH", "SHMEM_LOG_PATH"):
            assert Path(env[key]).resolve().is_relative_to(root.resolve())
        # The vendor always recreates install/; mimic only its manifest removal.
        manifest = source / "install/shmem/deepep-sdk.json"
        if manifest.exists():
            manifest.unlink()
        manifest.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(build_shmem.subprocess, "run", run)
    def record(sdk, revision, *, build_info):
        assert commands and revision == "new-mainline" and build_info["source_patches"] == []
        assert not (sdk / "deepep-sdk.json").exists()
        records.append((sdk, revision))
        path = sdk / "deepep-sdk.json"
        path.write_text(json.dumps(dict(revision=revision, build=build_info)))
        return path
    monkeypatch.setattr(build_shmem, "record", record)
    assert build_shmem.prepare(root, rebuild=rebuild) == source / "install/shmem/deepep-sdk.json"
    assert records == [(source / "install/shmem", "new-mainline")]
    assert len(commands) == 1


def test_old_sdk_requires_explicit_rebuild_without_mutation(project, monkeypatch):
    root, source = project
    manifest = source / "install/shmem/deepep-sdk.json"
    manifest.parent.mkdir(parents=True)
    original = json.dumps(dict(revision="old-revision", build={})).encode()
    manifest.write_bytes(original)
    monkeypatch.setattr(build_shmem.subprocess, "run", lambda *a, **k: pytest.fail("unexpected build"))
    with pytest.raises(RuntimeError, match="identity changed"):
        build_shmem.prepare(root)
    assert manifest.read_bytes() == original


def test_failed_vendor_build_never_records_a_new_identity(project, monkeypatch):
    root, _ = project
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])
    monkeypatch.setattr(build_shmem.subprocess, "run", fail)
    monkeypatch.setattr(build_shmem, "record", lambda *a, **k: pytest.fail("failed build was recorded"))
    with pytest.raises(subprocess.CalledProcessError):
        build_shmem.prepare(root, rebuild=True)
