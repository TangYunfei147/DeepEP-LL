"""CPU tests for NDR capability defaults and non-destructive SDK reuse."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import build_support
import build_shmem


@pytest.mark.parametrize("environment", [{}, {"DEEPEP_LL_RDMA": "ON"}, {"DEEPEP_LL_RDMA": "1"}])
def test_ndr_always_compiles_rdma(environment):
    assert build_support.ndr_rdma_enabled(environment)


@pytest.mark.parametrize("value", ["OFF", "0", "FALSE"])
def test_old_opt_out_is_actionable(value):
    with pytest.warns(FutureWarning, match="--backend mte"):
        assert build_support.ndr_rdma_enabled({"DEEPEP_LL_RDMA": value})


def test_invalid_switch_is_not_ignored():
    with pytest.warns(FutureWarning, match="deprecated and ignored"):
        assert build_support.ndr_rdma_enabled({"DEEPEP_LL_RDMA": "typo"})


def sdk_fixture(tmp_path, monkeypatch):
    root = tmp_path / "checkout with spaces"
    source = root / "third-party/shmem"
    (source / "scripts").mkdir(parents=True)
    (source / ".git").touch()
    (source / "scripts/build.sh").touch()
    (root / "dependencies.lock.json").write_text(json.dumps({
        "shmem": {"revision": "pinned"}, "cann_version": "9.3.0", "target": "Ascend950"}))
    manifest = source / "install/shmem/deepep-sdk.json"
    manifest.parent.mkdir(parents=True)
    identity = {"revision": "pinned", "build": {
        "cann_version": "9.3.0", "cann_timestamp": "known", "target": "Ascend950",
        "source_patches": [], "rdma_backend": "HNS_1825", "hns_1825_pi_on_chip": 0}}
    manifest.write_text(json.dumps(identity))
    monkeypatch.setattr(build_shmem.platform, "system", lambda: "Linux")
    monkeypatch.setattr(build_shmem, "source_patches", lambda _: [])
    monkeypatch.setattr(build_shmem, "cann_version", lambda _: {"Version": "9.3.0", "timestamp": "known"})
    monkeypatch.setattr(build_shmem.subprocess, "check_output", lambda args, **kw:
                        str(source) if args[-1] == "--show-toplevel" else "pinned")
    monkeypatch.setattr(build_shmem.subprocess, "run", lambda *a, **kw: pytest.fail("must not rebuild"))
    return root, manifest, identity


@pytest.mark.parametrize("kwargs", [{}, {"if_needed": False, "rdma_backend": None}])
def test_plain_build_reuses_verified_sdk(tmp_path, monkeypatch, kwargs):
    root, manifest, _ = sdk_fixture(tmp_path, monkeypatch)
    verified = []
    monkeypatch.setattr(build_support, "validate_shmem_sdk", lambda *args: verified.append(args))
    assert build_shmem.prepare(root, **kwargs) == manifest
    assert len(verified) == 1


def test_identity_change_requires_explicit_rebuild(tmp_path, monkeypatch):
    root, manifest, identity = sdk_fixture(tmp_path, monkeypatch)
    identity["build"]["cann_timestamp"] = "old"
    manifest.write_text(json.dumps(identity))
    with pytest.raises(RuntimeError, match="--rebuild"):
        build_shmem.prepare(root)


def test_corrupt_sdk_is_not_reused(tmp_path, monkeypatch):
    root, _, _ = sdk_fixture(tmp_path, monkeypatch)
    def reject(*args):
        raise RuntimeError("SHMEM SDK checksum mismatch")
    monkeypatch.setattr(build_support, "validate_shmem_sdk", reject)
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        build_shmem.prepare(root)


def test_unknown_build_option_is_not_reused(tmp_path, monkeypatch):
    root, manifest, identity = sdk_fixture(tmp_path, monkeypatch)
    identity["build"]["obsolete_source_transform"] = "retired"
    manifest.write_text(json.dumps(identity))
    with pytest.raises(RuntimeError, match="--rebuild"):
        build_shmem.prepare(root)


def test_source_edits_have_no_implicit_exemption(tmp_path, monkeypatch):
    root, _, _ = sdk_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(build_shmem, "source_patches", lambda _: [{"kind": "local-worktree", "sha256": "edited"}])
    with pytest.raises(RuntimeError, match="--allow-local-changes"):
        build_shmem.prepare(root)


def test_reuse_rejects_mode_disagreement_with_retained_sdk_build(tmp_path, monkeypatch):
    root, manifest, _ = sdk_fixture(tmp_path, monkeypatch)
    flags = root / "third-party/shmem/build/src/CMakeFiles/aclshmem_device.dir/flags.make"
    flags.parent.mkdir(parents=True)
    flags.write_text("CXX_DEFINES = -DACLSHMEMI_HNS_1825_PI_ON_CHIP=1\n")
    with pytest.raises(RuntimeError, match="doorbell mode"):
        build_shmem.prepare(root)


def test_reuse_checks_matching_device_flags_without_firmware_reprobe(tmp_path, monkeypatch):
    root, manifest, _ = sdk_fixture(tmp_path, monkeypatch)
    flags = root / "third-party/shmem/build/src/CMakeFiles/aclshmem_device.dir/flags.make"
    flags.parent.mkdir(parents=True)
    flags.write_text("CXX_DEFINES = -DACLSHMEMI_HNS_1825_PI_ON_CHIP=0\n")
    monkeypatch.setattr(build_support, "validate_shmem_sdk", lambda *args: None)
    assert build_shmem.prepare(root) == manifest


@pytest.mark.parametrize("mode", [0, 1, None])
def test_new_sdk_records_actual_device_mode_only_after_successful_build(tmp_path, monkeypatch, mode):
    root, manifest, _ = sdk_fixture(tmp_path, monkeypatch)
    captured = []

    def compile_sdk(*args, **kwargs):
        assert not manifest.exists()
        if mode is not None:
            flags = root / "third-party/shmem/build/src/CMakeFiles/aclshmem_device.dir/flags.make"
            flags.parent.mkdir(parents=True)
            flags.write_text(f"CXX_DEFINES = -DACLSHMEMI_HNS_1825_PI_ON_CHIP={mode}\n")

    def record_sdk(sdk, revision, *, build_info, build_root):
        assert build_root == root / "third-party/shmem/build"
        captured.append(build_info)
        return manifest

    monkeypatch.setattr(build_shmem.subprocess, "run", compile_sdk)
    monkeypatch.setattr(build_shmem, "record", record_sdk)
    if mode is None:
        with pytest.raises(RuntimeError, match="doorbell mode"):
            build_shmem.prepare(root, rebuild=True)
        assert not captured
    else:
        assert build_shmem.prepare(root, rebuild=True) == manifest
        assert captured[0]["hns_1825_pi_on_chip"] == mode


def test_failed_explicit_rebuild_cannot_reuse_old_sdk_identity(tmp_path, monkeypatch):
    root, manifest, _ = sdk_fixture(tmp_path, monkeypatch)

    def fail_build(*args, **kwargs):
        raise RuntimeError("deliberate compile failure")

    monkeypatch.setattr(build_shmem.subprocess, "run", fail_build)
    with pytest.raises(RuntimeError, match="compile failure"):
        build_shmem.prepare(root, rebuild=True)
    assert not manifest.exists()
