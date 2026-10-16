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
        "source_patches": [], "rdma_backend": "HNS_1825"}}
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
