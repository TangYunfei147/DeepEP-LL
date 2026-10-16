"""MTE reuse of an HNS SDK must not relax source or build identity checks."""
import copy
from pathlib import Path
import subprocess
import sys
import json
import types

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from shmem_rdma_patch import PATCH_ID, REPLACEMENTS, prepare as patch
from shmem_sdk_reuse import is_managed_hns_source, mte_compatible


BASE = dict(cann_version="9.3.0", cann_timestamp="test", target="Ascend950",
            source_patches=[{"kind": "local-worktree", "sha256": "known"}])


def identity():
    return dict(revision="locked", build={**copy.deepcopy(BASE),
                "rdma_backend": "HNS_1825", "rdma_patch": PATCH_ID})


def test_known_superset_and_plain_mte():
    assert mte_compatible(identity(), "locked", BASE, managed_hns=True)
    assert mte_compatible(dict(revision="locked", build=BASE), "locked", BASE)
    assert not mte_compatible(identity(), "locked", BASE)


@pytest.mark.parametrize("key,value", [
    ("cann_version", "different"), ("cann_timestamp", "different"),
    ("target", "different"), ("source_patches", []),
    ("rdma_backend", "different"), ("rdma_patch", "different"),
    ("unknown_option", True),
])
def test_changed_input_rejected(key, value):
    sdk = identity()
    sdk["build"][key] = value
    assert not mte_compatible(sdk, "locked", BASE, managed_hns=True)


def test_changed_revision_rejected():
    assert not mte_compatible(identity(), "other", BASE, managed_hns=True)


def test_only_complete_known_delta_accepted(tmp_path):
    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    git("init")
    for relative, (old, _) in REPLACEMENTS.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(old)
    git("add", ".")
    git("-c", "user.name=test", "-c", "user.email=test@example.com", "commit", "-m", "baseline")
    assert not is_managed_hns_source(tmp_path)
    patch(tmp_path, apply=True)
    assert is_managed_hns_source(tmp_path)
    extra = tmp_path / "extra.cpp"
    extra.write_text("unmanaged")
    assert not is_managed_hns_source(tmp_path)
    extra.unlink()
    relative, (old, _) = next(iter(REPLACEMENTS.items()))
    (tmp_path / relative).write_text(old)
    assert not is_managed_hns_source(tmp_path)
    (tmp_path / relative).write_text("unrelated edit")
    assert not is_managed_hns_source(tmp_path)


@pytest.mark.parametrize("corrupt", [False, True])
def test_prepare_default_reuse_runs_sdk_validation(tmp_path, monkeypatch, corrupt):
    import build_shmem
    source = tmp_path / "third-party/shmem"
    (source / "scripts").mkdir(parents=True)
    (source / ".git").mkdir()
    (source / "scripts/build.sh").write_text("unused")
    manifest = source / "install/shmem/deepep-sdk.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(identity()))
    (tmp_path / "dependencies.lock.json").write_text(json.dumps(dict(
        shmem=dict(revision="locked"), cann_version="9.3.0", target="Ascend950")))
    monkeypatch.setattr(build_shmem.platform, "system", lambda: "Linux")
    monkeypatch.setattr(build_shmem, "source_patches", lambda _: BASE["source_patches"])
    monkeypatch.setattr(build_shmem, "is_managed_hns_source", lambda _: True)
    monkeypatch.setattr(build_shmem, "cann_version", lambda _: dict(Version="9.3.0", timestamp="test"))
    monkeypatch.setattr(build_shmem.subprocess, "check_output", lambda args, **kw:
                        str(source) if args[-1] == "--show-toplevel" else "locked")
    validated = []
    def validate(sdk, expected):
        validated.append((sdk, expected))
        if corrupt:
            raise RuntimeError("SHMEM SDK checksum mismatch")
    monkeypatch.setitem(sys.modules, "build_support", types.SimpleNamespace(validate_shmem_sdk=validate))
    if corrupt:
        with pytest.raises(RuntimeError, match="checksum mismatch"):
            build_shmem.prepare(tmp_path)
    else:
        assert build_shmem.prepare(tmp_path) == manifest
    assert validated == [(manifest.parent, dict(revision="locked"))]
