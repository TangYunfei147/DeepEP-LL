"""MTE reuse of an HNS SDK must not relax source or build identity checks."""
import copy
from pathlib import Path
import sys
import json
import types

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from shmem_sdk_reuse import mte_compatible


BASE = dict(cann_version="9.3.0", cann_timestamp="test", target="Ascend950",
            source_patches=[])


def identity():
    return dict(revision="locked", build={**copy.deepcopy(BASE),
                "rdma_backend": "HNS_1825"})


def test_known_superset_and_plain_mte():
    assert mte_compatible(identity(), "locked", BASE)
    assert mte_compatible(dict(revision="locked", build=BASE), "locked", BASE)


@pytest.mark.parametrize("key,value", [
    ("cann_version", "different"), ("cann_timestamp", "different"),
    ("target", "different"), ("source_patches", [{"kind": "local-worktree", "sha256": "modified"}]),
    ("rdma_backend", "different"),
    ("unknown_option", True),
])
def test_changed_input_rejected(key, value):
    sdk = identity()
    sdk["build"][key] = value
    assert not mte_compatible(sdk, "locked", BASE)


def test_changed_revision_rejected():
    assert not mte_compatible(identity(), "other", BASE)


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


def test_hns_source_edits_have_no_implicit_exemption():
    edited = {**BASE, "source_patches": [{"kind": "local-worktree", "sha256": "edited"}]}
    sdk = dict(revision="locked", build={**edited, "rdma_backend": "HNS_1825"})
    assert not mte_compatible(sdk, "locked", edited)


def test_explicit_rebuild_preserves_old_outputs_outside_source(tmp_path):
    import build_shmem
    source = tmp_path / "third-party/shmem"
    (source / "build").mkdir(parents=True)
    (source / "install/shmem").mkdir(parents=True)
    (source / "build/object.o").write_bytes(b"previous object")
    (source / "install/shmem/deepep-sdk.json").write_bytes(b"previous identity")
    archive = build_shmem.archive_rebuild_outputs(tmp_path, source)
    assert archive.is_relative_to(tmp_path / ".cache/shmem-rebuild")
    assert not archive.is_relative_to(source)
    assert (archive / "build/object.o").read_bytes() == b"previous object"
    assert (archive / "install/shmem/deepep-sdk.json").read_bytes() == b"previous identity"
    assert not (source / "build").exists() and not (source / "install").exists()
    assert build_shmem.archive_rebuild_outputs(tmp_path, source) is None


def test_rebuild_archive_refuses_source_outside_project(tmp_path):
    import build_shmem
    root, outside = tmp_path / "project", tmp_path / "outside"
    root.mkdir()
    (outside / "install").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="inside the project"):
        build_shmem.archive_rebuild_outputs(root, outside)
    assert (outside / "install").exists()
