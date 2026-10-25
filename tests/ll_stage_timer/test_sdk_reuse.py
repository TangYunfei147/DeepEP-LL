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


@pytest.mark.parametrize("mode", [0, 1])
def test_known_hns_doorbell_mode_reused_without_changing_identity(mode):
    sdk = identity()
    sdk["build"]["hns_1825_pi_on_chip"] = mode
    original = copy.deepcopy(sdk)
    assert mte_compatible(sdk, "locked", BASE)
    assert sdk == original


@pytest.mark.parametrize("mode", [None, False, True, -1, 2, "0", "1", 0.0, 1.0, []])
def test_invalid_hns_doorbell_mode_rejected(mode):
    sdk = identity()
    sdk["build"]["hns_1825_pi_on_chip"] = mode
    assert not mte_compatible(sdk, "locked", BASE)


def test_doorbell_mode_without_hns_backend_rejected():
    sdk = dict(revision="locked", build={**BASE, "hns_1825_pi_on_chip": 1})
    assert not mte_compatible(sdk, "locked", BASE)


@pytest.mark.parametrize("mode", [None, 0, 1])
@pytest.mark.parametrize("key,value", [
    ("cann_version", "different"), ("cann_timestamp", "different"),
    ("target", "different"), ("source_patches", [{"kind": "local-worktree", "sha256": "modified"}]),
    ("rdma_backend", "different"),
    ("unknown_option", True),
])
def test_changed_input_rejected(key, value, mode):
    sdk = identity()
    if mode is not None:
        sdk["build"]["hns_1825_pi_on_chip"] = mode
    sdk["build"][key] = value
    assert not mte_compatible(sdk, "locked", BASE)


@pytest.mark.parametrize("mode", [None, 0, 1])
def test_changed_revision_rejected(mode):
    sdk = identity()
    if mode is not None:
        sdk["build"]["hns_1825_pi_on_chip"] = mode
    assert not mte_compatible(sdk, "other", BASE)


@pytest.mark.parametrize("corrupt", [False, True])
@pytest.mark.parametrize("mode", [None, 0, 1])
def test_prepare_default_reuse_runs_sdk_validation(tmp_path, monkeypatch, corrupt, mode):
    import build_shmem
    source = tmp_path / "third-party/shmem"
    (source / "scripts").mkdir(parents=True)
    (source / ".git").mkdir()
    (source / "scripts/build.sh").write_text("unused")
    manifest = source / "install/shmem/deepep-sdk.json"
    manifest.parent.mkdir(parents=True)
    sdk = identity()
    if mode is not None:
        sdk["build"]["hns_1825_pi_on_chip"] = mode
    manifest.write_text(json.dumps(sdk))
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


@pytest.mark.parametrize("mode", [None, 0, 1])
def test_hns_source_edits_have_no_implicit_exemption(mode):
    edited = {**BASE, "source_patches": [{"kind": "local-worktree", "sha256": "edited"}]}
    sdk = dict(revision="locked", build={**edited, "rdma_backend": "HNS_1825"})
    if mode is not None:
        sdk["build"]["hns_1825_pi_on_chip"] = mode
    assert not mte_compatible(sdk, "locked", edited)
