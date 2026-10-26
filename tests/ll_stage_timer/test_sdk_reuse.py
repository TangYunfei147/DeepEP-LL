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
OPAQUE_VALUES = [None, False, True, -1, 0, 1, 2, "0", "1", "vendor", "", 0.0, 1.0, -2.5,
                 [], [1, None, False], {}, {"mode": ["vendor", 2], "enabled": None}]


def identity():
    return dict(revision="locked", build={**copy.deepcopy(BASE),
                "rdma_backend": "HNS_1825"})


def test_known_superset_and_plain_mte():
    assert mte_compatible(identity(), "locked", BASE)
    assert mte_compatible(dict(revision="locked", build=BASE), "locked", BASE)


@pytest.mark.parametrize("mode", OPAQUE_VALUES)
@pytest.mark.parametrize("hns_backend", [False, True])
def test_opaque_hns_metadata_reused_without_changing_identity(mode, hns_backend):
    sdk = identity() if hns_backend else dict(revision="locked", build=copy.deepcopy(BASE))
    sdk["build"]["hns_1825_pi_on_chip"] = mode
    original = json.dumps(sdk, sort_keys=True)
    assert mte_compatible(sdk, "locked", BASE)
    assert json.dumps(sdk, sort_keys=True) == original
    assert type(sdk["build"]["hns_1825_pi_on_chip"]) is type(mode)


@pytest.mark.parametrize("mode", [float("nan"), float("inf"), float("-inf"), {"nested": [float("nan")]}, {1, 2}, object()])
def test_non_json_opaque_metadata_rejected(mode):
    sdk = identity()
    sdk["build"]["hns_1825_pi_on_chip"] = mode
    assert not mte_compatible(sdk, "locked", BASE)


@pytest.mark.parametrize("mode", [None, False, -1, 2, "vendor", [], {"future": [2, None]}])
@pytest.mark.parametrize("hns_backend", [False, True])
@pytest.mark.parametrize("key,value", [
    ("cann_version", "different"), ("cann_timestamp", "different"),
    ("target", "different"), ("source_patches", [{"kind": "local-worktree", "sha256": "modified"}]),
    ("rdma_backend", "different"),
    ("unknown_option", True),
])
def test_changed_input_rejected(key, value, mode, hns_backend):
    sdk = identity() if hns_backend else dict(revision="locked", build=copy.deepcopy(BASE))
    sdk["build"]["hns_1825_pi_on_chip"] = mode
    sdk["build"][key] = value
    assert not mte_compatible(sdk, "locked", BASE)


@pytest.mark.parametrize("mode", OPAQUE_VALUES)
def test_changed_revision_rejected(mode):
    sdk = identity()
    sdk["build"]["hns_1825_pi_on_chip"] = mode
    assert not mte_compatible(sdk, "other", BASE)


@pytest.mark.parametrize("corrupt", [False, True])
@pytest.mark.parametrize("mode", [None, True, -1, 2, "vendor", 1.5, [False, 2], {"future": None}])
@pytest.mark.parametrize("hns_backend", [False, True])
def test_prepare_default_reuse_runs_sdk_validation(tmp_path, monkeypatch, corrupt, mode, hns_backend):
    import build_shmem
    source = tmp_path / "third-party/shmem"
    (source / "scripts").mkdir(parents=True)
    (source / ".git").mkdir()
    (source / "scripts/build.sh").write_text("unused")
    manifest = source / "install/shmem/deepep-sdk.json"
    manifest.parent.mkdir(parents=True)
    sdk = identity() if hns_backend else dict(revision="locked", build=copy.deepcopy(BASE))
    sdk["build"]["hns_1825_pi_on_chip"] = mode
    manifest.write_text(json.dumps(sdk))
    original = manifest.read_bytes()
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
    assert manifest.read_bytes() == original


@pytest.mark.parametrize("mode", [None, 0, 1, "vendor", {"future": [2]}])
def test_hns_source_edits_have_no_implicit_exemption(mode):
    edited = {**BASE, "source_patches": [{"kind": "local-worktree", "sha256": "edited"}]}
    sdk = dict(revision="locked", build={**edited, "rdma_backend": "HNS_1825"})
    sdk["build"]["hns_1825_pi_on_chip"] = mode
    assert not mte_compatible(sdk, "locked", edited)
