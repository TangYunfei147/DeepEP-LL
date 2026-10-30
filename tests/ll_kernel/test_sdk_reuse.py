"""MTE reuse of an HNS SDK must not relax source or build identity checks."""
import copy
from pathlib import Path
import sys
import json
import shutil
import subprocess
import types

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from shmem_sdk_reuse import mte_compatible
from shmem_sdk_identity import validate_build_identity


BASE = dict(cann_version="9.3.0", cann_timestamp="test", target="Ascend950",
            source_patches=[])


def identity(pi_mode=None):
    build = {**copy.deepcopy(BASE), "rdma_backend": "HNS_1825"}
    if pi_mode is not None:
        build["hns_1825_pi_on_chip"] = pi_mode
    return dict(revision="locked", build=build)


@pytest.mark.parametrize("pi_mode", [None, 0, 1])
def test_known_superset_and_plain_mte(pi_mode):
    sdk = identity(pi_mode)
    assert mte_compatible(sdk, "locked", BASE)
    validate_build_identity(sdk["build"])
    assert mte_compatible(dict(revision="locked", build=BASE), "locked", BASE)


@pytest.mark.parametrize("key,value", [
    ("cann_version", "different"), ("cann_timestamp", "different"),
    ("target", "different"), ("source_patches", [{"kind": "local-worktree", "sha256": "modified"}]),
    ("rdma_backend", "different"),
    ("unknown_option", True),
])
@pytest.mark.parametrize("pi_mode", [None, 0, 1])
def test_changed_input_rejected(key, value, pi_mode):
    sdk = identity(pi_mode)
    sdk["build"][key] = value
    assert not mte_compatible(sdk, "locked", BASE)


def test_changed_revision_rejected():
    assert not mte_compatible(identity(), "other", BASE)


@pytest.mark.parametrize("corrupt", [False, True])
@pytest.mark.parametrize("pi_mode", [None, 0, 1])
def test_prepare_default_reuse_runs_sdk_validation(tmp_path, monkeypatch, corrupt, pi_mode):
    import build_shmem
    source = tmp_path / "third-party/shmem"
    (source / "scripts").mkdir(parents=True)
    (source / ".git").mkdir()
    (source / "scripts/build.sh").write_text("unused")
    manifest = source / "install/shmem/deepep-sdk.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(identity(pi_mode)))
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


@pytest.mark.parametrize("pi_mode", [None, 0, 1])
def test_hns_source_edits_have_no_implicit_exemption(pi_mode):
    edited = {**BASE, "source_patches": [{"kind": "local-worktree", "sha256": "edited"}]}
    sdk = dict(revision="locked", build={**edited, "rdma_backend": "HNS_1825"})
    if pi_mode is not None:
        sdk["build"]["hns_1825_pi_on_chip"] = pi_mode
    assert not mte_compatible(sdk, "locked", edited)


@pytest.mark.parametrize("mode", [True, False, "0", "1", 0.0, 1.0, -1, 2, None, [], {}])
def test_pi_mode_requires_literal_integer_zero_or_one(mode):
    sdk = identity()
    sdk["build"]["hns_1825_pi_on_chip"] = mode
    assert not mte_compatible(sdk, "locked", BASE)
    with pytest.raises(RuntimeError, match="identity"):
        validate_build_identity(sdk["build"])


def test_pi_mode_requires_hns_backend():
    build = {**BASE, "hns_1825_pi_on_chip": 0}
    assert not mte_compatible(dict(revision="locked", build=build), "locked", BASE)
    with pytest.raises(RuntimeError, match="identity"):
        validate_build_identity(build)


@pytest.mark.parametrize("build,accepted", [
    (BASE, True),
    (identity()["build"], True),
    (identity(0)["build"], True),
    (identity(1)["build"], True),
    ({**identity(0)["build"], "unknown_option": 0}, False),
    ({**identity()["build"], "unknown_option": 0}, False),
    ({**identity(0)["build"], "rdma_backend": "unknown"}, False),
    ({**BASE, "hns_1825_pi_on_chip": 0}, False),
    *[({**identity()["build"], "hns_1825_pi_on_chip": value}, False)
      for value in (True, False, "0", "1", 0.0, 1.0, -1, 2, None)],
])
def test_real_cmake_mte_consumer_identity_admission(tmp_path, build, accepted):
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is required for the real CPU configure admission test")
    source, sdk, cann = (tmp_path / name for name in ("project with spaces", "SDK with spaces", "CANN with spaces"))
    source.mkdir()
    (sdk / "lib").mkdir(parents=True)
    (cann / "lib64").mkdir(parents=True)
    for path in (sdk / "lib/libshmem.so", cann / "lib64/libascendcl.so", cann / "lib64/libruntime.so"):
        path.write_bytes(b"configure-only placeholder")
    (sdk / "deepep-sdk.json").write_text(json.dumps(dict(revision="locked", build=build)), encoding="utf-8")
    dependencies = Path(__file__).resolve().parents[2] / "cmake/Dependencies.cmake"
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.20)\nproject(sdk_mte_consumer NONE)\n"
        'set(CMAKE_FIND_LIBRARY_PREFIXES "lib")\nset(CMAKE_FIND_LIBRARY_SUFFIXES ".so")\n'
        f'set(ASCEND_HOME_PATH [=[{cann.as_posix()}]=])\n'
        f'set(SHMEM_ROOT [=[{sdk.as_posix()}]=])\n'
        f'include([=[{dependencies.as_posix()}]=])\n'
        'get_directory_property(_consumer_definitions COMPILE_DEFINITIONS)\n'
        'if(_consumer_definitions MATCHES "ACLSHMEM|HNS_1825|PI_ON_CHIP")\n'
        '  message(FATAL_ERROR "MTE consumer unexpectedly enables RDMA compilation")\nendif()\n'
        'get_target_property(_sdk_definitions shmem INTERFACE_COMPILE_DEFINITIONS)\n'
        'if(_sdk_definitions)\n  message(FATAL_ERROR "MTE import adds SDK definitions")\nendif()\n',
        encoding="utf-8")
    result = subprocess.run([cmake, "-S", str(source), "-B", str(tmp_path / "configure")],
                            capture_output=True, text=True)
    if accepted:
        assert result.returncode == 0, result.stdout + result.stderr
    else:
        assert result.returncode != 0
        assert "SHMEM SDK" in result.stdout + result.stderr


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
