"""SDK recording covers installed implementation headers and dependent libraries."""
import hashlib
import json
from pathlib import Path

import pytest

from scripts import record_shmem_sdk as recorder


BUILD = dict(cann_version="9.3.0", cann_timestamp="test", target="Ascend950", source_patches=[])


@pytest.fixture
def installed(tmp_path, monkeypatch):
    project, sdk = tmp_path / "project", tmp_path / "sdk"
    (project / "scripts").mkdir(parents=True)
    monkeypatch.setattr(recorder, "__file__", str(project / "scripts/record_shmem_sdk.py"))
    files = {
        "include/host/shmem_host_def.h": b"host\n",
        "include/shmem.h": b"public\n",
        "include/extra.hpp": b"extra interface\n",
        "src/device/gm2gm/shmemi_device_cc.h": b"collective implementation\n",
        "src/device/gm2gm/engine/shmem_device_udma.hpp": b"udma implementation\n",
        "lib/libshmem.so": b"main library",
        "lib/libshmem_utils.so": b"helper library",
        "lib/bootstrap/plugin.so": b"bootstrap plugin",
    }
    for relative, content in files.items():
        path = sdk / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    lock = dict(shmem=dict(revision="locked", header_sha256_lf={
        "host/shmem_host_def.h": hashlib.sha256(b"host\n").hexdigest()}))
    (project / "dependencies.lock.json").write_text(json.dumps(lock), encoding="utf-8")
    return sdk, files


def test_record_all_inputs_and_refuse_overwrite(installed):
    sdk, files = installed
    path = recorder.record(sdk, "locked", build_info=BUILD)
    recorded = json.loads(path.read_text())
    assert recorded["sha256"] == {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    assert recorder.check(sdk) == path
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        recorder.record(sdk, "locked", build_info=BUILD)
    assert path.read_bytes() == original


@pytest.mark.parametrize("relative", ["include/extra.hpp", "src/device/gm2gm/engine/shmem_device_udma.hpp",
                                      "lib/libshmem_utils.so", "lib/bootstrap/plugin.so"])
def test_implementation_and_plugin_changes_are_rejected(installed, relative):
    sdk, _ = installed
    recorder.record(sdk, "locked", build_info=BUILD)
    (sdk / relative).write_bytes(b"changed")
    with pytest.raises(ValueError, match="SDK checksum mismatch"):
        recorder.check(sdk)


@pytest.mark.parametrize("relative", ["include/shmem.h", "src/device/gm2gm/shmemi_device_cc.h", "lib/libshmem.so"])
def test_incomplete_sdk_never_receives_manifest(installed, relative):
    sdk, _ = installed
    (sdk / relative).unlink()
    with pytest.raises(ValueError, match="Incomplete SHMEM SDK"):
        recorder.record(sdk, "locked", build_info=BUILD)
    assert not (sdk / "deepep-sdk.json").exists()


def test_missing_manifest_coverage_and_new_inputs_are_rejected(installed):
    sdk, _ = installed
    path = recorder.record(sdk, "locked", build_info=BUILD)
    identity = json.loads(path.read_text())
    del identity["sha256"]["src/device/gm2gm/engine/shmem_device_udma.hpp"]
    path.write_text(json.dumps(identity), encoding="utf-8")
    with pytest.raises(ValueError, match="all installed include/src/lib"):
        recorder.check(sdk)


def test_new_installed_header_requires_new_identity(installed):
    sdk, _ = installed
    recorder.record(sdk, "locked", build_info=BUILD)
    (sdk / "include/new.hpp").write_text("new interface")
    with pytest.raises(ValueError, match="all installed include/src/lib"):
        recorder.check(sdk)


def test_locked_header_lf_normalization_is_preserved(installed):
    sdk, _ = installed
    (sdk / "include/host/shmem_host_def.h").write_bytes(b"host\r\n")
    recorder.record(sdk, "locked", build_info=BUILD)
    recorder.check(sdk)
