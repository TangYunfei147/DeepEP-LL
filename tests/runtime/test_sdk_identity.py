"""CPU admission tests for SDK identity, independent of CANN."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import pytest
from scripts.shmem_sdk_identity import sdk_artifacts, validate_build_identity, validate_sdk_hashes
from scripts.record_shmem_sdk import record

BASE = {"cann_version": "9.3.0", "cann_timestamp": "test", "target": "Ascend950", "source_patches": []}

@pytest.mark.parametrize("rdma", [False, True])
@pytest.mark.parametrize("modified", [False, True])
def test_known_capability_and_explicit_source_changes(rdma, modified):
    identity = deepcopy(BASE)
    if rdma:
        identity["rdma_backend"] = "HNS_1825"
    if modified:
        identity["source_patches"] = [{"kind": "local-worktree", "sha256": "a" * 64}]
    validate_build_identity(identity, require_rdma=rdma)

@pytest.mark.parametrize("change", ["extra", "missing", "backend", "patches", "empty", "type"])
def test_reject_unknown_or_incomplete_identity(change):
    identity = deepcopy(BASE)
    if change == "extra": identity["obsolete_extension"] = "value"
    if change == "missing": del identity["cann_timestamp"]
    if change == "backend": identity["rdma_backend"] = "unknown"
    if change == "patches": identity["source_patches"] = "unknown"
    if change == "empty": identity["cann_version"] = ""
    if change == "type": identity = []
    with pytest.raises(RuntimeError, match="identity"):
        validate_build_identity(identity)

def test_rdma_requires_attested_backend():
    with pytest.raises(RuntimeError, match="identity"):
        validate_build_identity(BASE, require_rdma=True)

def test_record_without_build_attestation_does_not_write(tmp_path):
    with pytest.raises(RuntimeError, match="identity"):
        record(tmp_path, "unverified")
    assert not (tmp_path / "deepep-sdk.json").exists()


@pytest.fixture
def sdk_files(tmp_path):
    sdk = tmp_path / "SDK with spaces"
    contents = {"include/host/shmem_host_def.h": b"host declarations\n",
                "include/device/example.hpp": b"inline API\n",
                "src/device/gm2gm/engine/example.hpp": b"device implementation\n",
                "lib/libshmem.so": b"library", "lib/libshmem_utils.so": b"support library",
                "lib/aclshmem_bootstrap_config_store.so": b"bootstrap library"}
    for name, data in contents.items():
        path = sdk / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}
    return sdk, hashes


def test_record_covers_public_and_device_headers_and_all_libraries(tmp_path, monkeypatch, sdk_files):
    import scripts.record_shmem_sdk as recorder
    sdk, hashes = sdk_files
    source = tmp_path / "project"
    source.mkdir()
    monkeypatch.setattr(recorder, "__file__", str(source / "scripts/record_shmem_sdk.py"))
    pinned = dict(revision="new-sdk", header_sha256_lf={
        "host/shmem_host_def.h": hashes["include/host/shmem_host_def.h"]})
    (source / "dependencies.lock.json").write_text(json.dumps({"shmem": pinned}), encoding="utf-8")
    manifest = recorder.record(sdk, "new-sdk", build_info=BASE)
    assert json.loads(manifest.read_text())["sha256"] == hashes
    assert recorder.check(sdk) == manifest
    with pytest.raises(FileExistsError):
        recorder.record(sdk, "new-sdk", build_info=BASE)


@pytest.mark.parametrize("artifact", ["include/device/example.hpp", "src/device/gm2gm/engine/example.hpp",
                                      "lib/libshmem_utils.so", "lib/aclshmem_bootstrap_config_store.so"])
@pytest.mark.parametrize("fault", ["omitted", "modified"])
def test_device_implementation_and_auxiliary_library_identity_cannot_be_omitted(sdk_files, artifact, fault):
    sdk, hashes = sdk_files
    if fault == "omitted":
        del hashes[artifact]
    else:
        (sdk / artifact).write_bytes(b"unrecorded replacement")
    with pytest.raises(RuntimeError, match="cover|checksum"):
        validate_sdk_hashes(sdk, hashes, {"host/shmem_host_def.h": "locked"})


def test_new_sdk_inventory_invalidates_incomplete_existing_manifest(sdk_files):
    sdk, hashes = sdk_files
    validate_sdk_hashes(sdk, hashes, {"host/shmem_host_def.h": "locked"})
    added = sdk / "src/device/new_api.hpp"
    added.write_bytes(b"new build input")
    with pytest.raises(RuntimeError, match="must cover"):
        validate_sdk_hashes(sdk, hashes, {"host/shmem_host_def.h": "locked"})
    assert added in sdk_artifacts(sdk)


def test_locked_public_headers_match_initialized_official_checkout():
    root = Path(__file__).resolve().parents[2]
    source = root / "third-party/shmem"
    if not (source / ".git").exists():
        pytest.skip("Initialize pinned SHMEM to verify the source header fingerprints")
    lock = json.loads((root / "dependencies.lock.json").read_text(encoding="utf-8"))["shmem"]
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    assert revision == lock["revision"]
    for relative, expected in lock["header_sha256_lf"].items():
        data = (source / "include" / relative).read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(data).hexdigest() == expected, relative


def test_previous_4dd_sdk_is_not_admitted_by_upgraded_source(sdk_files):
    from build_support import validate_shmem_sdk
    sdk, hashes = sdk_files
    old = dict(revision="4dd80e2129da42bd773982c2bb3758af1ed0bba8", build=BASE, sha256=hashes)
    (sdk / "deepep-sdk.json").write_text(json.dumps(old), encoding="utf-8")
    expected = dict(revision="8e3fc610f91ae9e5e8cf13748cfb74826b272c2a", header_sha256_lf={
        "host/shmem_host_def.h": hashes["include/host/shmem_host_def.h"]})
    with pytest.raises(RuntimeError, match="revision differs"):
        validate_shmem_sdk(sdk, expected)
