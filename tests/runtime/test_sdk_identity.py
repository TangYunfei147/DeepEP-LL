"""CPU admission tests for SDK identity, independent of CANN."""
from copy import deepcopy
import pytest
from scripts.shmem_sdk_identity import validate_build_identity
from scripts.record_shmem_sdk import record
from scripts.shmem_barrier_patch import patch_identity

BASE = {"cann_version": "9.3.0", "cann_timestamp": "test", "target": "Ascend950", "source_patches": []}

@pytest.mark.parametrize("rdma", [False, True])
@pytest.mark.parametrize("modified", [False, True])
def test_known_capability_and_explicit_source_changes(rdma, modified):
    identity = deepcopy(BASE)
    if rdma:
        identity["rdma_backend"] = "HNS_1825"
    if modified:
        identity["source_patches"] = [{"kind": "local-worktree", "sha256": "a" * 64}]
    if rdma:
        identity["source_patches"].append(patch_identity())
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


def test_old_or_changed_barrier_sdk_is_rejected():
    identity = dict(BASE, rdma_backend="HNS_1825")
    with pytest.raises(RuntimeError, match="barrier fix"):
        validate_build_identity(identity)
    identity["source_patches"] = [dict(patch_identity(), sha256="0" * 64)]
    with pytest.raises(RuntimeError, match="barrier fix"):
        validate_build_identity(identity)

def test_record_without_build_attestation_does_not_write(tmp_path):
    with pytest.raises(RuntimeError, match="identity"):
        record(tmp_path, "unverified")
    assert not (tmp_path / "deepep-sdk.json").exists()
