"""CPU admission tests for SDK identity, independent of CANN."""
from copy import deepcopy
import pytest
from scripts.shmem_sdk_identity import validate_build_identity
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

@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("require_rdma", [False, True])
def test_known_hns_doorbell_mode_is_valid(mode, require_rdma):
    identity = {**BASE, "rdma_backend": "HNS_1825", "hns_1825_pi_on_chip": mode}
    validate_build_identity(identity, require_rdma=require_rdma)

@pytest.mark.parametrize("mode", [None, False, True, -1, 2, "0", "1", 0.0, 1.0, []])
def test_hns_doorbell_mode_requires_exact_integer(mode):
    identity = {**BASE, "rdma_backend": "HNS_1825", "hns_1825_pi_on_chip": mode}
    with pytest.raises(RuntimeError, match="identity"):
        validate_build_identity(identity)

@pytest.mark.parametrize("mode", [0, 1])
def test_hns_doorbell_mode_requires_attested_hns_backend(mode):
    with pytest.raises(RuntimeError, match="identity"):
        validate_build_identity({**BASE, "hns_1825_pi_on_chip": mode})

def test_record_without_build_attestation_does_not_write(tmp_path):
    with pytest.raises(RuntimeError, match="identity"):
        record(tmp_path, "unverified")
    assert not (tmp_path / "deepep-sdk.json").exists()
