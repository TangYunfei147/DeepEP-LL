"""CPU admission tests for SDK identity, independent of CANN."""
from copy import deepcopy
import json
import pytest
from scripts.shmem_sdk_identity import validate_build_identity
from scripts.record_shmem_sdk import record

BASE = {"cann_version": "9.3.0", "cann_timestamp": "test", "target": "Ascend950", "source_patches": []}
OPAQUE_VALUES = [None, False, True, -1, 0, 1, 2, "0", "1", "host", "", 0.0, 1.0, -2.5,
                 [], [1, False, None, "mode"], {}, {"mode": [2, "vendor"], "enabled": None}]

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

@pytest.mark.parametrize("mode", OPAQUE_VALUES)
@pytest.mark.parametrize("require_rdma", [False, True])
def test_hns_doorbell_metadata_accepts_every_json_value_without_coercion(mode, require_rdma):
    identity = {**BASE, "rdma_backend": "HNS_1825", "hns_1825_pi_on_chip": mode}
    original = json.dumps(identity, sort_keys=True)
    validate_build_identity(identity, require_rdma=require_rdma)
    assert json.dumps(identity, sort_keys=True) == original
    assert type(identity["hns_1825_pi_on_chip"]) is type(mode)


@pytest.mark.parametrize("mode", OPAQUE_VALUES)
def test_mte_metadata_does_not_claim_an_rdma_backend(mode):
    identity = {**BASE, "hns_1825_pi_on_chip": mode}
    original = json.dumps(identity, sort_keys=True)
    validate_build_identity(identity)
    assert json.dumps(identity, sort_keys=True) == original
    with pytest.raises(RuntimeError, match="identity"):
        validate_build_identity(identity, require_rdma=True)


@pytest.mark.parametrize("mode", [None, False, 2, "vendor", {"mode": [2, None]}])
@pytest.mark.parametrize("backend", [None, False, "unknown", ""])
def test_opaque_metadata_never_relaxes_an_explicit_backend(mode, backend):
    for require_rdma in (False, True):
        with pytest.raises(RuntimeError, match="identity"):
            validate_build_identity({**BASE, "rdma_backend": backend, "hns_1825_pi_on_chip": mode},
                                    require_rdma=require_rdma)


@pytest.mark.parametrize("mode", [None, [], {"vendor": "future"}])
def test_only_the_named_opaque_field_is_exempt_from_unknown_field_rejection(mode):
    with pytest.raises(RuntimeError, match="identity"):
        validate_build_identity({**BASE, "hns_1825_pi_on_chip": mode, "other_vendor_option": mode})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"),
                                   {"nested": [float("nan")]}, {1, 2}, object()])
def test_opaque_metadata_must_still_be_json(value):
    with pytest.raises(RuntimeError, match="identity"):
        validate_build_identity({**BASE, "hns_1825_pi_on_chip": value})

def test_record_without_build_attestation_does_not_write(tmp_path):
    with pytest.raises(RuntimeError, match="identity"):
        record(tmp_path, "unverified")
    assert not (tmp_path / "deepep-sdk.json").exists()
