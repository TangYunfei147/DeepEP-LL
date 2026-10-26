"""Locked SDK admission; optional vendor metadata is retained without interpretation."""
import json


def validate_build_identity(build, require_rdma=False):
    required = {"cann_version", "cann_timestamp", "target", "source_patches"}
    if (not isinstance(build, dict) or not required.issubset(build)
            or set(build) - required - {"rdma_backend", "hns_1825_pi_on_chip"}
            or not isinstance(build.get("source_patches"), list)
            or build.get("target") != "Ascend950"
            or not all(isinstance(build.get(k), str) and build[k] for k in required - {"source_patches"})
            or ("rdma_backend" in build and build["rdma_backend"] != "HNS_1825")
            or (require_rdma and build.get("rdma_backend") != "HNS_1825")):
        raise RuntimeError("Require a current native SHMEM SDK build identity; stop SDK users and rebuild explicitly")
    # The SDK producer owns this optional field's type and value domain. Accept
    # every JSON value (including null/containers), with or without a backend
    # tag. Do not coerce it to bool, infer RDMA support, or remove it from the
    # recorded identity. The wheel manifest/cache retain the complete build.
    if "hns_1825_pi_on_chip" in build:
        try:
            json.dumps(build["hns_1825_pi_on_chip"], allow_nan=False)
        except (TypeError, ValueError, OverflowError, RecursionError) as error:
            raise RuntimeError("SHMEM build identity metadata must be a JSON value") from error
