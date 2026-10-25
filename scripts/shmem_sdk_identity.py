"""Build identity admission for locked SHMEM SDKs with explicit capability and source identity."""
def validate_build_identity(build, require_rdma=False):
    required = {"cann_version", "cann_timestamp", "target", "source_patches"}
    if (not isinstance(build, dict) or not required.issubset(build)
            or set(build) - required - {"rdma_backend", "hns_1825_pi_on_chip"}
            or not isinstance(build.get("source_patches"), list)
            or build.get("target") != "Ascend950"
            or not all(isinstance(build.get(k), str) and build[k] for k in required - {"source_patches"})
            or ("rdma_backend" in build and build["rdma_backend"] != "HNS_1825")
            or (require_rdma and build.get("rdma_backend") != "HNS_1825")
            or ("hns_1825_pi_on_chip" in build and
                (build.get("rdma_backend") != "HNS_1825"
                 or type(build["hns_1825_pi_on_chip"]) is not int
                 or build["hns_1825_pi_on_chip"] not in (0, 1)))):
        raise RuntimeError("Require a current native SHMEM SDK build identity; stop SDK users and rebuild explicitly")
