"""MTE consumer compatibility for a recorded SDK (never changes vendor sources)."""

try:
    from scripts.shmem_sdk_identity import validate_build_identity
except ModuleNotFoundError:
    from shmem_sdk_identity import validate_build_identity


def mte_compatible(identity, revision, expected_build):
    """Accept native HNS capabilities only with exactly matching build inputs."""
    if not isinstance(identity, dict) or identity.get("revision") != revision:
        return False
    build = identity.get("build", {})
    try:
        validate_build_identity(build)
    except RuntimeError:
        return False
    if build == expected_build:
        return True
    if expected_build.get("source_patches") or build.get("rdma_backend") != "HNS_1825":
        return False
    # The MTE consumer does not use the HNS doorbell. Preserve optional
    # attested mode metadata, while comparing every shared build input.
    shared_build = {key: value for key, value in build.items()
                    if key not in {"rdma_backend", "hns_1825_pi_on_chip"}}
    return shared_build == expected_build
