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
    if expected_build.get("source_patches"):
        return False
    hns = {**expected_build, "rdma_backend": "HNS_1825"}
    if build == hns:
        return True
    # MTE does not issue HNS doorbells. Its consumer may reuse either attested
    # HNS mode without changing the SDK or enabling RDMA in the consumer.
    mode = build.get("hns_1825_pi_on_chip")
    return type(mode) is int and mode in (0, 1) and build == {
        **hns, "hns_1825_pi_on_chip": mode,
    }
