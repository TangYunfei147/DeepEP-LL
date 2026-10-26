"""MTE consumer compatibility for a recorded SDK (never changes vendor sources)."""

try:
    from scripts.shmem_sdk_identity import validate_build_identity
except ModuleNotFoundError:
    from shmem_sdk_identity import validate_build_identity


def mte_compatible(identity, revision, expected_build):
    """Compare MTE inputs while preserving vendor metadata in the original identity."""
    if not isinstance(identity, dict) or identity.get("revision") != revision:
        return False
    build = identity.get("build", {})
    try:
        validate_build_identity(build)
        validate_build_identity(expected_build)
    except RuntimeError:
        return False
    if build == expected_build:
        return True
    if expected_build.get("source_patches"):
        return False
    # MTE does not interpret the optional vendor field, regardless of its JSON
    # representation or presence of the separately validated HNS capability.
    # This comparison must not rewrite either manifest or expected inputs.
    shared_build = {key: value for key, value in build.items()
                    if key not in {"rdma_backend", "hns_1825_pi_on_chip"}}
    shared_expected = {key: value for key, value in expected_build.items()
                       if key not in {"rdma_backend", "hns_1825_pi_on_chip"}}
    return shared_build == shared_expected
