"""MTE consumer compatibility for a recorded SDK (never changes vendor sources)."""


def mte_compatible(identity, revision, expected_build):
    """Accept native HNS capabilities only with exactly matching build inputs."""
    if identity.get("revision") != revision:
        return False
    build = identity.get("build", {})
    if build == expected_build:
        return True
    return not expected_build.get("source_patches") and build == {
        **expected_build, "rdma_backend": "HNS_1825",
    }
