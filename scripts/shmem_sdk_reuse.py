"""MTE consumer compatibility for a recorded SDK (never changes vendor sources)."""
from shmem_rdma_patch import PATCH_ID, REPLACEMENTS


def mte_compatible(identity, revision, expected_build, *, managed_hns=False):
    """Allow only the known HNS capability superset, with identical build inputs."""
    if identity.get("revision") != revision:
        return False
    build = identity.get("build", {})
    if build == expected_build:
        return True
    return managed_hns and build == {
        **expected_build, "rdma_backend": "HNS_1825", "rdma_patch": PATCH_ID,
    }


def is_managed_hns_source(source):
    """Require all three known deltas and no other tracked/untracked edits."""
    from shmem_rdma_patch import prepare
    import subprocess

    try:
        if not prepare(source, apply=False):
            return False
        changed = set(subprocess.check_output(
            ["git", "-C", str(source), "diff", "HEAD", "--name-only"], text=True
        ).splitlines())
        return changed == set(REPLACEMENTS)
    except (RuntimeError, OSError, subprocess.CalledProcessError):
        return False
