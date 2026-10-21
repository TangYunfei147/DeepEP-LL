"""Exact, versioned Host-barrier overlay for the pinned NDR SDK."""
import hashlib
from pathlib import Path
import subprocess

REVISION = "4dd80e2129da42bd773982c2bb3758af1ed0bba8"
TARGET = "src/device/gm2gm/shmemi_device_cc_kernel.cpp"
PAYLOAD = Path(__file__).resolve().parent / "patches/ndr_host_barrier.inc"
KIND = "ndr-host-barrier-v1"


def patch_identity():
    return {"kind": KIND, "sha256": hashlib.sha256(PAYLOAD.read_text(encoding="utf-8").encode()).hexdigest()}


def patched_source(original):
    anchor = "// kernels\nACLSHMEM_GLOBAL void barrier_on_stream_kernel"
    call = "    aclshmemi_sync<false>(team);"
    if original.count(anchor) != 1 or original.count(call) != 1:
        raise RuntimeError("Pinned SHMEM Host barrier layout differs; refusing patch")
    return original.replace(anchor, PAYLOAD.read_text(encoding="utf-8") + "\n" + anchor).replace(
        call, "    deepep_host_barrier(team);")


def expected_source(source):
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise RuntimeError("NDR barrier patch requires the locked SHMEM revision")
    original = subprocess.check_output(["git", "-C", str(source), "show", f"HEAD:{TARGET}"]).decode("utf-8")
    return original, patched_source(original)


def apply(source):
    source = Path(source)
    original, expected = expected_source(source)
    target = source / TARGET
    if target.is_symlink():
        raise RuntimeError("Refusing SHMEM barrier patch through a symlink")
    current = target.read_text(encoding="utf-8")
    if current == expected:
        return
    if current != original:
        raise RuntimeError("SHMEM barrier has unknown edits; preserve and reconcile them before building")
    target.write_text(expected, encoding="utf-8", newline="\n")


def is_managed_only(source):
    source = Path(source)
    _, expected = expected_source(source)
    if (source / TARGET).read_text(encoding="utf-8") != expected:
        return False
    changed = subprocess.check_output(
        ["git", "-C", str(source), "diff", "HEAD", "--name-only"], text=True).splitlines()
    untracked = subprocess.check_output(["git", "-C", str(source), "ls-files", "--others", "--exclude-standard"])
    return changed == [TARGET] and not untracked.strip()
