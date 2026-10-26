"""Build identity admission for locked SHMEM SDKs with explicit capability and source identity."""
import hashlib
from pathlib import Path


def sdk_artifacts(root):
    """All installed compile headers and shared libraries, including device .hpp."""
    root = Path(root).resolve()
    files = {path for directory in (root / "include", root / "src")
             for path in directory.rglob("*") if path.is_file() and path.suffix in {".h", ".hpp"}}
    files.update(path for path in (root / "lib").rglob("*")
                 if path.is_file() and (path.name.endswith(".so") or ".so." in path.name))
    files.add(root / "lib/libshmem.so")
    for path in files:
        if not path.resolve().is_relative_to(root) or not path.is_file():
            raise RuntimeError(f"Invalid or missing SHMEM SDK artifact: {path}")
    return sorted(files)


def validate_sdk_hashes(root, hashes, locked_headers):
    """Do not admit a manifest that omits any currently installed build input."""
    root = Path(root).resolve()
    if not isinstance(hashes, dict):
        raise RuntimeError("Invalid SHMEM SDK manifest checksums")
    required = {path.relative_to(root).as_posix() for path in sdk_artifacts(root)}
    required.update("include/" + name for name in locked_headers)
    if not required.issubset(hashes):
        missing = sorted(required - set(hashes))
        raise RuntimeError("SHMEM SDK manifest must cover all installed headers/device implementations/libraries: "
                           + ", ".join(missing[:8]))
    for relative, digest in hashes.items():
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise RuntimeError("Invalid SHMEM SDK manifest checksum entry")
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise RuntimeError(f"Invalid SHMEM SDK manifest path: {relative}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"SHMEM SDK checksum mismatch: {relative}")


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
