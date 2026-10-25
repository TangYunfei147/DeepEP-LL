# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Record or verify a prebuilt SDK; never download/build dependencies.

Header checks establish the expected public API, not binary provenance. The SDK
provider must attest that the library was built from the pinned commit.
HNS1825 identities also attest the actual compiled PI-on-chip doorbell mode;
build_shmem.py obtains it from the SDK device target's generated compile rules.
"""

import argparse
try:
    from scripts.shmem_sdk_identity import compiled_hns_1825_pi_on_chip, validate_build_identity
except ModuleNotFoundError:
    from shmem_sdk_identity import compiled_hns_1825_pi_on_chip, validate_build_identity
import hashlib
import json
from pathlib import Path


def record(sdk, revision, *, build_info=None, build_root=None):
    validate_build_identity(build_info)
    if build_info.get("rdma_backend") == "HNS_1825":
        if build_root is None:
            raise ValueError("HNS1825 SDK recording requires its actual device build graph (--build-root)")
        compiled_mode = compiled_hns_1825_pi_on_chip(build_root)
        if compiled_mode != build_info["hns_1825_pi_on_chip"]:
            raise ValueError("HNS1825 doorbell mode differs between build_info and the SDK device build graph")
    sdk = Path(sdk).resolve()
    lock = json.loads((Path(__file__).resolve().parents[1] / "dependencies.lock.json").read_text())
    expected = lock["shmem"]
    if revision != expected["revision"]:
        raise ValueError("SDK source revision must match dependencies.lock.json")
    for relative, digest in expected["header_sha256_lf"].items():
        header = sdk / "include" / relative
        normalized = header.read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(normalized).hexdigest() != digest:
            raise ValueError(f"SDK public header mismatch: {relative}")
    files = sorted((sdk / "include").rglob("*.h")) + [sdk / "lib/libshmem.so"]
    hashes = {}
    for path in files:
        if not path.resolve().is_relative_to(sdk) or not path.is_file():
            raise ValueError("SDK files must stay inside the selected root")
        hashes[path.relative_to(sdk).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = sdk / "deepep-sdk.json"
    if manifest.exists():
        raise FileExistsError("SDK manifest already exists; do not overwrite an existing SDK identity")
    identity = {"revision": revision, "sha256": hashes}
    if build_info is not None:
        identity["build"] = build_info
    manifest.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    return manifest


def check(sdk):
    """Validate an existing SDK identity without changing its files."""
    sdk = Path(sdk).resolve()
    lock = json.loads((Path(__file__).resolve().parents[1] / "dependencies.lock.json").read_text())["shmem"]
    manifest = sdk / "deepep-sdk.json"
    recorded = json.loads(manifest.read_text())
    validate_build_identity(recorded.get("build"))
    if recorded.get("revision") != lock["revision"]:
        raise ValueError("SDK source revision must match dependencies.lock.json")
    for relative, expected in lock["header_sha256_lf"].items():
        data = (sdk / "include" / relative).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"SDK public header mismatch: {relative}")
    hashes = recorded.get("sha256", {})
    required = {"lib/libshmem.so", *("include/" + name for name in lock["header_sha256_lf"])}
    if not required.issubset(hashes):
        raise ValueError("SDK manifest must cover the locked headers and libshmem.so")
    for relative, expected in hashes.items():
        path = (sdk / relative).resolve()
        if not path.is_relative_to(sdk) or not path.is_file():
            raise ValueError(f"Invalid SDK manifest path: {relative}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"SDK checksum mismatch: {relative}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--revision", help="SDK provider's source revision attestation; requires --build-info")
    mode.add_argument("--check", "--verify", dest="check", action="store_true", help="verify an existing SDK identity without modifying it")
    parser.add_argument("--build-info", type=Path, help="JSON file with complete SDK build identity")
    parser.add_argument("--build-root", type=Path, help="SDK build directory containing actual device compile rules; required for HNS1825")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]

    def project_path(value):
        path = Path(value)
        return path if path.is_absolute() else project_root / path

    if args.check:
        if args.build_info or args.build_root:
            parser.error("--check/--verify does not accept recording-only build arguments")
        print(check(project_path(args.sdk)))
    else:
        if args.build_info is None:
            parser.error("--revision requires --build-info")
        build_info = json.loads(project_path(args.build_info).read_text(encoding="utf-8"))
        print(record(project_path(args.sdk), args.revision, build_info=build_info,
                     build_root=project_path(args.build_root) if args.build_root else None))
