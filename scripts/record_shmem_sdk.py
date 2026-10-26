# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Record or verify a prebuilt SDK; never download/build dependencies.

Header checks establish the expected public API, not binary provenance. The SDK
provider must attest that the library was built from the pinned commit.
"""

import argparse
try:
    from scripts.shmem_sdk_identity import validate_build_identity
except ModuleNotFoundError:
    from shmem_sdk_identity import validate_build_identity
import hashlib
import json
from pathlib import Path


def sdk_input_files(sdk):
    """Return every installed compile/link input, including Device headers.

    The full SDK installs include/, src/{device,device_simt,host_device}/ and
    lib/. A manifest limited to public *.h files misses implementation *.hpp
    files and bootstrap/helper libraries that participate in the real build.
    """
    sdk = Path(sdk).resolve()
    required = ("include/shmem.h", "src/device/gm2gm/shmemi_device_cc.h", "lib/libshmem.so")
    for relative in required:
        path = sdk / relative
        if not path.is_file():
            raise ValueError(f"Incomplete SHMEM SDK: missing {relative}")
        if not path.resolve().is_relative_to(sdk):
            raise ValueError(f"SDK files must stay inside the selected root: {relative}")
    files = []
    for directory in ("include", "src", "lib"):
        for path in (sdk / directory).rglob("*"):
            if path.is_symlink() and not path.resolve().is_relative_to(sdk):
                raise ValueError(f"SDK files must stay inside the selected root: {path.relative_to(sdk)}")
            if path.is_file():
                if not path.resolve().is_relative_to(sdk):
                    raise ValueError(f"SDK files must stay inside the selected root: {path.relative_to(sdk)}")
                files.append(path)
    return sorted(files)


def require_manifest_coverage(sdk, hashes, required=None):
    """Explain an incomplete identity without attesting or changing any input."""
    if not isinstance(hashes, dict) or any(not isinstance(key, str) for key in hashes):
        raise ValueError("Invalid SDK manifest checksums")
    sdk = Path(sdk).resolve()
    if required is None:
        required = {path.relative_to(sdk).as_posix() for path in sdk_input_files(sdk)}
    missing = sorted(set(required) - set(hashes))
    if not missing:
        return
    counts = ", ".join(
        f"{name}={sum(path.startswith(name + '/') for path in missing)}" for name in ("include", "src", "lib"))
    examples = ", ".join(missing[:8])
    remaining = f"; {len(missing) - 8} more" if len(missing) > 8 else ""
    raise ValueError(
        "SHMEM SDK manifest must cover all installed include/src/lib inputs: "
        f"{len(missing)} unrecorded files ({counts}); examples: {examples}{remaining}. "
        "An old/partial manifest or changed installation cannot be repaired by accepting a build field. "
        "After stopping users of this SDK, rebuild explicitly with python scripts/build_shmem.py --rebuild; "
        "do not regenerate hashes over unverified old binaries."
    )


def record(sdk, revision, *, build_info=None):
    validate_build_identity(build_info)
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
    files = sdk_input_files(sdk)
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
    with manifest.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(identity, stream, indent=2)
        stream.write("\n")
    return manifest


def check(sdk):
    """Validate an existing SDK identity without changing its files."""
    sdk = Path(sdk).resolve()
    lock = json.loads((Path(__file__).resolve().parents[1] / "dependencies.lock.json").read_text())["shmem"]
    manifest = sdk / "deepep-sdk.json"
    recorded = json.loads(manifest.read_text())
    if not isinstance(recorded, dict):
        raise ValueError("Invalid SDK identity")
    validate_build_identity(recorded.get("build"))
    if recorded.get("revision") != lock["revision"]:
        raise ValueError("SDK source revision must match dependencies.lock.json")
    for relative, expected in lock["header_sha256_lf"].items():
        data = (sdk / "include" / relative).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"SDK public header mismatch: {relative}")
    hashes = recorded.get("sha256", {})
    required = {path.relative_to(sdk).as_posix() for path in sdk_input_files(sdk)}
    required.update("include/" + name for name in lock["header_sha256_lf"])
    require_manifest_coverage(sdk, hashes, required)
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
    mode.add_argument("--revision", help="SDK provider's source revision attestation (complete build_info is required; use build_shmem.py)")
    mode.add_argument("--check", action="store_true", help="verify an existing SDK identity without modifying it")
    args = parser.parse_args()
    print(check(args.sdk) if args.check else record(args.sdk, args.revision))
