#!/usr/bin/env python3
"""Rebuild the immutable cb59 original in an independent, clean OFF group.

The original CMake graph, compiler options, Host/device code and SDK submodule
are retained. Only SDK admission checks and the clean SDK builder are patched.
Planning is read-only and does not import a framework or start a subprocess.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = "cb59bff99ecd39d27ef57788c5e6fd0009e4e636"
FORMAT = "deepep-cb59-original-build-v1"
PATCH_PATHS = (
    "build_support.py", "cmake/Dependencies.cmake", "deep_ep/_native.py",
    "tests/elastic/test_ep_low_latency.py", "scripts/build_shmem.py",
    "scripts/clean_build_artifacts.py",
)
COPIED_PATHS = ("scripts/build_shmem.py", "scripts/clean_build_artifacts.py")


def _replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f"Original source anchor is not unique: {before[:90]!r}")
    return text.replace(before, after, 1)


def _replace_region(text, start, end, replacement):
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError("Original source region is missing or ambiguous")
    first, last = text.index(start), text.index(end)
    if last <= first:
        raise ValueError("Original source region is reversed")
    return text[:first] + replacement + text[last:]


def patch_build_support(text):
    tree = ast.parse(text)
    matches = [node for node in tree.body if isinstance(node, ast.FunctionDef)
               and node.name == "validate_shmem_sdk"]
    if len(matches) != 1:
        raise ValueError("Expected the original validate_shmem_sdk function")
    node = matches[0]
    lines = text.splitlines(keepends=True)
    text = "".join(lines[:node.lineno - 1] + lines[node.end_lineno:])
    text = _replace_region(
        text, "    for name, header in (\n", '    result["pybind11"] =',
        '    cann = env.get("ASCEND_HOME_PATH")\n'
        '    if not cann or not (Path(cann) / "include/acl/acl.h").is_file():\n'
        '        raise RuntimeError("ASCEND_HOME_PATH must name an installed SDK containing include/acl/acl.h")\n'
        '    result["ASCEND_HOME_PATH"] = str(Path(cann).resolve())\n'
        '    result["SHMEM_ROOT"] = str(Path(env.get("SHMEM_ROOT", default_shmem)).resolve())\n'
        '    lock = json.loads((ROOT / "dependencies.lock.json").read_text(encoding="utf-8"))\n'
        '    # Fresh source build: SDK identity and hash admission are intentionally absent.\n'
        '    result["shmem"] = {"revision": "not-recorded"}\n',
    )
    return _replace_region(text, '    sdk_root = Path(result["SHMEM_ROOT"])\n',
                           '    result["ep_gmm_fused"] = {', "")


def patch_native(text):
    return _replace_region(
        text, '    info = json.loads(Path(__file__).with_name("_build_info.json")',
        "    return module\n", "",
    )


def patch_preflight(text):
    for line in (
        '        [p for p in mapped if p.endswith("/libshmem.so")],\n',
        '        [p for p in mapped if p.endswith("/aclshmem_bootstrap_config_store.so")],\n',
    ):
        text = _replace_once(text, line, "")
    return _replace_once(text, "Cannot audit native/SDK mappings", "Cannot audit project native mappings")


def patched_text(relative, text):
    if relative == "build_support.py":
        return patch_build_support(text)
    if relative == "deep_ep/_native.py":
        return patch_native(text)
    if relative == "tests/elastic/test_ep_low_latency.py":
        return patch_preflight(text)
    if relative == "cmake/Dependencies.cmake":
        return _replace_once(text, "foreach(sdk ASCEND_HOME_PATH SHMEM_ROOT)", "foreach(sdk ASCEND_HOME_PATH)")
    raise ValueError(f"Not a source patch: {relative}")


def _git(source, *args, text=False):
    return subprocess.check_output(["git", "-C", str(source), *args], text=text, timeout=90)


def _original(source, relative):
    return _git(source, "show", f"{REFERENCE}:{relative}")


def apply_patches(source, root):
    source, root = Path(source), Path(root)
    # Compute every patch before writing anything, rejecting drift in the clone.
    changes = {}
    for relative in PATCH_PATHS:
        target = source / relative
        if relative in COPIED_PATHS:
            changes[relative] = (root / relative).read_bytes()
        else:
            original = _original(source, relative)
            if target.read_bytes() != original:
                raise ValueError(f"Original source changed before patching: {relative}")
            changes[relative] = patched_text(relative, original.decode("utf-8")).encode("utf-8")
    for relative, data in changes.items():
        (source / relative).write_bytes(data)


def verify_source(source, root):
    """Prove exact original csrc and Python bytes outside the narrow loader patch."""
    source, root = Path(source), Path(root)
    if _git(source, "rev-parse", "HEAD", text=True).strip() != REFERENCE:
        raise ValueError("Original source checkout is not the immutable cb59 reference")
    if (source / ".git/objects/info/alternates").exists():
        raise ValueError("Original clone must own its objects, without borrowed Git alternates")
    archive = _git(source, "archive", REFERENCE, "csrc", "deep_ep")
    expected = set()
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        for member in stream.getmembers():
            if not member.isfile():
                continue
            expected.add(member.name)
            if member.name == "deep_ep/_native.py":
                continue
            if (source / member.name).read_bytes() != stream.extractfile(member).read():
                raise ValueError(f"Original implementation bytes changed: {member.name}")
    actual = {path.relative_to(source).as_posix() for folder in ("csrc", "deep_ep")
              for path in (source / folder).rglob("*") if path.is_file()}
    if actual != expected:
        raise ValueError("Original csrc/deep_ep file inventory changed")
    for relative in PATCH_PATHS:
        expected_data = ((root / relative).read_bytes() if relative in COPIED_PATHS
                         else patched_text(relative, _original(source, relative).decode("utf-8")).encode("utf-8"))
        if (source / relative).read_bytes() != expected_data:
            raise ValueError(f"Unexpected original source patch: {relative}")
    changed = set(_git(source, "diff", "--name-only", REFERENCE, "--", ".",
                       ":(exclude)third-party", text=True).splitlines())
    if changed - set(PATCH_PATHS):
        raise ValueError(f"Changes outside the original reproduction allowlist: {sorted(changed)}")
    # The original monolithic entry, all dtype branches and original flags are
    # covered by the byte comparison. No installed-device core query is made.
    return {"reference": REFERENCE, "source_patch_paths": list(PATCH_PATHS),
            "csrc_byte_identical": True, "deep_ep_original_except_native_loader": True}


def build_environment(group, inherited=None):
    group = Path(group)
    source = group / "source"
    env = dict(os.environ if inherited is None else inherited)
    for name in tuple(env):
        if name.startswith("DEEPEP_") or name in {"SHMEM_ROOT", "CATLASS_ROOT", "PYTHONPATH", "PYTHONHOME"}:
            del env[name]
    cache = source / ".cache/original-repro"
    logs = source / ".logs/original-repro"
    env.update({
        "DEEPEP_NPU_ARCH": "Ascend950", "DEEPEP_BUILD_TYPE": "Release",
        "DEEPEP_BUILD_MEGAMOE": "OFF", "DEEPEP_MEGAMOE_TIMER": "OFF",
        "DEEPEP_BUILD_EP_GMM_FUSED": "OFF", "DEEPEP_LL_STAGE_TIMER": "OFF",
        "DEEPEP_LL_RANKS_PER_GROUP": "16", "CCACHE_DISABLE": "1",
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1", "PIP_CACHE_DIR": str(cache / "pip"),
        "XDG_CACHE_HOME": str(cache), "TMPDIR": str(cache / "tmp"),
        "TMP": str(cache / "tmp"), "TEMP": str(cache / "tmp"),
        "ASCEND_CACHE_PATH": str(cache / "ascend"), "ASCEND_WORK_PATH": str(logs / "ascend"),
        "ASCEND_PROCESS_LOG_PATH": str(logs / "ascend"), "SHMEM_LOG_PATH": str(logs / "shmem"),
    })
    own_library = str(source / "third-party/shmem/install/shmem/lib")
    env["LD_LIBRARY_PATH"] = own_library + (os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    return env


def make_plan(group, root=ROOT, python=sys.executable):
    """Return all target-side commands without checking files or creating a path."""
    group, root = Path(group).absolute(), Path(root).absolute()
    source, venv = group / "source", group / "venv"
    vpython = venv / "bin/python"
    helper = root / "tests/ll_stage_timer/cb59_original_build.py"
    steps = []

    def add(name, argv, cwd=group):
        steps.append({"name": name, "argv": [str(x) for x in argv], "cwd": str(cwd)})

    add("clone", ["git", "-c", "core.autocrlf=false", "clone", "--no-hardlinks", "--dissociate", "--no-checkout", root, source])
    add("checkout", ["git", "-c", "core.autocrlf=false", "checkout", "--detach", REFERENCE], source)
    add("submodules", ["git", "submodule", "update", "--init", "--recursive", "--depth", "1"], source)
    add("patch", [python, helper, "_patch", "--source", source, "--root", root])
    add("verify-source", [python, helper, "_verify-source", "--source", source, "--root", root])
    add("venv", [python, "-m", "venv", "--system-site-packages", venv])
    add("build-dependencies", [vpython, "-I", helper, "_verify-dependencies"], source)
    add("sdk", [vpython, source / "scripts/build_shmem.py"], source)
    add("clean-wheel", [vpython, source / "scripts/clean_build_artifacts.py", "--wheel"], source)
    add("wheel", [vpython, "-m", "build", "--wheel", "--no-isolation", "--outdir", group / "wheels/off", source], source)
    add("install", [python, helper, "_install", "--group", group])
    add("verify-installed", [vpython, "-I", helper, "_verify-installed", "--group", group])
    add("receipt", [python, helper, "_receipt", "--group", group, "--root", root])
    return {"format": FORMAT, "reference": REFERENCE, "group": str(group), "root": str(root),
            "python": str(python), "source": str(source), "venv": str(venv),
            "wheels": str(group / "wheels/off"), "receipt": str(group / "build-receipt.json"),
            "source_patch_paths": list(PATCH_PATHS),
            "precondition": ["git", "-C", str(root), "cat-file", "-e", f"{REFERENCE}^{{commit}}"],
            "environment": build_environment(group, {}), "steps": steps}


def _one_wheel(group):
    wheels = sorted((Path(group) / "wheels/off").glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError(f"Expected exactly one original OFF wheel, found {len(wheels)}")
    return wheels[0]


def install_wheel(group):
    group = Path(group)
    subprocess.run([str(group / "venv/bin/python"), "-m", "pip", "install", "--no-index", "--no-deps",
                    "--force-reinstall", str(_one_wheel(group))], cwd=group, check=True)


def verify_dependencies():
    """Check availability in the inherited environment without importing Torch."""
    import importlib.util
    required = ("torch", "torch_npu", "build", "setuptools", "wheel", "pybind11")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError("The selected Python environment is missing existing build/runtime dependencies: "
                           + ", ".join(missing)
                           + ". Select or prepare the same installed environment used for the previous builds; "
                           "this reproduction does not install or update dependencies.")
    print("Existing dependency modules found (no Torch import): " + ", ".join(required))


def verify_installed(group):
    """Host loading only; never construct a Session, tensor or query AIV cores."""
    group = Path(group)
    source = group / "source"
    import deep_ep
    from deep_ep._native import require_native
    package = Path(deep_ep.__file__).resolve().parent
    if not package.is_relative_to((group / "venv").resolve()):
        raise ValueError("Host verification imported a package outside the isolated venv")
    native = require_native("original build host verification")
    expected = {"runtime_protocol": 1, "low_latency_core_protocol": 2,
                "low_latency_ranks_per_group": 16, "low_latency_stage_timer_enabled": False}
    for name, value in expected.items():
        if getattr(native, name) != value:
            raise ValueError(f"Original wheel capability mismatch: {name}")
    info = json.loads((package / "_build_info.json").read_text(encoding="utf-8"))
    if (info["source_version"] != (source / "VERSION").read_text().strip()
            or info["arch"] != "Ascend950" or info["build_type"] != "Release"
            or info["ll_stage_timer"] is not False or info["megamoe"]["enabled"] is not False
            or info["ep_gmm_fused"]["enabled"] is not False):
        raise ValueError("Installed wheel is not the requested original OFF configuration")
    # Source digest protects the wheel implementation, not SHMEM SDK identity.
    import runpy
    support = runpy.run_path(str(source / "build_support.py"))
    if info["source_sha256"] != support["source_fingerprint"]():
        raise ValueError("Installed wheel does not match the isolated patched original source")
    result = {"host_only": True, "device_queried": False, "capabilities": expected,
              "source_sha256": info["source_sha256"], "source_version": info["source_version"],
              "core_policy": "original dynamic device/platform AIV minimum, 1:1 split",
              "build_metadata": {key: info.get(key) for key in (
                  "cann", "compiler", "compiler_path", "compiler_sha256", "compiler_version_sha256",
                  "device_compiler_sha256", "cmake_path", "cmake_version", "python", "python_executable",
                  "machine", "arch", "build_type", "build_environment")}}
    (group / "logs/host_verify.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


def write_receipt(group, root):
    group, root = Path(group), Path(root)
    proof = verify_source(group / "source", root)
    host = json.loads((group / "logs/host_verify.json").read_text(encoding="utf-8"))
    wheel = _one_wheel(group)
    # Observational metadata only. Neither a mismatch nor an unavailable Git
    # revision changes SDK admission; the SDK was rebuilt from the checkout.
    expected_sdk = json.loads((group / "source/dependencies.lock.json").read_text(encoding="utf-8")).get("shmem", {}).get("revision", "unknown")
    try:
        observed_sdk = _git(group / "source/third-party/shmem", "rev-parse", "HEAD", text=True).strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        observed_sdk = "unknown"
    receipt = {"format": FORMAT, **proof, "scope": "original-off-build; host-only verification; no NPU run",
               "source": "source", "python": "venv/bin/python", "wheel": wheel.relative_to(group).as_posix(),
               "host_verification": "logs/host_verify.json", "source_diff": "logs/source_patch.diff",
               "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(), "host": host,
               "sdk_source_metadata": {"expected_source_revision": expected_sdk,
                                       "observed_source_revision": observed_sdk,
                                       "informational_only": True}}
    diff = _git(group / "source", "diff", REFERENCE, "--", *PATCH_PATHS)
    # The helper newly introduced into the original tree is untracked in Git.
    for relative in COPIED_PATHS:
        if not _git(group / "source", "ls-files", "--", relative).strip():
            data = (group / "source" / relative).read_text(encoding="utf-8").splitlines()
            diff += (f"\ndiff --git a/{relative} b/{relative}\nnew file mode 100644\n--- /dev/null\n"
                     f"+++ b/{relative}\n@@ -0,0 +1,{len(data)} @@\n" +
                     "".join("+" + line + "\n" for line in data)).encode("utf-8")
    (group / "logs/source_patch.diff").write_bytes(diff)
    path = group / "build-receipt.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return path


def execute(plan):
    if platform.system() != "Linux":
        raise RuntimeError("The original build executes only on the Linux target with CANN; --plan is CPU-only")
    if plan != make_plan(plan["group"], root=plan["root"], python=plan["python"]):
        raise ValueError("Build plan differs from the fixed original reproduction")
    group, root = Path(plan["group"]), Path(plan["root"])
    if group.exists() or group.is_symlink():
        raise ValueError(f"Build group must be new; existing data is preserved: {group}")
    # Verify local object availability before even creating a log directory.
    try:
        subprocess.run(plan["precondition"], check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=90)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"The immutable original object is missing. Run git -C {root} fetch origin {REFERENCE} first") from exc
    sys.path.insert(0, str(root / "scripts"))
    try:
        from clean_build_artifacts import assert_local_path
        # The group is project-local and no ancestor may redirect its writes.
        assert_local_path(root, group)
    finally:
        sys.path.pop(0)
    group.mkdir(parents=True)
    for relative in ("logs", "wheels/off"):
        (group / relative).mkdir(parents=True)
    (group / "build_plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    env = build_environment(group)
    for index, step in enumerate(plan["steps"]):
        # Clone requires a nonexistent source; provision SDK-local logs/cache
        # only after checkout has created that project directory.
        if step["name"] == "submodules":
            for name in ("TMPDIR", "XDG_CACHE_HOME", "PIP_CACHE_DIR", "ASCEND_CACHE_PATH",
                         "ASCEND_WORK_PATH", "ASCEND_PROCESS_LOG_PATH", "SHMEM_LOG_PATH"):
                Path(env[name]).mkdir(parents=True, exist_ok=True)
        log = group / "logs" / f"{index:02d}-{step['name']}.log"
        print(f"[original-build] {step['name']} -> {log}", flush=True)
        with log.open("wb") as stream:
            subprocess.run(step["argv"], cwd=step["cwd"], env=env, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
    return Path(plan["receipt"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("_patch", "_verify-source", "_verify-dependencies", "_install", "_verify-installed", "_receipt"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--group", type=Path)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    if args.action == "_patch":
        apply_patches(args.source, args.root)
    elif args.action == "_verify-source":
        print(json.dumps(verify_source(args.source, args.root), sort_keys=True))
    elif args.action == "_verify-dependencies":
        verify_dependencies()
    elif args.action == "_install":
        install_wheel(args.group)
    elif args.action == "_verify-installed":
        verify_installed(args.group)
    else:
        print(write_receipt(args.group, args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
