"""CPU-only guards for the independent historical build, never CANN or NPU."""
import ast
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import types

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cb59_original_build as original


def historical(path):
    return original._original(original.ROOT, path).decode("utf-8")


def ast_nodes(text):
    return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(text).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))}


def test_plan_is_pure_and_complete(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("plan started a subprocess")
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "check_output", forbidden)
    group = tmp_path / "with spaces/new group"
    plan = original.make_plan(group, root=tmp_path / "repo with spaces", python="test-python")
    assert not group.exists()
    assert plan["reference"] == original.REFERENCE
    assert json.loads(json.dumps(plan)) == plan
    steps = {step["name"]: step for step in plan["steps"]}
    assert "--no-hardlinks" in steps["clone"]["argv"]
    assert "--dissociate" in steps["clone"]["argv"]
    assert "--no-checkout" in steps["clone"]["argv"]
    assert "--shared" not in steps["clone"]["argv"]
    assert steps["checkout"]["argv"][-1] == original.REFERENCE
    assert steps["submodules"]["argv"] == ["git", "submodule", "update", "--init", "--recursive", "--depth", "1"]
    assert "--system-site-packages" in steps["venv"]["argv"]
    assert "--no-isolation" in steps["wheel"]["argv"]
    assert "--wheel" in steps["clean-wheel"]["argv"]
    assert "-I" in steps["verify-installed"]["argv"]
    assert "_verify-dependencies" in steps["build-dependencies"]["argv"]
    assert "pip" not in steps["build-dependencies"]["argv"]
    assert plan["receipt"].endswith("build-receipt.json")
    assert all(step["cwd"].startswith(str(group)) for step in plan["steps"])
    assert plan["precondition"][-1] == original.REFERENCE + "^{commit}"


def test_environment_removes_current_experiment_without_changing_toolchain(tmp_path):
    env = original.build_environment(tmp_path, {
        "DEEPEP_LL_STAGE_TIMER": "ON", "DEEPEP_LL_DISPATCH_IMPL": "cb59-small",
        "DEEPEP_LL_SEND_CORES": "30", "DEEPEP_SKIP_HT": "ON", "DEEPEP_BUILD_SKIP_HT": "ON",
        "SHMEM_ROOT": "/wrong/sdk", "CATLASS_ROOT": "/wrong/catlass",
        "PYTHONPATH": "/wrong/python", "CXXFLAGS": "original compiler flags",
        "ASCEND_HOME_PATH": "/external/cann", "LD_LIBRARY_PATH": "/external/cann/lib64",
        "ASCEND_RT_VISIBLE_DEVICES": "0,1", "CCACHE_DISABLE": "0",
    })
    assert {name for name in env if name.startswith("DEEPEP_LL_")} == {
        "DEEPEP_LL_STAGE_TIMER", "DEEPEP_LL_RANKS_PER_GROUP"}
    assert env["DEEPEP_LL_STAGE_TIMER"] == "OFF"
    assert env["DEEPEP_LL_RANKS_PER_GROUP"] == "16"
    assert "DEEPEP_SKIP_HT" not in env and "DEEPEP_BUILD_SKIP_HT" not in env
    assert "SHMEM_ROOT" not in env and "CATLASS_ROOT" not in env and "PYTHONPATH" not in env
    assert env["CXXFLAGS"] == "original compiler flags"
    assert env["ASCEND_RT_VISIBLE_DEVICES"] == "0,1"
    assert env["CCACHE_DISABLE"] == "1"
    for name in ("TMPDIR", "PIP_CACHE_DIR", "ASCEND_CACHE_PATH", "ASCEND_WORK_PATH", "SHMEM_LOG_PATH"):
        assert Path(env[name]).is_relative_to(tmp_path / "source")


def test_build_patch_keeps_original_build_graph_and_compiler_options():
    before = historical("build_support.py")
    after = original.patch_build_support(before)
    original_nodes, patched_nodes = ast_nodes(before), ast_nodes(after)
    for name in original_nodes.keys() - {"validate_shmem_sdk", "build_configuration"}:
        assert original_nodes[name] == patched_nodes[name], name
    assert "validate_shmem_sdk" not in patched_nodes
    assert "deepep-sdk.json" not in after
    assert "header_sha256_lf" not in after
    assert "shmem_device_headers" not in after
    assert 'result["device_compiler_sha256"]' in after
    assert 'result["source_sha256"]' in after
    assert 'result["compiler_sha256"]' in after
    assert 'result["cann"]' in after
    assert 'result["pybind11"]' in after
    # Only the intended three regions can differ; no wholesale builder port.
    assert before[before.index("class VariantBuild"): ] == after[after.index("class VariantBuild"): ]


def test_native_patch_preserves_extension_ownership_and_runtime_abi():
    before = historical("deep_ep/_native.py")
    after = original.patch_native(before)
    old_nodes, new_nodes = ast_nodes(before), ast_nodes(after)
    assert all(new_nodes[name] == old_nodes[name] for name in old_nodes if name != "_load_native")
    assert 'module.runtime_protocol != 1' in after
    assert 'Native extension must belong to this deep_ep installation' in after
    assert "shmem_library_path" not in after and "file_digest" not in after


def test_st_patch_only_removes_sdk_mapping_classes():
    before = historical("tests/elastic/test_ep_low_latency.py")
    after = original.patch_preflight(before)
    restored = after.replace("Cannot audit project native mappings", "Cannot audit native/SDK mappings")
    anchor = '        [p for p in mapped if p.endswith("/libdeepep_kernels.so")],\n'
    restored = restored.replace(anchor, anchor +
        '        [p for p in mapped if p.endswith("/libshmem.so")],\n' +
        '        [p for p in mapped if p.endswith("/aclshmem_bootstrap_config_store.so")],\n')
    assert restored == before
    assert '"/deep_ep/_C."' in after and 'hashlib.sha256(Path(paths[0]).read_bytes())' in after
    assert ast_nodes(after)["make_routes"] == ast_nodes(before)["make_routes"]


def test_cmake_patch_retains_normal_dependency_linking():
    before = historical("cmake/Dependencies.cmake")
    after = original.patched_text("cmake/Dependencies.cmake", before)
    assert after.replace("foreach(sdk ASCEND_HOME_PATH)", "foreach(sdk ASCEND_HOME_PATH SHMEM_ROOT)") == before
    assert 'find_library(SHMEM_LIBRARY NAMES shmem' in after


@pytest.mark.parametrize("patch,path", [(original.patch_build_support, "build_support.py"),
                                       (original.patch_native, "deep_ep/_native.py"),
                                       (original.patch_preflight, "tests/elastic/test_ep_low_latency.py")])
def test_patches_reject_reapplication(patch, path):
    with pytest.raises(ValueError):
        patch(patch(historical(path)))


def test_verify_preserves_byte_exact_csrc_and_python(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    archive = original._git(original.ROOT, "archive", original.REFERENCE, "csrc", "deep_ep")
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        for member in stream.getmembers():
            if member.isfile():
                path = source / member.name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(stream.extractfile(member).read())
    for relative in original.PATCH_PATHS:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((original.ROOT / relative).read_bytes() if relative in original.COPIED_PATHS
                         else original.patched_text(relative, historical(relative)).encode())
    real_git = original._git
    def fake_git(root, *args, text=False):
        if args[:2] == ("rev-parse", "HEAD"):
            return original.REFERENCE + "\n"
        if args[:2] == ("diff", "--name-only"):
            return "\n".join(original.PATCH_PATHS)
        return real_git(original.ROOT, *args, text=text)
    monkeypatch.setattr(original, "_git", fake_git)
    assert original.verify_source(source, original.ROOT)["csrc_byte_identical"] is True
    target = next((source / "csrc").rglob("*.cpp"))
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="implementation bytes changed"):
        original.verify_source(source, original.ROOT)


def test_execute_fails_missing_object_before_writing(tmp_path, monkeypatch):
    group = tmp_path / "new"
    plan = original.make_plan(group)
    monkeypatch.setattr(original.platform, "system", lambda: "Linux")
    def missing(argv, **kwargs):
        raise subprocess.CalledProcessError(1, argv)
    monkeypatch.setattr(subprocess, "run", missing)
    with pytest.raises(RuntimeError, match="fetch origin " + original.REFERENCE):
        original.execute(plan)
    assert not group.exists()


def test_execute_refuses_existing_group_and_changed_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(original.platform, "system", lambda: "Linux")
    with pytest.raises(ValueError, match="must be new"):
        original.execute(original.make_plan(tmp_path))
    plan = original.make_plan(tmp_path / "new")
    plan["steps"][0]["argv"].append("--shared")
    with pytest.raises(ValueError, match="differs"):
        original.execute(plan)


def test_execute_stops_on_first_command_failure_and_preserves_log(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts/clean_build_artifacts.py").write_bytes((original.ROOT / "scripts/clean_build_artifacts.py").read_bytes())
    group = root / "output/new"
    plan = original.make_plan(group, root=root)
    calls = []
    monkeypatch.setattr(original.platform, "system", lambda: "Linux")
    def run(argv, **kwargs):
        calls.append(argv)
        if "clone" in argv:
            kwargs["stdout"].write(b"fixture clone failure\n")
            raise subprocess.CalledProcessError(1, argv)
        return types.SimpleNamespace(returncode=0)
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        original.execute(plan)
    assert len(calls) == 2
    assert (group / "logs/00-clone.log").read_bytes() == b"fixture clone failure\n"
    assert not (group / "build-receipt.json").exists()
    assert not (group / "source").exists()


def test_dependency_check_uses_specs_without_importing_framework(monkeypatch):
    import importlib.util
    seen = []
    def spec(name):
        seen.append(name)
        return object()
    monkeypatch.setattr(importlib.util, "find_spec", spec)
    original.verify_dependencies()
    assert seen == ["torch", "torch_npu", "build", "setuptools", "wheel", "pybind11"]
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None if name == "torch_npu" else object())
    with pytest.raises(RuntimeError, match="torch_npu.*does not install or update"):
        original.verify_dependencies()


def test_install_selects_only_own_wheel_and_venv(tmp_path, monkeypatch):
    wheel_root = tmp_path / "wheels/off"
    wheel_root.mkdir(parents=True)
    wheel = wheel_root / "deep_ep-original.whl"
    wheel.write_bytes(b"fixture")
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda argv, **kwargs: calls.append((argv, kwargs)))
    original.install_wheel(tmp_path)
    argv, kwargs = calls[0]
    assert argv[0] == str(tmp_path / "venv/bin/python")
    assert "--no-index" in argv and "--no-deps" in argv
    assert argv[-1] == str(wheel)
    assert kwargs["check"] is True
    (wheel_root / "stale.whl").write_bytes(b"stale")
    with pytest.raises(ValueError, match="exactly one"):
        original.install_wheel(tmp_path)


def test_host_verification_does_not_query_device():
    import inspect
    code = inspect.getsource(original.verify_installed)
    assert "low_latency_core_config(" not in code
    assert "Session(" not in code and "torch" not in code
    assert '"device_queried": False' in code
    assert 'info["source_sha256"]' in code


def test_host_verification_records_toolchain_without_querying_device(tmp_path, monkeypatch):
    package = tmp_path / "venv/lib/deep_ep"
    package.mkdir(parents=True)
    (tmp_path / "source").mkdir()
    (tmp_path / "source/VERSION").write_text("original-version\n")
    (tmp_path / "logs").mkdir()
    info = {"source_version": "original-version", "source_sha256": "source-hash", "arch": "Ascend950",
            "build_type": "Release", "ll_stage_timer": False, "megamoe": {"enabled": False},
            "ep_gmm_fused": {"enabled": False}, "cann": {"version": "9.3.0"},
            "compiler_sha256": "host-hash", "device_compiler_sha256": "device-hash", "python": "test-python"}
    (package / "_build_info.json").write_text(json.dumps(info))
    native = types.SimpleNamespace(runtime_protocol=1, low_latency_core_protocol=2,
                                   low_latency_ranks_per_group=16, low_latency_stage_timer_enabled=False)
    monkeypatch.setitem(sys.modules, "deep_ep", types.SimpleNamespace(__file__=str(package / "__init__.py")))
    monkeypatch.setitem(sys.modules, "deep_ep._native", types.SimpleNamespace(require_native=lambda _: native))
    import runpy
    monkeypatch.setattr(runpy, "run_path", lambda _: {"source_fingerprint": lambda: "source-hash"})
    original.verify_installed(tmp_path)
    result = json.loads((tmp_path / "logs/host_verify.json").read_text())
    assert result["device_queried"] is False
    assert result["build_metadata"]["compiler_sha256"] == "host-hash"
    assert result["build_metadata"]["device_compiler_sha256"] == "device-hash"
    assert result["build_metadata"]["cann"] == {"version": "9.3.0"}
    info["source_sha256"] = "mismatched-source"
    (package / "_build_info.json").write_text(json.dumps(info))
    with pytest.raises(ValueError, match="does not match"):
        original.verify_installed(tmp_path)


def test_receipt_sdk_revision_is_informational_only(tmp_path, monkeypatch):
    (tmp_path / "source").mkdir()
    (tmp_path / "source/dependencies.lock.json").write_text(json.dumps({"shmem": {"revision": "old-sdk"}}))
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs/host_verify.json").write_text('{}')
    (tmp_path / "wheels/off").mkdir(parents=True)
    (tmp_path / "wheels/off/original.whl").write_bytes(b"wheel")
    monkeypatch.setattr(original, "verify_source", lambda *_: {"reference": original.REFERENCE})
    def fake_git(source, *args, **kwargs):
        if args[:2] == ("rev-parse", "HEAD"):
            raise subprocess.CalledProcessError(1, ["git"])
        return b"fixture"
    monkeypatch.setattr(original, "_git", fake_git)
    path = original.write_receipt(tmp_path, original.ROOT)
    assert path.name == "build-receipt.json"
    receipt = json.loads(path.read_text())
    assert receipt["sdk_source_metadata"] == {"expected_source_revision": "old-sdk",
        "observed_source_revision": "unknown", "informational_only": True}
