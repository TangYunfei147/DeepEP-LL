# Copyright (c) 2026, Lu Lu

"""Build preflight and artifact checks runnable without SDKs or devices."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[2]
if not (ROOT / "build_support.py").is_file():
    pytest.skip("Build-source tests run before packaging", allow_module_level=True)


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def build(monkeypatch):
    module = load(ROOT / "build_support.py")
    monkeypatch.setattr(module.platform, "system", lambda: "Linux")
    return module


def test_platform_and_architecture_rejected(build, monkeypatch):
    with pytest.raises(RuntimeError, match="DEEPEP_NPU_ARCH"):
        build.build_configuration({})
    monkeypatch.setattr(build.platform, "system", lambda: "Windows")
    with pytest.raises(RuntimeError, match="Linux"):
        build.build_configuration({"DEEPEP_NPU_ARCH": "Ascend950"})


def test_clean_native_commands_disable_ccache_without_mutating_callers(build, monkeypatch):
    given = {'CCACHE_DISABLE': '0', 'KEEP': 'value'}
    seen = []
    monkeypatch.setattr(build.subprocess, 'run', lambda argv, **kwargs: seen.append((argv, kwargs)))
    build.checked_run(['cmake', '--build', 'build/native'], env=given)
    assert given == {'CCACHE_DISABLE': '0', 'KEEP': 'value'}
    assert seen[0][1]['env'] == {'CCACHE_DISABLE': '1', 'KEEP': 'value'}
    assert seen[0][1]['check'] is True


@pytest.mark.parametrize("value", ["on", "yes", "1", ""])
def test_beta_never_silently_enabled(build, value):
    with pytest.raises(RuntimeError, match="MegaMoE"):
        build.build_configuration({"DEEPEP_NPU_ARCH": "Ascend950", "DEEPEP_BUILD_MEGAMOE": value})


def test_missing_sdk_rejected(build):
    with pytest.raises(RuntimeError, match="ASCEND_HOME_PATH"):
        build.build_configuration({"DEEPEP_NPU_ARCH": "Ascend950"})


def test_beta_requires_sdk_and_timer_requires_beta(build):
    with pytest.raises(RuntimeError, match="ASCEND_HOME_PATH"):
        build.build_configuration({"DEEPEP_NPU_ARCH": "Ascend950", "DEEPEP_BUILD_MEGAMOE": "ON"})
    with pytest.raises(RuntimeError, match="MegaMoE"):
        build.build_configuration({"DEEPEP_NPU_ARCH": "Ascend950", "DEEPEP_MEGAMOE_TIMER": "ON"})


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "missing",
        "json",
        "list",
        "revision",
        "header",
        "checksum",
        "coverage",
        "path",
        "missing_library",
        "missing_device_header",
        "missing_public_header",
        "unknown_fields",
        "build_type",
    ],
)
def test_shmem_metadata_and_missing_inputs_do_not_block_compiler_execution(build, monkeypatch, tmp_path, fault):
    import hashlib
    import warnings
    from types import SimpleNamespace

    cann, sdk = tmp_path / "cann", tmp_path / "shmem"
    files = {
        cann / "include/acl/acl.h": "",
        cann / "compiler/version.info": "Version=9.2.0",
        cann / "tools/bisheng_compiler/bin/bisheng": "compiler",
        sdk / "include/host/shmem_host_def.h": "host",
        sdk / "include/shmem.h": "public",
        sdk / "src/device/gm2gm/shmemi_device_cc.h": "device",
        sdk / "lib/libshmem.so": "library",
    }
    for path, value in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    lock = json.loads((ROOT / "dependencies.lock.json").read_text())
    lock["shmem"]["header_sha256_lf"] = {"host/shmem_host_def.h": hashlib.sha256(b"host").hexdigest()}
    source = tmp_path / "source"
    source.mkdir()
    (source / "dependencies.lock.json").write_text(json.dumps(lock))
    (source / "VERSION").write_text("test")
    monkeypatch.setattr(build, "ROOT", source)
    monkeypatch.setattr(build, "source_fingerprint", lambda: "test")
    monkeypatch.setattr(build, "find_host_compiler", lambda env: (sys.executable, b"test"))
    monkeypatch.setattr(build.shutil, "which", lambda *args, **kwargs: sys.executable)
    monkeypatch.setattr(build.subprocess, "check_output", lambda *args, **kwargs: "cmake version test")
    monkeypatch.setitem(sys.modules, "pybind11", SimpleNamespace(__version__=lock["python_build"]["pybind11"]))
    identity = {
        "revision": lock["shmem"]["revision"],
        "build": {"cann_version": "test", "cann_timestamp": "test", "target": "Ascend950", "source_patches": []},
        "sha256": {
            str(path.relative_to(sdk).as_posix()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
            if path.is_relative_to(sdk)
        },
    }
    if fault == "revision":
        identity["revision"] = "custom"
    elif fault == "header":
        (sdk / "include/host/shmem_host_def.h").write_text("changed header")
    elif fault == "checksum":
        (sdk / "lib/libshmem.so").write_text("changed library")
    elif fault == "coverage":
        identity["sha256"] = {}
    elif fault == "path":
        identity["sha256"]["../outside"] = "0" * 64
    elif fault == "unknown_fields":
        identity["build"].update(rdma_backend="future-vendor", hns_1825_pi_on_chip={"future": [2, None]}, unknown=True)
    elif fault == "build_type":
        identity["build"] = ["provider-owned", False]
    manifest = sdk / "deepep-sdk.json"
    if fault != "missing":
        manifest.write_text("invalid json" if fault == "json" else "[]" if fault == "list" else json.dumps(identity))
    if fault == "missing_library":
        (sdk / "lib/libshmem.so").unlink()
    elif fault == "missing_device_header":
        (sdk / "src/device/gm2gm/shmemi_device_cc.h").unlink()
    elif fault == "missing_public_header":
        (sdk / "include/host/shmem_host_def.h").unlink()
    before_manifest = manifest.read_bytes() if manifest.is_file() else None
    env = {
        "DEEPEP_NPU_ARCH": "Ascend950",
        "ASCEND_HOME_PATH": str(cann),
        "SHMEM_ROOT": str(sdk),
    }
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter("always")
        result = build.build_configuration(env)
    assert not messages
    assert "sha256" not in result["shmem"] and "shmem_device_headers" not in result
    assert result["shmem"]["revision"] == ("unknown" if fault in ("missing", "json", "list") else identity["revision"])
    assert result["shmem"]["build"] == (None if fault in ("missing", "json", "list") else identity["build"])
    assert manifest.exists() == (fault != "missing")
    assert (manifest.read_bytes() if manifest.is_file() else None) == before_manifest


@pytest.mark.parametrize('value', [None, False, True, 0, 1, 2, -1, '0', 'auto', 0.0, [], {'mode': [1, 2]}])
def test_optional_sdk_notes_are_preserved_without_hashing_any_sdk_file(build,tmp_path,monkeypatch,value):
    sdk=tmp_path/'sdk';sdk.mkdir()
    metadata=dict(revision='provider-revision',build=dict(hns_1825_pi_on_chip=value,unknown_field='preserved'),
                  sha256={'../outside':'untrusted-path','lib/libshmem.so':'obsolete'})
    manifest=sdk/'deepep-sdk.json';manifest.write_text(json.dumps(metadata))
    original=manifest.read_bytes()
    monkeypatch.setattr(Path,'read_bytes',lambda path:pytest.fail('SDK input files must not be hashed'))
    result=build.shmem_metadata(sdk)
    assert result==dict(revision=metadata['revision'],build=metadata['build'])
    assert type(result['build']['hns_1825_pi_on_chip']) is type(value)
    assert manifest.read_text().encode()==original


def test_unavailable_sdk_notes_are_optional(build,tmp_path,monkeypatch):
    assert build.shmem_metadata(tmp_path)==dict(revision='unknown',build=None)
    monkeypatch.setattr(Path,'read_text',lambda *a,**k:(_ for _ in ()).throw(PermissionError('unreadable optional notes')))
    assert build.shmem_metadata(tmp_path)==dict(revision='unknown',build=None)


def test_metadata_no_sdk_or_framework(tmp_path):
    # Query in a copy so egg metadata does not alter the checkout.
    import shutil

    for name in ("setup.py", "build_support.py", "VERSION", "scripts/clean_build_artifacts.py",
                 "scripts/ll_dispatch_implementation.py"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, tmp_path / name)
    result = subprocess.run(
        [sys.executable, "setup.py", "--name"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ascend-deepep"


@pytest.mark.parametrize(
    "members,error",
    [
        ({}, "native extension"),
        ({"deep_ep/_C.fake.so": b"test"}, "build identity"),
        (
            {
                "deep_ep/_C.fake.so": b"test",
                "deep_ep/_build_info.json": b"{}",
                "tests/x.py": b"",
            },
            "Unexpected",
        ),
    ],
)
def test_wheel_rejects_missing_or_unexpected_contents(tmp_path, members, error):
    validator = load(ROOT / "scripts/validate_package.py")
    wheel = tmp_path / "test.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, value in members.items():
            archive.writestr(name, value)
    with pytest.raises(RuntimeError, match=error):
        validator.inspect_wheel(wheel)


def test_lock_declares_operators_without_claiming_device_qualification():
    lock = json.loads((ROOT / "dependencies.lock.json").read_text())
    assert lock["device_kernels"] == [
        "full-notify-metadata",
        "expanded-dispatch-urma-mte",
        "expanded-combine-urma-mte",
        "expanded-dispatch-low-latency",
        "expanded-combine-low-latency",
    ]
    assert len(lock["shmem"]["revision"]) == 40
    assert len(lock["shmem"]["header_sha256_lf"]) == 6


@pytest.mark.parametrize(
    "capability",
    [
        "elastic-notify-metadata-v6",
        "elastic-notify-metadata-v7",
        "elastic-dispatch-expanded-v1",
        "elastic-combine-expanded-v3",
    ],
)
def test_elastic_wheel_requires_device_library(tmp_path, capability):
    validator = load(ROOT / "scripts/validate_package.py")
    operations = ["ElasticBuffer.dispatch", "ElasticBuffer.combine"]
    for device_library in (False, True):
        wheel = tmp_path / f"elastic-{device_library}.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr("deep_ep/_C.fake.so", b"test")
            archive.writestr("deep_ep/deepseek-deepep.LICENSE", b"test")
            archive.writestr(
                "deep_ep/_build_info.json",
                json.dumps(
                    {
                        "communication_operations": operations,
                        "capabilities": [capability],
                    }
                ),
            )
            if device_library:
                archive.writestr("deep_ep/libdeepep_kernels.so", b"test")
        if device_library:
            assert validator.inspect_wheel(wheel)["communication_operations"] == operations
        else:
            with pytest.raises(RuntimeError, match="Device library"):
                validator.inspect_wheel(wheel)


@pytest.mark.parametrize(
    "missing", [None, "deep_ep/libdeepep_kernels.so", "deep_ep_experimental/megamoe/libmegamoe_kernel.so"]
)
def test_combined_elastic_megamoe_wheel_requires_both_payloads(tmp_path, missing):
    validator = load(ROOT / "scripts/validate_package.py")
    info = {
        "megamoe": {"enabled": True, "timer": True},
        "capabilities": ["elastic-dispatch-expanded-v1", "elastic-combine-expanded-v3"],
        "communication_operations": ["ElasticBuffer.dispatch", "ElasticBuffer.combine"],
    }
    members = {
        "deep_ep/_C.fake.so",
        "deep_ep/libdeepep_kernels.so",
        "deep_ep/deepseek-deepep.LICENSE",
        "deep_ep_experimental/megamoe/api.py",
        "deep_ep_experimental/megamoe/benchmark.py",
        "deep_ep_experimental/megamoe/configs/smoke.json",
        "deep_ep_experimental/megamoe/_C.fake.so",
        "deep_ep_experimental/megamoe/libmegamoe_kernel.so",
        "deep_ep_experimental/megamoe/cann.LICENSE",
    }
    wheel = tmp_path / "combined.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("deep_ep/_build_info.json", json.dumps(info))
        for name in sorted(members - {missing}):
            archive.writestr(name, b"fixture")
    if missing is None:
        assert validator.inspect_wheel(wheel) == info
    else:
        with pytest.raises(RuntimeError, match="Device library|Incomplete MegaMoE"):
            validator.inspect_wheel(wheel)


def test_native_product_identity_names_toolchain_configuration(build):
    configuration = {
        "source_sha256": "before",
        "source_version": "1",
        "build_type": "Release",
        "ASCEND_HOME_PATH": "/sdk/cann",
        "SHMEM_ROOT": "/sdk/shmem",
        "arch": "Ascend950",
        "compiler_sha256": "host-1",
        "device_compiler_sha256": "device-1",
        "python_executable": "/python/bin/python",
        "build_environment": {"CXXFLAGS": ""},
    }
    key = build.build_cache_key(configuration)
    assert build.build_cache_key({**configuration, "source_sha256": "after", "source_version": "2"}) == key
    for name, value in (
        ("build_type", "RelWithDebInfo"),
        ("ASCEND_HOME_PATH", "/sdk/new-cann"),
        ("SHMEM_ROOT", "/sdk/new-shmem"),
        ("compiler_sha256", "host-2"),
        ("device_compiler_sha256", "device-2"),
        ("python_executable", "/new-python/bin/python"),
        ("build_environment", {"CXXFLAGS": "-g"}),
    ):
        assert build.build_cache_key({**configuration, name: value}) != key


@pytest.mark.parametrize("beta,timer", [(False, False), (True, False), (True, True)])
def test_packaging_uses_cmake_capabilities_and_removes_old_library(build, monkeypatch, tmp_path, beta, timer):
    from setuptools import Distribution, Extension

    # This CPU-only test mocks the build tools, including their discovery paths.
    pybind11_cmake_dir = str(tmp_path / "pybind11 cmake")
    pybind11 = ModuleType("pybind11")
    pybind11.get_cmake_dir = lambda: pybind11_cmake_dir
    monkeypatch.setitem(sys.modules, "pybind11", pybind11)

    configuration = {
        "megamoe": {"enabled": beta, "timer": timer},
        "CATLASS_ROOT": "/private/catlass" if beta else "",
        "source_sha256": "current",
        "source_version": "1",
        "cmake_path": "/tools/cmake",
        "compiler_path": "/tools/c++",
        "ASCEND_HOME_PATH": "/sdk/cann",
        "SHMEM_ROOT": "/sdk/shmem",
        "build_type": "RelWithDebInfo",
        "python_executable": "/private/python",
        "build_environment": {"CXXFLAGS": "-I/private/include"},
    }
    monkeypatch.setattr(build, "build_configuration", lambda: configuration)
    monkeypatch.setattr(build, "ROOT", tmp_path)
    monkeypatch.setenv("CMAKE_BUILD_PARALLEL_LEVEL", "6")
    ext = Extension("deep_ep._C", sources=[])
    command = build.CMakeBuild(Distribution({"ext_modules": [ext]}))
    command.ensure_finalized()
    command.build_temp = str(tmp_path / "build/temp")
    command.build_lib = str(tmp_path / "lib")
    destination = Path(command.get_ext_fullpath(ext.name)).parent
    destination.mkdir(parents=True)
    (destination / "libdeepep_notify.so").write_bytes(b"old")
    obsolete_modules = [
        f"_{op}{suffix}" for op in ("notify", "dispatch", "combine") for suffix in (".py", "_layout.py")
    ]
    for name in obsolete_modules:
        (destination / name).write_text("old module\n")
    calls = []
    capabilities = {"capabilities": ["test-built-target"], "communication_operations": []}
    native_directory=Path(command.build_temp)/build.build_cache_key(configuration)
    native_directory.mkdir(parents=True)
    (native_directory/'stale.o').write_text('stale native object')
    sibling=tmp_path/'build/other-variant';sibling.mkdir()
    (sibling/'keep.o').write_text('another build variant')
    history=tmp_path/'diagnostic-runs/old/wheel.whl';history.parent.mkdir(parents=True)
    history.write_text('previous wheel')

    def run(argv):
        calls.append(list(map(str, argv)))
        if "-B" in argv:
            directory = Path(argv[argv.index("-B") + 1])
            assert not (directory/'stale.o').exists()
            (directory / "compiler-info.json").write_text('{"id":"test","version":"1"}')
            (directory / "capabilities.json").write_text(json.dumps(capabilities))
        if "--install" in argv:
            (destination / "libdeepep_kernels.so").write_bytes(b"new")

    monkeypatch.setattr(build, "checked_run", run)
    command.build_extension(ext)
    assert "-DCMAKE_BUILD_TYPE=RelWithDebInfo" in calls[0]
    assert "-DDEEPEP_DEVICE_JOBS=6" in calls[0]
    assert f"-DDEEPEP_BUILD_MEGAMOE={'ON' if beta else 'OFF'}" in calls[0]
    assert f"-DDEEPEP_MEGAMOE_TIMER={'ON' if timer else 'OFF'}" in calls[0]
    assert f"-Dpybind11_DIR={pybind11_cmake_dir}" in calls[0]
    assert calls[1][-2:] == ["--parallel", "6"]
    assert not (destination / "libdeepep_notify.so").exists()
    assert all(not (destination / name).exists() for name in obsolete_modules)
    info = json.loads((destination / "_build_info.json").read_text())
    assert info["capabilities"] == capabilities["capabilities"]
    assert info["source_sha256"] == "current"
    assert info["megamoe"] == {"enabled": beta, "timer": timer}
    assert (sibling/'keep.o').read_text()=='another build variant'
    assert history.read_text()=='previous wheel'
    assert "CATLASS_ROOT" not in info
    assert not {"cmake_path", "compiler_path", "python_executable", "build_environment"}.intersection(info)
    # An identical second invocation still removes its old native products
    # and enters configure/build/install; matching configuration is not reuse.
    (native_directory/'stale.o').write_text('previous successful build')
    command.build_extension(ext)
    assert len(calls)==6 and calls[3]==calls[0]
    assert not (native_directory/'stale.o').exists()


@pytest.mark.parametrize('outside', [False,True])
def test_native_clean_boundary_and_compiler_errors_stop_the_build(build,monkeypatch,tmp_path,outside):
    from setuptools import Distribution, Extension
    pybind11=ModuleType('pybind11');pybind11.get_cmake_dir=lambda:'pybind11'
    monkeypatch.setitem(sys.modules,'pybind11',pybind11)
    root=tmp_path/'project';root.mkdir()
    monkeypatch.setattr(build,'ROOT',root)
    configuration=dict(build_type='Release',cmake_path='cmake',compiler_path='c++',
                       ASCEND_HOME_PATH='cann',SHMEM_ROOT='missing-sdk')
    monkeypatch.setattr(build,'build_configuration',lambda:configuration)
    ext=Extension('deep_ep._C',sources=[])
    command=build.CMakeBuild(Distribution({'ext_modules':[ext]}));command.ensure_finalized()
    command.build_temp=str(tmp_path/'external-build' if outside else root/'build/native')
    command.build_lib=str(root/'build/staged')
    native=Path(command.build_temp)/build.build_cache_key(configuration)
    native.mkdir(parents=True);marker=native/'keep.o';marker.write_text('existing')
    calls=[]
    def fail(arguments):
        calls.append(arguments)
        raise subprocess.CalledProcessError(1,arguments,stderr='ordinary compiler missing-header failure')
    monkeypatch.setattr(build,'checked_run',fail)
    if outside:
        with pytest.raises(ValueError,match='outside'):
            command.build_extension(ext)
        assert marker.read_text()=='existing' and not calls
    else:
        with pytest.raises(subprocess.CalledProcessError,match='non-zero exit status'):
            command.build_extension(ext)
        assert not marker.exists() and len(calls)==1


def test_cmake_keeps_normal_library_resolution_without_shmem_attestation():
    source=(ROOT/'cmake/Dependencies.cmake').read_text()
    assert 'deepep-sdk.json' not in source and 'string(JSON' not in source
    assert 'hns_1825_pi_on_chip' not in source and 'header_sha256' not in source
    assert 'find_library(SHMEM_LIBRARY NAMES shmem' in source
    assert 'find_library(ASCENDCL_LIBRARY' in source


def test_compiler_symlink_keeps_invocation_name(build, monkeypatch, tmp_path):
    from types import SimpleNamespace

    cann, sdk = tmp_path / "cann", tmp_path / "sdk"
    files = {
        cann / "include/acl/acl.h": "",
        cann / "compiler/version.info": "Version=9.2.0",
        cann / "tools/bisheng_compiler/bin/bisheng": "device compiler",
        sdk / "include/host/shmem_host_def.h": "host declarations",
        sdk / "include/shmem.h": "public declarations",
        sdk / "src/device/gm2gm/shmemi_device_cc.h": "device declarations",
        sdk / "lib/libshmem.so": "installed library",
    }
    for path, data in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data)
    # Optional producer notes do not affect compiler-wrapper discovery.
    import hashlib

    lock = json.loads((ROOT / "dependencies.lock.json").read_text())
    lock["shmem"]["header_sha256_lf"] = {
        "host/shmem_host_def.h": hashlib.sha256(files[sdk / "include/host/shmem_host_def.h"].encode()).hexdigest()
    }
    source = tmp_path / "source"
    source.mkdir()
    (source / "dependencies.lock.json").write_text(json.dumps(lock))
    (source / "VERSION").write_text((ROOT / "VERSION").read_text())
    monkeypatch.setattr(build, "ROOT", source)
    manifest = {
        "revision": lock["shmem"]["revision"],
        "build": {"cann_version": "test", "cann_timestamp": "test", "target": "Ascend950", "source_patches": []},
        "sha256": {
            str(path.relative_to(sdk)): hashlib.sha256(data.encode()).hexdigest()
            for path, data in files.items()
            if path.is_relative_to(sdk)
        },
    }
    (sdk / "deepep-sdk.json").write_text(json.dumps(manifest))
    tools = tmp_path / "bin"
    tools.mkdir()
    wrapper = tools / "ccache"
    wrapper.write_text('#!/bin/sh\ncase "$0" in */c++) echo "C++ compiler";; *) exit 1;; esac\n')
    wrapper.chmod(0o755)
    compiler = tools / "c++"
    compiler.symlink_to(wrapper)
    cmake = tools / "cmake"
    cmake.write_text('#!/bin/sh\necho "cmake version test"\n')
    cmake.chmod(0o755)
    lock = json.loads((ROOT / "dependencies.lock.json").read_text())
    monkeypatch.setitem(sys.modules, "pybind11", SimpleNamespace(__version__=lock["python_build"]["pybind11"]))
    monkeypatch.setattr(build, "source_fingerprint", lambda: "test sources")
    config = build.build_configuration(
        {
            "DEEPEP_NPU_ARCH": "Ascend950",
            "ASCEND_HOME_PATH": str(cann),
            "SHMEM_ROOT": str(sdk),
            "PATH": str(tools),
        }
    )
    assert config["compiler_path"] == str(compiler)
    # The exact path passed to CMake must still dispatch as a C++ compiler.
    assert subprocess.check_output([config["compiler_path"], "--version"], text=True).strip() == "C++ compiler"


@pytest.mark.parametrize(
    "available,expected",
    [
        (["c++", "g++", "clang++"], "c++"),
        (["g++", "clang++"], "g++"),
        (["clang++"], "clang++"),
        (["bisheng++", "bisheng"], "bisheng++"),
        (["bisheng"], "bisheng"),
    ],
)
def test_auto_host_compiler_order(build, tmp_path, available, expected):
    for name in available:
        path = tmp_path / name
        path.write_text('#!/bin/sh\necho "test C++ compiler"\n')
        path.chmod(0o755)
    compiler, _ = build.find_host_compiler({"PATH": str(tmp_path)})
    assert compiler == str(tmp_path / expected)


def test_host_compiler_fallback_and_explicit_override(build, tmp_path):
    broken, good = tmp_path / "c++", tmp_path / "clang++"
    broken.write_text("#!/bin/sh\nexit 1\n")
    good.write_text('#!/bin/sh\necho "test clang"\n')
    broken.chmod(0o755)
    good.chmod(0o755)
    env = {"PATH": str(tmp_path)}
    assert build.find_host_compiler(env)[0] == str(good)
    with pytest.raises(RuntimeError, match="Explicit CXX"):
        build.find_host_compiler({**env, "CXX": str(broken)})
    assert build.find_host_compiler({**env, "CXX": str(good)})[0] == str(good)
    cache = tmp_path / "ccache"
    cache.write_text('#!/bin/sh\necho "ccache version"\n')
    cache.chmod(0o755)
    with pytest.raises(RuntimeError, match="standalone cache launcher"):
        build.find_host_compiler({**env, "CXX": str(cache)})


def test_host_compiler_cann_path_and_missing(build, tmp_path):
    env = {"PATH": str(tmp_path), "ASCEND_HOME_PATH": str(tmp_path / "cann")}
    with pytest.raises(RuntimeError, match="No runnable Host C[+][+] compiler"):
        build.find_host_compiler(env)
    compiler = tmp_path / "cann/tools/bisheng_compiler/bin/bisheng"
    compiler.parent.mkdir(parents=True)
    compiler.write_text('#!/bin/sh\necho "bisheng"\n')
    compiler.chmod(0o755)
    assert build.find_host_compiler(env)[0] == str(compiler)
