"""Diagnostic-profile validation and native-cache isolation without SDK/device use."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROFILE_NAMES = ("sparse",)
LEGACY_PROFILES = ("full", "overview", "prepare", "prefix", "send", "recv")


@pytest.fixture
def build(monkeypatch):
    spec = importlib.util.spec_from_file_location("ll_profile_build", ROOT / "build_support.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.platform, "system", lambda: "Linux")
    return module


@pytest.mark.parametrize("timer", ["ON", "OFF"])
@pytest.mark.parametrize("profile", PROFILE_NAMES)
def test_profile_is_validated_and_off_is_normalized(build, timer, profile):
    assert build.ll_timer_profile({"DEEPEP_LL_STAGE_TIMER": timer,
                                   "DEEPEP_LL_TIMER_PROFILE": profile}) == "sparse"
    assert build.ll_timer_profile({}) == "sparse"


@pytest.mark.parametrize("timer", ["ON", "OFF"])
@pytest.mark.parametrize("profile", ["", "FULL", "disabled", "prefix;recv", "0", *LEGACY_PROFILES])
def test_invalid_profile_fails_before_sdk_discovery(build, timer, profile):
    with pytest.raises(ValueError, match="DEEPEP_LL_TIMER_PROFILE"):
        build.build_configuration(dict(DEEPEP_NPU_ARCH="Ascend950", DEEPEP_LL_STAGE_TIMER=timer,
                                       DEEPEP_LL_TIMER_PROFILE=profile))


def test_profile_changes_on_cache_but_not_off_cache(build):
    def key(timer, profile):
        return build.build_cache_key(dict(ll_stage_timer=timer, ll_timer_profile=profile))
    assert len({key(True, p) for p in (*LEGACY_PROFILES, *PROFILE_NAMES)}) == 7
    assert len({key(False, p) for p in (*LEGACY_PROFILES, *PROFILE_NAMES)}) == 1
    assert key(True, "sparse") != key(False, "sparse")


@pytest.mark.parametrize("profile", PROFILE_NAMES)
def test_profile_reaches_cmake_and_installed_build_identity(build, monkeypatch, tmp_path, profile):
    from setuptools import Distribution, Extension
    monkeypatch.setitem(sys.modules, "pybind11", SimpleNamespace(get_cmake_dir=lambda: "pybind11-cmake"))
    configuration = dict(ll_stage_timer=True, ll_timer_profile=profile, build_type="Release",
                         cmake_path="cmake", compiler_path="c++", ASCEND_HOME_PATH="cann", SHMEM_ROOT="shmem")
    monkeypatch.setattr(build, "build_configuration", lambda: configuration)
    extension = Extension("deep_ep._C", sources=[])
    command = build.CMakeBuild(Distribution({"ext_modules": [extension]}))
    command.ensure_finalized()
    command.build_temp = str(tmp_path / "temporary native build")
    command.build_lib = str(tmp_path / "staged package")
    destination = Path(command.get_ext_fullpath(extension.name)).parent
    destination.mkdir(parents=True)
    calls = []

    def run(arguments):
        calls.append(list(map(str, arguments)))
        if "-B" in arguments:
            directory = Path(arguments[arguments.index("-B") + 1])
            (directory / "compiler-info.json").write_text('{"id":"mock","version":"1"}')
            (directory / "capabilities.json").write_text('{"capabilities":[],"communication_operations":[]}')

    monkeypatch.setattr(build, "checked_run", run)
    command.build_extension(extension)
    assert f"-DDEEPEP_LL_TIMER_PROFILE={profile}" in calls[0]
    assert "-DDEEPEP_LL_STAGE_TIMER=ON" in calls[0]
    assert Path(calls[0][calls[0].index("-B") + 1]).name == build.build_cache_key(configuration)
    assert json.loads((destination / "_build_info.json").read_text())["ll_timer_profile"] == profile


@pytest.mark.parametrize("timer", ["ON", "OFF"])
@pytest.mark.parametrize("profile", PROFILE_NAMES)
def test_real_cmake_profile_mapping_matches_python(build, tmp_path, timer, profile):
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake unavailable")
    output = tmp_path / "profile.txt"
    script = tmp_path / "profile.cmake"
    script.write_text('include("' + (ROOT / "cmake/LlTimerProfile.cmake").as_posix() + '")\n'
                      'file(WRITE "' + output.as_posix() + '" "${DEEPEP_LL_TIMER_PROFILE};${DEEPEP_LL_TIMER_PROFILE_ID}")\n')
    subprocess.run([cmake, f"-DDEEPEP_LL_STAGE_TIMER={timer}", f"-DDEEPEP_LL_TIMER_PROFILE={profile}",
                    "-P", str(script)], check=True, capture_output=True, text=True, timeout=30)
    expected = build.ll_timer_profile(dict(DEEPEP_LL_STAGE_TIMER=timer, DEEPEP_LL_TIMER_PROFILE=profile))
    assert output.read_text() == f"{expected};6"


@pytest.mark.parametrize("timer", ["ON", "OFF"])
@pytest.mark.parametrize("profile", ["invalid", *LEGACY_PROFILES])
def test_real_cmake_rejects_unknown_or_legacy_profile(tmp_path, timer, profile):
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake unavailable")
    script = tmp_path / "invalid.cmake"
    script.write_text('include("' + (ROOT / "cmake/LlTimerProfile.cmake").as_posix() + '")\n')
    result = subprocess.run([cmake, f"-DDEEPEP_LL_STAGE_TIMER={timer}", f"-DDEEPEP_LL_TIMER_PROFILE={profile}",
                             "-P", str(script)], capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "DEEPEP_LL_TIMER_PROFILE must be sparse" in result.stderr
