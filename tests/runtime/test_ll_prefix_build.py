"""Compile-time P is validated before SDK discovery and separates native builds."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def build(monkeypatch):
    spec = importlib.util.spec_from_file_location('ll_prefix_build', ROOT / 'build_support.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.platform, 'system', lambda: 'Linux')
    return module


@pytest.mark.parametrize('value', ['0', '1', '4', '6', '8', '26'])
def test_valid_prefix_and_default(build, value):
    assert build.ll_small_b_prefix_cores({}, 26) == 0
    assert build.ll_small_b_prefix_cores({'DEEPEP_LL_SMALL_B_PREFIX_CORES': value}, 26) == int(value)


@pytest.mark.parametrize('value', ['', '-1', '27', '56', '1.0', '01', '+1', ' 1', '1 ', '１'])
def test_invalid_prefix_rejected_before_sdk_discovery(build, value):
    with pytest.raises(ValueError, match='DEEPEP_LL_SMALL_B_PREFIX_CORES'):
        build.build_configuration(dict(DEEPEP_NPU_ARCH='Ascend950', DEEPEP_LL_SEND_CORES='30',
                                       DEEPEP_LL_RECV_CORES='26', DEEPEP_LL_SMALL_B_PREFIX_CORES=value))


def test_prefix_separates_on_and_off_native_cache(build):
    keys = {build.build_cache_key(dict(ll_stage_timer=timer, ll_send_cores=30, ll_recv_cores=26,
                                      ll_small_b_prefix_cores=prefix))
            for timer in (False, True) for prefix in (0, 1, 6, 26)}
    assert len(keys) == 8


@pytest.mark.parametrize('prefix', [0, 1, 6, 26])
def test_prefix_reaches_cmake_and_installed_identity(build, monkeypatch, tmp_path, prefix):
    from setuptools import Distribution, Extension
    monkeypatch.setitem(sys.modules, 'pybind11', SimpleNamespace(get_cmake_dir=lambda: 'pybind11-cmake'))
    configuration = dict(ll_stage_timer=True, ll_timer_profile='sparse', build_type='Release',
                         ll_send_cores=30, ll_recv_cores=26, ll_small_b_prefix_cores=prefix,
                         cmake_path='cmake', compiler_path='c++', ASCEND_HOME_PATH='cann', SHMEM_ROOT='shmem')
    monkeypatch.setattr(build, 'build_configuration', lambda: configuration)
    extension = Extension('deep_ep._C', sources=[])
    command = build.CMakeBuild(Distribution({'ext_modules': [extension]}))
    command.ensure_finalized()
    command.build_temp = str(tmp_path / 'temporary native build')
    command.build_lib = str(tmp_path / 'staged package')
    destination = Path(command.get_ext_fullpath(extension.name)).parent
    destination.mkdir(parents=True)
    calls = []

    def run(arguments):
        calls.append(list(map(str, arguments)))
        if '-B' in arguments:
            directory = Path(arguments[arguments.index('-B') + 1])
            (directory / 'compiler-info.json').write_text('{"id":"mock","version":"1"}')
            (directory / 'capabilities.json').write_text('{"capabilities":[],"communication_operations":[]}')

    monkeypatch.setattr(build, 'checked_run', run)
    command.build_extension(extension)
    assert f'-DDEEPEP_LL_SMALL_B_PREFIX_CORES={prefix}' in calls[0]
    assert Path(calls[0][calls[0].index('-B') + 1]).name == build.build_cache_key(configuration)
    assert json.loads((destination / '_build_info.json').read_text())['ll_small_b_prefix_cores'] == prefix
