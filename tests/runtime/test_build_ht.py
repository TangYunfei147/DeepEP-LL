"""Actual CMake inventory and native preprocessor checks, without CANN/NPU."""
import importlib.util
from pathlib import Path
import re
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('setting', ('default', 'ON', 'OFF'))
def test_cmake_host_device_and_capability_inventory(tmp_path, setting):
    cmake = shutil.which('cmake')
    if not cmake:
        pytest.skip('CMake is required to execute the production source inventory')
    script = tmp_path/'selection.cmake'
    result = tmp_path/'inventory.txt'
    prefix = '' if setting == 'default' else f'set(DEEPEP_BUILD_HT {setting} CACHE BOOL "fixture")\n'
    script.write_text(prefix + f'''cmake_minimum_required(VERSION 3.20)
include("{(ROOT/'cmake/OperatorSources.cmake').as_posix()}")
file(WRITE "{result.as_posix()}" "${{DEEPEP_CORE_HOST_SOURCES}}\\n${{DEEPEP_CORE_DEVICE_SOURCES}}\\n${{DEEPEP_CORE_CAPABILITIES}}\\n")
''', encoding='utf-8')
    subprocess.run([cmake, '-P', str(script)], check=True, capture_output=True, text=True)
    data = dict(zip(('host', 'device', 'capabilities'),
                    (line.split(';') for line in result.read_text(encoding='utf-8').splitlines())))
    ht_host = {'csrc/ops/notify.cpp', 'csrc/ops/dispatch.cpp', 'csrc/ops/combine.cpp'}
    ht_device = {'csrc/kernels/notify/notify.cpp', 'csrc/kernels/dispatch/dispatch.cpp',
                 'csrc/kernels/combine/combine.cpp', 'csrc/kernels/combine/reuse.cpp',
                 'csrc/kernels/combine/combine_ep8_mte.cpp'}
    assert len(data['host']) == len(set(data['host']))
    assert len(data['device']) == len(set(data['device']))
    assert len([path for path in data['device'] if '/variants/' in path]) == 7
    if setting == 'OFF':
        assert set(data['host']) == {'csrc/bindings/module.cpp', 'csrc/runtime/session.cpp', 'csrc/ops/low_latency.cpp'}
        assert len(data['device']) == 9 and not ht_device.intersection(data['device'])
        assert not any(any(word in cap for word in ('urma', 'notify', 'expanded', 'sparse-qp', 'mpi-bootstrap')) for cap in data['capabilities'])
    else:
        assert ht_host.issubset(data['host']) and ht_device.issubset(data['device'])
        assert 'runtime-urma' in data['capabilities'] and 'elastic-notify-metadata-v7' in data['capabilities']


@pytest.mark.parametrize('ht', (0, 1))
def test_native_preprocessor_excludes_ht_bindings(tmp_path, ht):
    compiler = shutil.which('g++') or shutil.which('clang++')
    if not compiler:
        pytest.skip('C++ preprocessor required; this is not a CANN compilation test')
    # The test processes the actual binding body/guards; unrelated SDK headers
    # are not needed to decide which public bindings survive preprocessing.
    source = (ROOT/'csrc/bindings/module.cpp').read_text(encoding='utf-8')
    source = re.sub(r'^\s*#include[^\n]*', '', source, flags=re.M)
    path = tmp_path/'bindings.cpp'; path.write_text(source, encoding='utf-8')
    result = subprocess.check_output([compiler, '-E', '-P', '-x', 'c++', f'-DDEEPEP_BUILD_HT={ht}', str(path)], text=True)
    assert 'bind_low_latency(module)' in result and 'create_low_latency_runtime' in result
    assert 'high_throughput_enabled' in result and f'bool({ht})' in result
    for item in ('bind_notify(module)', 'bind_dispatch(module)', 'bind_combine(module)',
                 'create_mpi_runtime', 'create_combine_runtime'):
        assert (item in result) == bool(ht)


def test_flag_validation_and_cache_key_are_separate_from_reference_profile():
    spec = importlib.util.spec_from_file_location('ht_build_support', ROOT/'build_support.py')
    build = importlib.util.module_from_spec(spec); spec.loader.exec_module(build)
    assert build.enabled('DEEPEP_BUILD_HT', {}, default='ON') is True
    assert build.enabled('DEEPEP_BUILD_HT', {'DEEPEP_BUILD_HT':'OFF'}, default='ON') is False
    with pytest.raises(ValueError):
        build.enabled('DEEPEP_BUILD_HT', {'DEEPEP_BUILD_HT':'maybe'}, default='ON')
    keys = {build.build_cache_key(dict(build_ht=ht, ll_reference_profile=profile)) for ht in (False, True) for profile in (False, True)}
    assert len(keys) == 4
    wrapper = (ROOT/'cmake/Device.cmake').read_text(encoding='utf-8')
    assert '"-DDEEPEP_BUILD_HT=${DEEPEP_BUILD_HT}"' in wrapper
    device = (ROOT/'cmake/device/CMakeLists.txt').read_text(encoding='utf-8')
    assert 'UNITY_BUILD OFF' in device and '--npu-arch=dav-3510' in device and ':-O2>' in device
