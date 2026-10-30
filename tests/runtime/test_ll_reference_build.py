"""Current reference-only build identity and configuration forwarding."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[2]


@pytest.fixture
def build(monkeypatch):
    spec=importlib.util.spec_from_file_location('reference_build_contract',ROOT/'build_support.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module.platform,'system',lambda:'Linux')
    return module


@pytest.mark.parametrize('settings',[dict(DEEPEP_LL_STAGE_TIMER='ON'),dict(DEEPEP_LL_TIMER_PROFILE='sparse'),dict(DEEPEP_LL_TIMER_PROFILE='full')])
def test_legacy_build_requests_fail_before_sdk_discovery(build,settings):
    with pytest.raises(RuntimeError,match='removed'):
        build.build_configuration(dict(DEEPEP_NPU_ARCH='Ascend950',**settings))


def test_reference_on_off_and_algorithms_have_separate_cache_identity(build):
    keys={build.build_cache_key(dict(source_sha256='same',ll_reference_profile=enabled,ll_reference_profile_schema=7,
                                     ll_dispatch_impl=choice)) for enabled in (False,True)
          for choice in ('current','cb59','cb59-stream','cb59-pair','cb59-small')}
    assert len(keys)==10


@pytest.mark.parametrize('enabled',[False,True])
def test_cmake_and_installed_metadata_receive_reference_contract(build,monkeypatch,tmp_path,enabled):
    from setuptools import Distribution,Extension
    monkeypatch.setitem(sys.modules,'pybind11',SimpleNamespace(get_cmake_dir=lambda:'pybind-cmake'))
    config=dict(ll_reference_profile=enabled,ll_reference_profile_schema=7,ll_instrumentation_source='reference-mte7',
                dispatch_launch_bytes=144,combine_launch_bytes=88,build_type='Release',cmake_path='cmake',
                compiler_path='c++',ASCEND_HOME_PATH='cann',SHMEM_ROOT='shmem')
    monkeypatch.setattr(build,'build_configuration',lambda:config);monkeypatch.setattr(build,'ROOT',tmp_path)
    ext=Extension('deep_ep._C',sources=[])
    command=build.CMakeBuild(Distribution({'ext_modules':[ext]}));command.ensure_finalized()
    command.build_temp=str(tmp_path/'build/native');command.build_lib=str(tmp_path/'package')
    output=Path(command.get_ext_fullpath(ext.name)).parent;output.mkdir(parents=True)
    (output/'utils').mkdir()
    (output/'utils/ll_stage_timer.py').write_text('retired fixture')
    (output/'utils/event.py').write_text('keep unrelated module')
    calls=[]
    def run(argv):
        calls.append(list(map(str,argv)))
        if '-B' in argv:
            directory=Path(argv[argv.index('-B')+1])
            (directory/'compiler-info.json').write_text('{"id":"mock","version":"1"}')
            (directory/'capabilities.json').write_text('{}')
    monkeypatch.setattr(build,'checked_run',run)
    command.build_extension(ext)
    assert not (output/'utils/ll_stage_timer.py').exists()
    assert (output/'utils/event.py').read_text()=='keep unrelated module'
    assert '-DDEEPEP_LL_REFERENCE_PROFILE='+('ON' if enabled else 'OFF') in calls[0]
    assert not any('STAGE_TIMER' in flag or 'TIMER_PROFILE' in flag for flag in calls[0])
    info=json.loads((output/'_build_info.json').read_text())
    assert all(info[key]==config[key] for key in ('ll_reference_profile','ll_reference_profile_schema','ll_instrumentation_source','dispatch_launch_bytes','combine_launch_bytes'))
    cmake=(ROOT/'CMakeLists.txt').read_text(encoding='utf-8')
    assert 'target_compile_definitions(_C PRIVATE DEEPEP_LL_REFERENCE_PROFILE=1)' in cmake
    assert '-DDEEPEP_LL_REFERENCE_PROFILE=${DEEPEP_LL_REFERENCE_PROFILE}' in (ROOT/'cmake/Device.cmake').read_text()
