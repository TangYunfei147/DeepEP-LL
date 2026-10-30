"""Reference enablement isolates build identity without changing device launch work."""
import importlib.util
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[2]


def module():
    spec=importlib.util.spec_from_file_location('reference_build_support',ROOT/'build_support.py')
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def test_reference_flag_separates_build_caches_without_source_shortcuts():
    build=module()
    plain=dict(source_sha256='a',source_version='v1',ll_reference_profile=False,ll_send_cores=30,ll_recv_cores=26)
    assert build.build_cache_key(plain)==build.build_cache_key(dict(plain,source_sha256='b',source_version='v2'))
    assert build.build_cache_key(plain)!=build.build_cache_key(dict(plain,ll_reference_profile=True))
    source=(ROOT/'cmake/Device.cmake').read_text()
    assert 'add_custom_target(device_build' in source and '--build "${DEEPEP_DEVICE_BINARY_DIR}"' in source


def test_legacy_build_switch_is_rejected_before_sdk_preparation(monkeypatch):
    build=module();monkeypatch.setattr(build.platform,'system',lambda:'Linux')
    with pytest.raises(RuntimeError,match='Legacy LL tracing was removed'):
        build.build_configuration(dict(DEEPEP_NPU_ARCH='Ascend950',DEEPEP_LL_STAGE_TIMER='ON'))


def test_only_reference_flag_reaches_native_configuration():
    source=(ROOT/'build_support.py').read_text()
    assert 'result["ll_reference_profile"] = enabled("DEEPEP_LL_REFERENCE_PROFILE", env)' in source
    assert 'result["ll_stage_timer"]' not in source and '"-DDEEPEP_LL_STAGE_TIMER=' not in source
    child=(ROOT/'cmake/Device.cmake').read_text()
    assert '"-DDEEPEP_LL_REFERENCE_PROFILE=${DEEPEP_LL_REFERENCE_PROFILE}"' in child
    assert 'DEEPEP_LL_STAGE_TIMER' not in child
    device=(ROOT/'cmake/device/CMakeLists.txt').read_text()
    assert 'DEEPEP_LL_REFERENCE_PROFILE=1' in device
    assert 'DEEPEP_LL_STAGE_TIMER=1' not in device
