# Copyright (c) 2026, Lu Lu
# Modified by lishaoxun 2026

"""Native identity checks with local test doubles; no framework/driver loading."""

from types import SimpleNamespace
import warnings
import json

import pytest

from deep_ep import _native


def project_identity():
    return dict(ll_instrumentation_source='reference-mte7',dispatch_launch_bytes=144,combine_launch_bytes=88,
                ll_reference_profile=False,ll_reference_profile_schema=7)


def native_identity():
    return dict(low_latency_instrumentation_source='reference-mte7',dispatch_launch_bytes=144,combine_launch_bytes=88,
                low_latency_reference_profile_enabled=False,low_latency_reference_profile_layout=lambda:dict(schema=7))


@pytest.mark.parametrize("fault", [None, "origin", "protocol", "missing", "load_failure"])
def test_native_loader_identity(monkeypatch, tmp_path, fault):
    module = SimpleNamespace(runtime_protocol=2 if fault == "protocol" else 1,**native_identity())
    (tmp_path/'_build_info.json').write_text(json.dumps(project_identity()))
    spec = SimpleNamespace(origin=str((tmp_path.parent if fault == "origin" else tmp_path) / "_C.so"))
    monkeypatch.setattr(_native, "__file__", str(tmp_path / "_native.py"))
    monkeypatch.setattr(
        _native.importlib.util,
        "find_spec",
        lambda name: None if fault == "missing" else spec,
    )
    def load(name):
        if fault == "load_failure":
            raise ImportError('native linker failure')
        return module
    monkeypatch.setattr(_native.importlib, "import_module", load)
    _native._load_native.cache_clear()
    try:
        if fault:
            with pytest.raises(ImportError):
                _native._load_native()
        else:
            assert _native._load_native() is module
    finally:
        _native._load_native.cache_clear()


@pytest.mark.parametrize('metadata', [None, 'invalid json', '{}',
    '{"shmem":{"sha256":{"lib/libshmem.so":"obsolete"},"build":{"hns_1825_pi_on_chip":{"vendor":[2,null]}}}}'])
def test_loader_does_not_read_or_validate_shmem_metadata_or_library(monkeypatch,tmp_path,metadata):
    info=project_identity()
    if metadata is not None:info['shmem']=metadata
    (tmp_path/'_build_info.json').write_text(json.dumps(info))
    # SHMEM metadata may be absent, opaque or invalid; only this project's own
    # source/ABI identity is required by the new package.
    (tmp_path/'deepep-sdk.json').write_text(metadata or 'invalid SDK metadata')
    def forbidden():
        pytest.fail('SHMEM library hashing/path lookup must not happen during native import')
    native=SimpleNamespace(runtime_protocol=1,shmem_library_path=forbidden,**native_identity())
    monkeypatch.setattr(_native,'__file__',str(tmp_path/'_native.py'))
    monkeypatch.setattr(_native.importlib.util,'find_spec',lambda name:SimpleNamespace(origin=str(tmp_path/'_C.so')))
    monkeypatch.setattr(_native.importlib,'import_module',lambda name:native)
    _native._load_native.cache_clear()
    try:
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter('always')
            assert _native._load_native() is native
        assert not recorded
    finally:
        _native._load_native.cache_clear()


@pytest.mark.parametrize('fault',['stale_module','schema','launch','native_boolean','missing_project_identity'])
def test_reference_package_rejects_mixed_or_retired_artifacts(monkeypatch,tmp_path,fault):
    info=project_identity();attributes=native_identity()
    if fault=='schema':info['ll_reference_profile_schema']=6
    elif fault=='launch':attributes['dispatch_launch_bytes']=136
    elif fault=='native_boolean':attributes['low_latency_reference_profile_enabled']=0
    elif fault=='stale_module':
        (tmp_path/'utils').mkdir();(tmp_path/'utils/ll_stage_timer.py').write_text('retired fixture')
    if fault!='missing_project_identity':(tmp_path/'_build_info.json').write_text(json.dumps(info))
    monkeypatch.setattr(_native,'__file__',str(tmp_path/'_native.py'))
    with pytest.raises(ImportError):_native._validate_ll_source_identity(SimpleNamespace(**attributes))
