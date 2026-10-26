# Copyright (c) 2026, Lu Lu
# Modified by lishaoxun 2026

"""Native identity checks with local test doubles; no framework/driver loading."""

from types import SimpleNamespace
import warnings

import pytest

from deep_ep import _native


@pytest.mark.parametrize("fault", [None, "origin", "protocol", "missing", "load_failure"])
def test_native_loader_identity(monkeypatch, tmp_path, fault):
    module = SimpleNamespace(runtime_protocol=2 if fault == "protocol" else 1)
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
    if metadata is not None:
        (tmp_path/'_build_info.json').write_text(metadata)
    def forbidden():
        pytest.fail('SHMEM library hashing/path lookup must not happen during native import')
    native=SimpleNamespace(runtime_protocol=1,shmem_library_path=forbidden)
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
