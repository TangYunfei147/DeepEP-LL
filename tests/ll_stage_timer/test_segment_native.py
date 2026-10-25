"""Actual current sparse recorder OFF/ON CPU mocks; legacy formats stay offline."""
import importlib.util
from pathlib import Path
import pytest


@pytest.mark.parametrize('enabled',[False,True])
def test_actual_sparse_recorder_is_bounded_and_delays_gm_export(tmp_path,enabled):
    path=Path(__file__).with_name('test_sparse_control_native.py')
    spec=importlib.util.spec_from_file_location('sparse_recorder_native',path)
    helper=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.run_sparse_recorder_mock(tmp_path,enabled)
