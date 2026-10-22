"""Exercise the ST preflight without importing Torch or starting an NPU."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


entry = load('schema_entry', 'tests/elastic/test_ep_low_latency.py')
timer = load('schema_timer', 'deep_ep/utils/ll_stage_timer.py')


def test_current_timer_schema_passes_st_preflight():
    entry.validate_stage_timer_capability(dict(stage_timer=True, stage_timer_schema=timer.SCHEMA))


@pytest.mark.parametrize('schema', [7, 9, 10, 11, 12, None])
def test_stale_or_unknown_schema_is_rejected(schema):
    with pytest.raises(RuntimeError, match=f'timer schema 13; got {schema}'):
        entry.validate_stage_timer_capability(dict(stage_timer=True, stage_timer_schema=schema))


def test_timer_off_wheel_is_still_rejected():
    with pytest.raises(RuntimeError, match='DEEPEP_LL_STAGE_TIMER=ON'):
        entry.validate_stage_timer_capability(dict(stage_timer=False, stage_timer_schema=timer.SCHEMA))


def test_worker_uses_tested_preflight():
    import inspect
    assert 'validate_stage_timer_capability(capability)' in inspect.getsource(entry.worker)
