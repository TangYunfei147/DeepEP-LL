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


def capability(**changes):
    result=dict(stage_timer=True, stage_timer_schema=16, timer_profile='sparse', timer_profile_id=6,
                stage_timer_slots=8, stage_timer_fields=8, stage_timer_recording='sparse-milestones',
                stage_timer_points=list(timer.SPARSE_POINTS))
    result.update(changes)
    return result


def test_current_timer_schema_passes_st_preflight():
    entry.validate_stage_timer_capability(capability())


@pytest.mark.parametrize('schema', [7, 9, 10, 11, 12, 13, 14, 15, 17, None])
def test_stale_or_unknown_schema_is_rejected(schema):
    with pytest.raises(RuntimeError, match=f'timer schema 16; got {schema}'):
        entry.validate_stage_timer_capability(capability(stage_timer_schema=schema))


def test_timer_off_wheel_is_still_rejected():
    with pytest.raises(RuntimeError, match='DEEPEP_LL_STAGE_TIMER=ON'):
        entry.validate_stage_timer_capability(capability(stage_timer=False))


def test_worker_uses_tested_preflight():
    import inspect
    assert 'validate_stage_timer_capability(capability)' in inspect.getsource(entry.worker)


@pytest.mark.parametrize('changes', [dict(timer_profile='full'), dict(timer_profile_id=0),
    dict(stage_timer_slots=88), dict(stage_timer_fields=4), dict(stage_timer_recording='segments')])
def test_stale_sparse_layout_is_rejected(changes):
    with pytest.raises(RuntimeError, match='sparse profile 6'):
        entry.validate_stage_timer_capability(capability(**changes))


def test_point_names_and_order_are_checked():
    points=list(timer.SPARSE_POINTS)
    points[0],points[1]=points[1],points[0]
    with pytest.raises(RuntimeError, match='names/order'):
        entry.validate_stage_timer_capability(capability(stage_timer_points=points))


def test_sparse_layout_helper_uses_same_contract():
    entry.validate_sparse_timer_layout(dict(schema=16,slots=8,fields=8,recording='sparse-milestones',
        timer_profile='sparse',timer_profile_id=6,points=list(timer.SPARSE_POINTS)))
