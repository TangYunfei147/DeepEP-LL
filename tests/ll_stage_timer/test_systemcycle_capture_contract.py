"""Current capture preserves layout identity, one launch and deferred export."""
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


timer = load('systemcycle_capture_timer', 'deep_ep/utils/ll_stage_timer.py')
entry = load('systemcycle_capture_entry', 'tests/elastic/test_ep_low_latency.py')


def layout():
    return dict(schema=18, slots=16, fields=8, stages=list(timer.STAGES),
                points=list(timer.systemcycle.POINTS), recording='systemcycle-stages',
                timer_profile='systemcycle', timer_profile_id=9,
                cycle_metric_names=list(timer.systemcycle.METRIC_NAMES),
                cycle_metric_indices=list(timer.systemcycle.METRIC_INDICES),
                counter_indices=list(timer.systemcycle.COUNTER_INDICES),
                metric_word_offset=48, metric_record_words=2)


def capability():
    data = layout()
    return dict(stage_timer=True, stage_timer_schema=18, timer_profile='systemcycle',
                timer_profile_id=9, overlap_stage=3,
                **{'stage_timer_' + key: data[key] for key in (
                    'slots', 'fields', 'points', 'recording', 'cycle_metric_names',
                    'cycle_metric_indices', 'counter_indices', 'metric_word_offset', 'metric_record_words')})


def raw(operation='dispatch'):
    result = []
    for core in range(56):
        role = (3 if core < 30 else 2) if operation == 'dispatch' else (1 if core < 30 else 4)
        words = [0] * 128
        words[:8] = [timer.MAGIC, 18, 1 if operation == 'dispatch' else 2, 56, core,
                     1 | (30 << 8) | (role << 16) | (9 << 24), 0, 0]
        for name, tick in [('ENTRY', 100), ('INIT_DONE', 200), ('WORK_DONE', 1000)]:
            index = timer.systemcycle.POINTS.index(name)
            words[6] |= 1 << index
            words[8 + index] = tick + core
        metric = timer.systemcycle.METRIC_NAMES.index('COUNT_COMPUTE' if operation == 'dispatch' else 'COMBINE_SEND')
        words[7] |= 1 << (24 + metric)
        words[48 + 2 * metric:50 + 2 * metric] = [100, 2]
        result.append([words[i:i + 8] for i in range(0, 128, 8)])
    return result


def fake_native(monkeypatch, operation):
    for name in ('deep_ep', 'deep_ep.utils'):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    copies = []

    class Tensor:
        def __init__(self):
            self.ptr, self.device = 256, 'npu:0'

        def data_ptr(self):
            return self.ptr

        def narrow(self, dim, offset, count):
            self.ptr += offset * 8
            return self

        def view(self, *shape):
            self.shape = shape
            return self

        def cpu(self):
            copies.append(self.shape)
            return self

        def tolist(self):
            return raw(operation)

    native = SimpleNamespace(low_latency_stage_timer_enabled=True,
        low_latency_stage_timer_layout=layout, low_latency_overlap_stage=3,
        low_latency_core_config=lambda: dict(active_aiv=56, device=0, runtime_device_queried=True))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(int64=0, zeros=lambda *a, **k: Tensor()))
    monkeypatch.setitem(sys.modules, 'deep_ep._native', SimpleNamespace(require_native=lambda _: native))
    module = load('deep_ep.utils.ll_stage_timer', 'deep_ep/utils/ll_stage_timer.py')
    return module, native, copies


@pytest.mark.parametrize('operation', ('dispatch', 'combine'))
def test_capture_both_operations_aligns_attaches_once_and_defers_copy(monkeypatch, tmp_path, operation):
    module, native, copies = fake_native(monkeypatch, operation)
    trace = module.StageTrace(object(), operation)
    assert trace.tensor.shape == (56, 16, 8) and trace.tensor.data_ptr() % 512 == 0
    tensors = {'input': SimpleNamespace(device='npu:0')}
    with module.capture(trace.buffer, operation, prepared=trace, defer_complete=True):
        module.attach(trace.buffer, tensors, operation)
        assert tensors['ll_stage_timer_mode'] == 1 and tensors['ll_stage_timer'] is trace.tensor
        with pytest.raises(RuntimeError, match='multiple launches'):
            module.attach(trace.buffer, tensors, operation)
    assert not copies
    trace.complete()
    trace.save(tmp_path / 'sample_0000.json', dict(rank=0, iteration=0))
    assert copies == [(56, 16, 8)]
    saved = json.loads((tmp_path / 'sample_0000.json').read_text())
    assert saved['schema'] == 18 and saved['timer_profile'] == 'systemcycle'
    chrome = json.loads((tmp_path / 'trace_sample_0000.json').read_text())
    assert chrome['metadata']['format'] == 'systemcycle-stages'
    assert chrome['metadata']['all_calls_in_metrics']
    assert chrome['metadata']['cumulative_metrics_have_no_timeline']
    assert len([row for row in chrome['traceEvents'] if row['ph']=='X']) == len(saved['intervals'])
    assert len(saved['cycle_metrics'])==56
    native.low_latency_overlap_stage = 2
    with pytest.raises(RuntimeError, match='stage3'):
        module.StageTrace(object(), operation)


@pytest.mark.parametrize('field,value', [
    ('slots', 8), ('fields', 4), ('points', list(timer.bandwidth.POINTS)),
    ('timer_profile_id', 8), ('timer_profile', 'bandwidth'), ('recording', 'stage-bandwidth'),
    ('cycle_metric_names', list(reversed(timer.systemcycle.METRIC_NAMES))),
    ('cycle_metric_indices', list(reversed(timer.systemcycle.METRIC_INDICES))),
    ('counter_indices', []), ('metric_word_offset', 32), ('metric_record_words', 4)])
def test_native_layout_drift_is_rejected_before_capture(monkeypatch, field, value):
    module, native, copies = fake_native(monkeypatch, 'dispatch')
    wrong = layout()
    wrong[field] = value
    native.low_latency_stage_timer_layout = lambda: wrong
    with pytest.raises(RuntimeError, match='schema mismatch'):
        module.StageTrace(object(), 'dispatch')
    assert not copies


def test_entry_requires_stage3_and_exact_selected_metric_dictionary():
    entry.validate_stage_timer_capability(capability())
    for field in ('cycle_metric_names', 'cycle_metric_indices', 'counter_indices',
                  'metric_word_offset', 'metric_record_words'):
        wrong = capability()
        wrong['stage_timer_' + field] = None
        with pytest.raises(RuntimeError, match='SystemCycle statistic layout'):
            entry.validate_stage_timer_capability(wrong)
    wrong = capability()
    wrong['overlap_stage'] = 2
    with pytest.raises(RuntimeError, match='stage3'):
        entry.validate_stage_timer_capability(wrong)
    for mode in ('mte3-tx', 'mte3-rx', 'bandwidth-complete'):
        with pytest.raises(RuntimeError, match='schema16'):
            entry.validate_stage_timer_capability(capability(), mode)


def test_matching_capabilities_include_the_full_cycle_layout():
    left, right = capability(), capability()
    entry.validate_matching_capabilities([left, right])
    right['stage_timer_metric_word_offset'] += 1
    with pytest.raises(RuntimeError, match='differs across ranks'):
        entry.validate_matching_capabilities([left, right])


def test_historical_capture_layouts_keep_their_original_points_and_slots():
    assert [len(timer.CAPTURE_POINTS[schema]) for schema in (15, 16, 17, 18)] == [14, 18, 18, 40]
    assert [timer.CAPTURE_SCHEMAS[schema][0] for schema in (15, 16, 17, 18)] == [8, 8, 16, 16]
    for schema, source in ((16, 'test_mte3_capture_contract.py'), (17, 'test_bandwidth_capture_contract.py')):
        fixture = load(f'historical_capture_{schema}', f'tests/ll_stage_timer/{source}')
        entry.validate_stage_timer_capability(fixture.capability())
        assert timer.parse_records(fixture.raw(1), 'dispatch')['schema'] == schema
