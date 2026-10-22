"""Profile identity, native allocation handshake and historical carrier shapes."""
import copy
import importlib.util
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


fixture = load('profile_segments', 'tests/ll_stage_timer/test_segments.py')
timer = load('profile_timer', 'deep_ep/utils/ll_stage_timer.py')


def add_duration(raw, stage):
    words = [v for row in raw[0] for v in row]
    events = [tuple(words[8 + i * 2:10 + i * 2]) for i in range(words[6])]
    first_metric = next(i for i, (tag, _) in enumerate(events) if (tag & 0xffffffff) >= 0x8000)
    events[first_metric:first_metric] = [(stage * 2, 700), (stage * 2 + 1, 705)]
    events.append(((1 << 32) | (0x8000 + stage), 5))
    words[6] = len(events)
    for i, pair in enumerate(events):
        words[8 + 2 * i:10 + 2 * i] = pair
    raw[0] = [words[i:i + 8] for i in range(0, len(words), 8)]
    return raw


@pytest.mark.parametrize('schema,slots,metrics', [
    (7,72,64),(9,72,66),(10,72,70),(11,72,78),(12,72,81),(13,80,108),(14,88,136)])
def test_real_historical_carrier_shapes(schema, slots, metrics):
    data = fixture.raw(schema=schema)
    assert len(data[0]) == slots
    decoded = timer.parse_records(data, 'dispatch')
    assert decoded['schema'] == schema
    assert decoded['timer_profile'] == 'full' and decoded['timer_profile_id'] == 0
    assert timer.SCHEMA_METRICS[schema] == metrics


def test_schema13_does_not_accept_schema14_metrics_or_carrier():
    with pytest.raises(ValueError, match='Invalid interval event'):
        timer.parse_records(add_duration(fixture.raw(schema=13), 108), 'dispatch')
    data = fixture.raw(schema=13)
    for rows in data:
        rows.extend([[0] * 8 for _ in range(8)])
    with pytest.raises(ValueError, match='Timer shape mismatch'):
        timer.parse_records(data, 'dispatch')


@pytest.mark.parametrize('profile,name', list(enumerate(timer.TIMER_PROFILES)))
def test_profile_survives_decode_and_trace_metadata(profile, name):
    data = fixture.raw(schema=14, profile=profile)
    report = timer.parse_records(data, 'dispatch', 1e9)
    assert (report['timer_profile_id'], report['timer_profile']) == (profile, name)
    assert report['cores'][10]['initial_role'] == 'cumsum'
    assert report['cores'][10]['role'] == 'cumsum'  # Missing TX counter is not evidence of activity.
    assert all(r['timer_profile'] == name for r in report['records'] + report['intervals'] + report['cores'])
    trace = timer.chrome_trace([dict(raw=data, operation='dispatch', frequency_hz=1e9)])
    assert trace['metadata']['timer_profile'] == name
    assert trace['metadata']['timer_profile_id'] == profile
    assert trace['metadata']['cores'][0]['timer_profile'] == name


def test_unknown_mixed_and_legacy_profile_bits_rejected():
    with pytest.raises(ValueError, match='Unknown timer profile'):
        timer.parse_records(fixture.raw(schema=14, profile=6), 'dispatch')
    mixed = fixture.raw(schema=14)
    mixed[-1][0][5] |= 1 << 24
    with pytest.raises(ValueError, match='Mixed timer profiles'):
        timer.parse_records(mixed, 'dispatch')
    with pytest.raises(ValueError, match='Unexpected segment header bits'):
        timer.parse_records(fixture.raw(schema=13, profile=1), 'dispatch')
    captures = [dict(raw=fixture.raw(schema=14, profile=i), operation='dispatch', frequency_hz=1e9,
                     metadata=dict(rank=i, iteration=0)) for i in (0,1)]
    with pytest.raises(ValueError, match='Mixed timer profiles'):
        timer.chrome_trace(captures)


def test_final_metric_is_a_duration_and_counter_ids_are_unchanged():
    report = timer.parse_records(add_duration(fixture.raw(schema=14), 135), 'dispatch')
    stage = next(row for row in report['records'] if row['stage'] == 'TIMER_SETUP')
    assert stage['total_ticks'] == 5 and stage['time_measured']
    assert timer.COUNTERS == frozenset((*range(51,56),69,*range(74,78),*range(103,108)))


def fake_native(monkeypatch, profile=0, **layout_changes):
    package = ModuleType('deep_ep'); package.__path__ = []
    utils = ModuleType('deep_ep.utils'); utils.__path__ = []
    monkeypatch.setitem(sys.modules, 'deep_ep', package)
    monkeypatch.setitem(sys.modules, 'deep_ep.utils', utils)
    data = fixture.raw(schema=14, profile=profile)

    class Tensor:
        device = 'npu:0'
        ptr = 256
        def data_ptr(self): return self.ptr
        def narrow(self, dim, offset, count): self.ptr += offset * 8; return self
        def view(self, *shape): self.shape = shape; return self
        def cpu(self): return self
        def tolist(self): return data

    layout = dict(schema=timer.SCHEMA, slots=timer.SLOTS, fields=8, stages=list(timer.STAGES),
                  timer_profile=timer.TIMER_PROFILES[profile], timer_profile_id=profile)
    layout.update(layout_changes)
    native = SimpleNamespace(low_latency_stage_timer_enabled=True,
        low_latency_stage_timer_layout=lambda: copy.deepcopy(layout),
        low_latency_core_config=lambda: dict(active_aiv=56, send_cores=10, device=0, runtime_device_queried=True))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(int64=0, zeros=lambda *a, **k: Tensor()))
    monkeypatch.setitem(sys.modules, 'deep_ep._native', SimpleNamespace(require_native=lambda _: native))
    return load('deep_ep.utils.ll_stage_timer', 'deep_ep/utils/ll_stage_timer.py'), data


def test_native_profile_and_allocation_handshake(monkeypatch):
    module, data = fake_native(monkeypatch, profile=3)
    trace = module.StageTrace(object(), 'dispatch')
    assert trace.tensor.shape == (56,88,8) and trace.tensor.data_ptr() % 512 == 0
    assert (trace.timer_profile_id, trace.timer_profile) == (3, 'prefix')
    trace.launches = 1
    trace.complete()
    for rows in data:
        rows[0][5] = (rows[0][5] & ~(255 << 24)) | (4 << 24)
    with pytest.raises(RuntimeError, match='different timer profile'):
        trace.complete()


@pytest.mark.parametrize('changes', [dict(slots=80),dict(schema=13),dict(timer_profile_id=None),
    dict(timer_profile_id=6),dict(timer_profile='prefix'),dict(timer_profile_id=True)])
def test_native_stale_or_invalid_profile_rejected(monkeypatch, changes):
    module, _ = fake_native(monkeypatch, **changes)
    with pytest.raises(RuntimeError, match='Python/native schema mismatch'):
        module.StageTrace(object(), 'dispatch')


def relay_counter_only(profile):
    data=fixture.raw(schema=14,profile=profile)
    words=[v for row in data[0] for v in row]
    position=words[6]
    words[8+2*position:10+2*position]=[(3<<32)|(0x8000+69),0]
    words[6]+=1
    data[0]=[words[i:i+8] for i in range(0,len(words),8)]
    return data


@pytest.mark.parametrize('profile',[1,2,3,4])
def test_sparse_relay_counter_is_retained_without_fake_issue_times(profile):
    decoded=timer.parse_records(relay_counter_only(profile),'dispatch',1e9)
    relay=load('sparse_relay_report','tests/ll_stage_timer/relay_timing.py')
    result=relay.report([(dict(rank=0,iteration=0),decoded)])
    row=result['per_core'][0]
    assert row['status']=='counter_only' and row['relay_writes']==3
    assert row['groups'] is None and row['issue_ticks_total'] is None
    assert row['issue_ticks_per_write'] is None and row['relay_writes_per_group'] is None
    assert result['per_core'][1]['issue_ticks_total'] is None


@pytest.mark.parametrize('profile',[0,5])
def test_full_recv_profiles_still_reject_incomplete_relay(profile):
    with pytest.raises(ValueError,match='Relay metrics without issue'):
        timer.parse_records(relay_counter_only(profile),'dispatch')


@pytest.mark.parametrize('profile',[1,2,3,4])
def test_sparse_profiles_reject_completion_capture_before_allocation(monkeypatch,profile):
    module,_=fake_native(monkeypatch,profile=profile)
    with pytest.raises(ValueError,match='full or recv'):
        module.StageTrace(object(),'dispatch','relay-complete')
    with pytest.raises(ValueError,match='full or recv'):
        timer.parse_records(fixture.raw(schema=14,profile=profile,mode=3),'dispatch')
