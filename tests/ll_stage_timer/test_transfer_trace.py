"""Transferred timelines preserve real intervals and reject incompatible inputs."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


trace = load('transferred_trace', 'tests/ll_stage_timer/transfer_trace.py')
fixture = load('transferred_segments', 'tests/ll_stage_timer/test_segments.py')
timer = fixture.timer


def add_duration(raw, name, begin, end):
    stage = timer.STAGES.index(name)
    words = [word for row in raw[0] for word in row]
    events = [tuple(words[8+i*2:10+i*2]) for i in range(words[6])]
    position = next(i for i, pair in enumerate(events) if (pair[0] & 0xffffffff) >= 0x8000)
    events[position:position] = [(stage*2, begin), (stage*2+1, end)]
    events.append(((1 << 32) | (0x8000+stage), end-begin))
    words[6] = len(events)
    for i, pair in enumerate(events):
        words[8+i*2:10+i*2] = pair
    raw[0] = [words[i:i+8] for i in range(0, len(words), 8)]


def sample(schema=14, profile=0, rank=0, iteration=0, offset=0, frequency=1e9):
    raw = fixture.raw(schema=schema, profile=profile)
    if schema == 14:
        add_duration(raw, 'TIMER_SETUP', 1, 6)
        add_duration(raw, 'PREFIX_READY_WAIT', 600, 670)
        add_duration(raw, 'BEFORE_FIRST_RECV_POLL', 100, 210)
    if offset:
        for rows in raw:
            words = [word for row in rows for word in row]
            for i in range(words[6]):
                if (words[8+i*2] & 0xffffffff) < 0x8000:
                    words[9+i*2] += offset
            rows[:] = [words[i:i+8] for i in range(0, len(words), 8)]
    meta = dict(rank=rank, iteration=iteration, world_size=2, config=dict(B=1, M=32),
                run_id='run-a', case_id='small', correctness='passed')
    saved = dict(raw=raw, operation='dispatch', metadata=meta, frequency_hz=frequency)
    return saved, (meta, timer.parse_records(raw, 'dispatch', frequency))


def intervals(result):
    return [event for event in result['traceEvents'] if event['ph'] == 'X']


@pytest.mark.parametrize('schema,profile', [(13,0),(14,0),(14,3),(14,5)])
def test_matches_native_parser_real_intervals(schema, profile):
    saved, decoded = sample(schema=schema, profile=profile)
    before = copy.deepcopy(decoded)
    actual = trace.trace_from_decoded([decoded])
    native = timer.chrome_trace([saved])
    def key(event, restored):
        args = event['args']
        return (args['rank'], args['iteration'], args['core'] if restored else event['tid'],
                event['name'], event['ts'], event['dur'], args['sample'],
                args['raw_begin_cycle'], args['raw_end_cycle'])
    assert sorted(key(event, True) for event in intervals(actual)) == sorted(
        key(event, False) for event in intervals(native))
    assert decoded == before
    json.dumps(actual, allow_nan=False)


def test_real_waits_and_setup_remain_without_metric_synthesis():
    _, decoded = sample(profile=3)
    result = trace.trace_from_decoded([decoded])
    events = intervals(result)
    core0 = [event for event in events if event['args']['core'] == 0]
    waits = [event for event in core0 if event['name'] == 'PREFIX_READY_WAIT']
    assert len(waits) == 1 and waits[0]['dur'] == pytest.approx(.07)
    polls = [event for event in core0 if event['name'] == 'RECV_POLL']
    assert len(polls) == 2  # Accumulated calls=3; the third interval is not invented.
    assert sum(event['dur'] for event in polls) == pytest.approx(.01)
    assert not any(event['name'] == 'POLL_RECV_COUNTS_ITERATIONS' for event in events)
    setup = next(event for event in core0 if event['name'] == 'TIMER_SETUP')
    kernel = next(event for event in core0 if event['name'] == 'KERNEL')
    milestone = next(event for event in core0 if event['name'] == 'BEFORE_FIRST_RECV_POLL')
    assert setup['ts'] == 0 and kernel['ts'] == pytest.approx(.099)
    assert len({setup['tid'], kernel['tid'], milestone['tid'], polls[0]['tid']}) == 4
    assert all(event['args']['timer_profile'] == 'prefix' for event in events)
    assert result['metadata']['cores'][0]['origin_tick'] == 1


def test_selected_iteration_separate_rank_origins_and_readable_tracks():
    samples = [sample(rank=rank, iteration=iteration, offset=rank*1000000+iteration*20000)[1]
               for iteration in (0,1) for rank in (0,1)]
    result = trace.trace_from_decoded(samples, trace_iteration=1)
    events = intervals(result)
    assert {event['args']['iteration'] for event in events} == {1}
    assert {event['args']['rank'] for event in events} == {0,1}
    assert {event['pid'] for event in events} == {1,2}
    origins = [row['origin_tick'] for row in result['metadata']['cores']]
    assert origins == [20001,1020001]
    assert all(min(event['ts'] for event in events if event['pid'] == pid) == 0 for pid in (1,2))
    metadata = [event for event in result['traceEvents'] if event['ph'] == 'M']
    assert sum(event['name'] == 'process_name' for event in metadata) == 2
    assert all(type(event['pid']) is int and type(event['tid']) is int for event in result['traceEvents'])
    assert any('rank 1 / dispatch / iteration 1 / full' == event['args'].get('name') for event in metadata)
    assert any('AIV 00' in event['args'].get('name', '') for event in metadata)
    assert result['metadata']['source_samples'] == 4 and result['metadata']['selected_samples'] == 2
    assert result['metadata']['clock'] == 'Rank-local origins; no cross-rank alignment'


@pytest.mark.parametrize('field,value', [
    ('schema',13), ('boundary_mode','relay-complete'), ('send_cores',28),
    ('frequency_hz',2e9), ('timer_profile','prefix')])
def test_mixed_decoder_identity_rejected(field, value):
    left = sample(rank=0)[1]
    right = sample(rank=1)[1]
    right[1][field] = value
    if field == 'timer_profile':
        right[1]['timer_profile_id'] = 3
    with pytest.raises(ValueError, match='Mixed|profile'):
        trace.trace_from_decoded([left,right])


def test_configuration_and_duplicate_identity_rejected():
    left = sample(rank=0)[1]
    right = sample(rank=1)[1]
    right[0]['config']['B'] = 2
    with pytest.raises(ValueError, match='Mixed configurations'):
        trace.trace_from_decoded([left,right])
    with pytest.raises(ValueError, match='Duplicate rank'):
        trace.trace_from_decoded([left,copy.deepcopy(left)])


@pytest.mark.parametrize('schema', [13, 14, 15])
def test_rank_local_capture_observations_are_not_transfer_identities(schema):
    if schema == 15:
        sparse_fixture = load('transfer_sparse_observations', 'tests/ll_stage_timer/test_sparse_timer.py')
        samples = [(dict(sparse_fixture.metadata(rank=rank), world_size=2),
                    timer.parse_records(sparse_fixture.raw(), 'dispatch', 1e9)) for rank in range(2)]
    else:
        samples = [sample(schema=schema, rank=rank)[1] for rank in range(2)]
    for rank, (meta, _) in enumerate(samples):
        meta['active_capture_profile'] = dict(measurement='active-capture full profiler kernel task',
            status='available' if rank == 0 else 'unavailable', all_ranks_status='unavailable',
            kernel_us=40.125 if rank == 0 else None, reason=None if rank == 0 else 'missing task')
    before = copy.deepcopy(samples)
    result = trace.trace_from_decoded(samples)
    assert {event['args']['rank'] for event in intervals(result)} == {0, 1}
    assert samples == before
    samples[1][0]['active_capture_profile']['measurement'] = 'event time'
    with pytest.raises(ValueError, match='Mixed configurations'):
        trace.trace_from_decoded(samples)


@pytest.mark.parametrize('frequency', [None,0,-1,float('nan'),float('inf'),True])
def test_explicit_frequency_is_required(frequency):
    data = sample()[1]
    data[1]['frequency_hz'] = frequency
    with pytest.raises(ValueError, match='frequency_hz'):
        trace.trace_from_decoded([data])
    result = trace.trace_from_decoded([data], frequency_hz=2e9)
    assert next(event for event in intervals(result) if event['name'] == 'TIMER_SETUP')['dur'] == .0025


def test_missing_iteration_or_intervals_is_not_fabricated():
    data = sample()[1]
    with pytest.raises(ValueError, match='No diagnostic iteration'):
        trace.trace_from_decoded([data], trace_iteration=3)
    data[1]['intervals'] = []
    with pytest.raises(ValueError, match='No actual intervals'):
        trace.trace_from_decoded([data])
    data[1]['schema'] = 12
    with pytest.raises(ValueError, match='schema 13 or 14'):
        trace.trace_from_decoded([data])


@pytest.mark.parametrize('field,value', [('total_ticks',999),('last_tick',0),('first_tick',-1),
                                      ('sample',False),('core',56)])
def test_invalid_interval_cannot_create_a_trace(field, value):
    data = sample()[1]
    data[1]['intervals'][0][field] = value
    with pytest.raises(ValueError, match='Invalid|exceeds'):
        trace.trace_from_decoded([data])


def test_raw_ticks_remain_exact_beyond_float_integer_precision():
    _, data = sample(offset=2**54)
    result = trace.trace_from_decoded([data])
    setup = next(event for event in intervals(result) if event['name'] == 'TIMER_SETUP')
    assert setup['args']['raw_begin_cycle'] == 2**54+1
    assert setup['args']['raw_end_cycle'] == 2**54+6
    assert setup['args']['raw_duration_cycles'] == 5
    assert setup['ts'] == 0 and setup['dur'] == .005
