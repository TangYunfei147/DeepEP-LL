"""CPU checks for lossless sampled-interval rendering after text restoration."""
import copy
import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import receiver_export as export
import receiver_trace as render


def result_fixture():
    case = json.dumps(dict(B=256, route='正常'), ensure_ascii=False)
    samples = []
    for rank in (0, 1):
        for iteration in (0, 1):
            origin = 1000000000 * (rank + 1) + iteration * 1000000
            cores = []
            for core in (0, 1):
                arms = {}
                for index, name in enumerate(export.CASES):
                    begin = origin + 1000 * index + 10 * core
                    ticks = 30 + index * 10
                    arms[name] = dict(ticks=ticks,
                        flag_batches=0 if name == 'skip-flags' else 3,
                        payload_bytes=0 if name == 'no-payload-output' else 35 * 3696,
                        first_tick=begin, last_tick=begin + ticks)
                cores.append(dict(core=core, arms=arms))
            samples.append(dict(rank=rank, iteration=iteration, frequency_hz=1000000000,
                                order=iteration, cores=cores))
    return dict(identity=dict(mode='ready-breakdown'), summary={case: {'limitations': '采样区间'}},
                samples={case: samples})


def make_transfer(tmp_path, result=None, limit=60000):
    root = tmp_path / 'restored evidence with spaces'
    export.export_result(result or result_fixture(), root, limit)
    return root


def load_outputs(paths):
    return {path.name: json.loads(path.read_text(encoding='utf-8')) for path in paths}


def test_transfer_ranks_are_separate_and_each_capture_has_its_own_origin(tmp_path):
    root = make_transfer(tmp_path)
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    output = load_outputs(render.render_bundle(root, 'receiver-transfer'))
    traces = [value for name, value in output.items() if name != 'render-report.json']
    assert len(traces) == 2
    for rank, trace in enumerate(traces):
        events = [event for event in trace['traceEvents'] if event['ph'] == 'X']
        assert {event['args']['rank'] for event in events} == {rank}
        assert {event['pid'] for event in events} == {1, 2}
        assert {event['tid'] for event in events} == {0, 1}
        assert len(events) == 12
        for pid in (1, 2):
            capture = [event for event in events if event['pid'] == pid]
            assert min(event['ts'] for event in capture) == 0
            assert {event['dur'] for event in capture} == {.03, .04, .05}
        assert {event['args']['frequency_hz'] for event in events} == {1000000000}
        assert 'not a complete dispatch trace' in trace['metadata']['scope']
        assert 'No cross-rank or cross-capture alignment' in trace['metadata']['clock']
    report = output['render-report.json']
    assert not report['skipped_ranks']
    assert all(capture['status'] == 'rendered' for capture in report['captures'])
    assert all((root / name).read_bytes() == content for name, content in before.items())


def test_missing_and_partial_intervals_are_reported_without_synthetic_events(tmp_path):
    result = result_fixture()
    rows = next(iter(result['samples'].values()))
    for row in rows:
        if row['rank'] == 1:
            for core in row['cores']:
                for arm in core['arms'].values():
                    arm['first_tick'] = arm['last_tick'] = None
    rows[0]['cores'][0]['arms']['baseline']['first_tick'] = None
    root = make_transfer(tmp_path, result)
    output = load_outputs(render.render_bundle(root, 'receiver-transfer'))
    report = output['render-report.json']
    assert report['skipped_ranks'] == [dict(case='case_000', rank=1,
        reason='No actual sampled intervals in any capture')]
    assert len(report['rendered_trace_files']) == 1
    events = [event for event in output[report['rendered_trace_files'][0]]['traceEvents'] if event['ph'] == 'X']
    assert len(events) == 11
    assert report['captures'][0]['status'] == 'partial'
    assert report['captures'][0]['missing_intervals'] == [dict(core=0, arm='baseline', reason='missing_sampled_interval')]


def test_all_missing_still_writes_an_explicit_report(tmp_path):
    result = result_fixture()
    for row in next(iter(result['samples'].values())):
        for core in row['cores']:
            for arm in core['arms'].values():
                arm.pop('first_tick')
                arm.pop('last_tick')
    output = load_outputs(render.render_bundle(make_transfer(tmp_path, result), 'receiver-transfer'))
    assert list(output) == ['render-report.json']
    assert len(output['render-report.json']['skipped_ranks']) == 2


def test_fragmented_records_are_reassembled_and_verified_before_rendering(tmp_path):
    result = result_fixture()
    next(iter(result['summary'].values()))['long_note'] = '完整记录😀' * 11000
    root = make_transfer(tmp_path, result)
    assert any('fragment_record_sha256' in p.read_text(encoding='ascii')
               for p in root.glob('*.json'))
    assert len(render.render_bundle(root, 'receiver-transfer')) == 3


def test_checksum_failure_does_not_create_rendered_output(tmp_path):
    root = make_transfer(tmp_path)
    part = next(root.glob('*-cores.json'))
    part.write_bytes(part.read_bytes() + b' ')
    with pytest.raises(ValueError):
        render.render_bundle(root, 'receiver-transfer')
    assert not (root / 'rendered').exists()


@pytest.mark.parametrize('damage', ['duration', 'frequency', 'duplicate_capture'])
def test_inconsistent_transfer_timestamps_and_identities_are_rejected(tmp_path, damage):
    result = result_fixture()
    rows = next(iter(result['samples'].values()))
    if damage == 'duration':
        rows[0]['cores'][0]['arms']['baseline']['last_tick'] += 1
    elif damage == 'frequency':
        rows[0]['frequency_hz'] = 0
    else:
        rows.append(copy.deepcopy(rows[0]))
    root = make_transfer(tmp_path, result)
    with pytest.raises(ValueError):
        render.render_bundle(root, 'receiver-transfer')
    assert not (root / 'rendered').exists()


def raw_fixture(rank=3, iteration=8):
    timer = render._timer()
    raw = []
    for core in range(56):
        # RECV_INPUT has three calls but only two real intervals. The renderer
        # must not manufacture a third interval from its accumulated duration.
        pairs = [(100, 110), (101, 120), (120, 210), (121, 230),
                 (120, 260), (121, 280), (0, 200), (1, 500)]
        metrics = [(0, 1, 300), (50, 1, 10), (60, 3, 90)]
        records = pairs + [((0x8000 + stage) | (calls << 32), ticks)
                           for stage, calls, ticks in metrics]
        words = [0] * (timer.SLOTS * timer.FIELDS)
        role = 3 if core < 30 else 2
        words[:8] = [timer.MAGIC, 15, 1, 56, core, 1 | (30 << 8) | (role << 16), len(records), 1]
        for index, pair in enumerate(records):
            words[8 + 2 * index:10 + 2 * index] = pair
        raw.append([words[index:index + 8] for index in range(0, len(words), 8)])
    return dict(raw=raw, operation='dispatch', frequency_hz=1000000000,
                metadata=dict(rank=rank, iteration=iteration, case_id='测试', config=dict(B=256)))


def test_raw_capture_uses_existing_parser_without_expanding_cumulative_metrics(tmp_path):
    root = tmp_path / 'raw run'
    root.mkdir()
    for iteration in (8, 9):
        (root / f'sample_{iteration}.json').write_text(json.dumps(raw_fixture(iteration=iteration)), encoding='utf-8')
    (root / 'summary.json').write_text('{}', encoding='utf-8')
    output = load_outputs(render.render_bundle(root, 'receiver-run'))
    report = output['render-report.json']
    assert len(report['rendered_trace_files']) == 2
    for filename in report['rendered_trace_files']:
        trace = output[filename]
        events = [event for event in trace['traceEvents'] if event['ph'] == 'X']
        assert len(events) == 56 * 4
        assert {event['args']['rank'] for event in events} == {3}
        assert len({event['args']['iteration'] for event in events}) == 1
        assert {event['name'] for event in events} == {'INIT', 'KERNEL', 'RECV_INPUT'}
        assert len([event for event in events if event['tid'] == 0 and event['name'] == 'RECV_INPUT']) == 2
        assert min(event['ts'] for event in events) == 0
        assert trace['metadata']['origin_tick'] == 110
        assert trace['metadata']['frequency_hz'] == 1000000000


def test_duplicate_raw_capture_fails_before_writing(tmp_path):
    root = tmp_path / 'raw run'
    root.mkdir()
    for name in ('one.json', 'two.json'):
        (root / name).write_text(json.dumps(raw_fixture()), encoding='utf-8')
    with pytest.raises(ValueError, match='Duplicate raw'):
        render.render_bundle(root, 'receiver-run')
    assert not (root / 'rendered').exists()


def test_existing_output_is_preserved(tmp_path):
    root = make_transfer(tmp_path)
    output = root / 'rendered'
    output.mkdir()
    sentinel = output / 'keep.txt'
    sentinel.write_text('user file', encoding='utf-8')
    with pytest.raises(ValueError, match='exists'):
        render.render_bundle(root, 'receiver-transfer')
    assert sentinel.read_text(encoding='utf-8') == 'user file'


def test_unknown_source_kind_does_not_create_output(tmp_path):
    with pytest.raises(ValueError, match='source kind'):
        render.render_bundle(tmp_path, 'unknown')
    assert not (tmp_path / 'rendered').exists()
