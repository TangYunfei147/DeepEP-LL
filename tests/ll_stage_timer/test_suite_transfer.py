"""Offline suite compression, coverage and exact reconstruction checks."""
import hashlib
import importlib.util
import json
from pathlib import Path
import random

import pytest


def module(name):
    spec = importlib.util.spec_from_file_location('test_suite_' + name, Path(__file__).with_name(name + '.py'))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


suite_transfer = module('suite_transfer')


def make_suite(tmp_path, profiles=('prefix', 'send'), bs=(1, 32), noise_bytes=0):
    suite = tmp_path / 'suite'
    (suite / 'node_000').mkdir(parents=True)
    plan = dict(matrix=dict(B=list(bs), M=32, H=3584, K=8, E=384, dtype='fp8'), nodes=1,
                stages=[dict(name='off', timer=False, profile='full')] +
                       [dict(name=p, timer=True, profile=p) for p in profiles],
                python='/private/environment/not-to-transfer', venv='/private/venv')
    (suite / 'node_000/plan.json').write_text(json.dumps(plan))
    (suite / 'runs/off').mkdir(parents=True)
    (suite / 'runs/off/summary.json').write_text(json.dumps(dict(results=[
        dict(config=dict(B=b), route='normal', dispatch=dict(max_rank_mean_kernel_us=17+b/10)) for b in bs])))
    (suite / 'PERFORMANCE_COMPARE.txt').write_text('OFF and ON full tasks: exact comparisons\n')
    transfer = module('transfer_report')
    expected = []
    for profile in profiles:
        for b in bs:
            meta = dict(rank=0, iteration=0, config=dict(B=b, M=32), source_revision='revision-a',
                        note='one repeated metadata block ' * 100)
            if noise_bytes:
                meta['noise'] = random.Random(12).randbytes(noise_bytes).hex()
            decoded = dict(schema=14, operation='dispatch', active_aiv=2, send_cores=1,
                           timer_profile=profile, timer_profile_id=('full', 'overview', 'prepare', 'prefix', 'send', 'recv').index(profile),
                           frequency_hz=1000000000, boundary_mode='observe',
                           records=[dict(core=c, stage='KERNEL', total_ticks=1000+c, calls=1,
                                         time_measured=True, role='send' if c == 0 else 'recv') for c in range(2)],
                           intervals=[dict(core=c, stage='KERNEL', sample=0,
                                           first_tick=100, last_tick=1100+c, total_ticks=1000+c) for c in range(2)],
                           cores=[dict(core=c, omitted_intervals=0) for c in range(2)])
            context = dict(schema=14, timer_profile=profile, world_size=1,
                           sampling=dict(diagnostic_iters=1), configurations=dict(dispatch=dict(B=b, M=32)))
            directory = suite / 'analysis' / profile / f'case_{b:05d}' / 'normal'
            transfer.export_bundle(directory, context, [(meta, decoded)])
            expected.append((profile, b, context, [(meta, decoded)]))
            active = suite / 'runs' / profile / f'case_{b:05d}' / 'normal'
            active.mkdir(parents=True)
            (active / 'active_capture_dispatch.json').write_text(json.dumps(dict(
                config=dict(B=b, M=32), status='available', profile=profile,
                comparison=dict(source_revision='revision-a'), diagnostic_iters=1)))
    return suite, expected


def group_payloads(suite, manifest):
    codec = module('text_codec')
    directory = suite / manifest['output']
    values = []
    for group in manifest['groups']:
        count = group['part_count']
        names = [f'suite_{group["group"]:03d}_part_{index:03d}_of_{count:03d}.txt' for index in range(1, count+1)]
        texts = [(directory / name).read_text() for name in names]
        status, raw = codec.inspect_parts(texts[::-1] + texts[:1])
        assert raw is not None, status
        assert hashlib.sha256(raw).hexdigest() == group['payload_sha256']
        values.append(json.loads(raw))
    return values


def test_suite_is_exact_compact_allowlisted_and_idempotent(tmp_path):
    suite, expected = make_suite(tmp_path)
    (suite / 'node_000/private.log').write_text('MUST NOT LEAVE THE MACHINE')
    manifest = suite_transfer.export_suite(suite)
    assert manifest['complete'] is True
    assert len(manifest['groups']) == 1
    payload = group_payloads(suite, manifest)[0]
    serialized = json.dumps(payload)
    assert 'MUST NOT LEAVE' not in serialized and '/private/' not in serialized
    assert 'PERFORMANCE_COMPARE.txt' in payload['files']
    assert len(payload['files']) == 6
    assert manifest['transfer_bytes'] < manifest['uncompressed_bytes'] / 2
    transfer = module('transfer_report')
    for case, (profile, b, context, samples) in zip(payload['cases'], expected):
        recovered_context, recovered_samples = transfer.decode_samples(json.dumps(case['payload']).encode())
        assert (recovered_context, recovered_samples) == (context, samples)
        assert case['profile'] == profile and case['B'] == [b]
    assert suite_transfer.export_suite(suite) == manifest
    for path in (suite / manifest['output']).iterdir():
        assert len(path.read_bytes()) <= 60000
        assert len(path.read_text(encoding='utf-8')) <= 60000


def test_large_suite_groups_preserve_boundaries_and_expected_hashes(tmp_path, monkeypatch):
    suite, expected = make_suite(tmp_path)
    monkeypatch.setattr(suite_transfer, 'GROUP_BYTES', 1000)
    manifest = suite_transfer.export_suite(suite, limit=6000)
    payloads = group_payloads(suite, manifest)
    assert len(payloads) == len(expected)
    assert payloads[-1]['group_count'] == len(payloads)
    assert payloads[-1]['preceding_groups'] == [
        dict(group=group['group'], payload_sha256=group['payload_sha256']) for group in manifest['groups'][:-1]]
    assert all(not payload['files'] and payload['coverage']['complete'] is None for payload in payloads[:-1])
    assert 'runs/off/summary.json' in payloads[-1]['files']
    assert len(payloads[-1]['suite_case_index']) == len(expected)
    assert [case['profile'] for payload in payloads for case in payload['cases']] == [item[0] for item in expected]
    for path in (suite / manifest['output']).iterdir():
        assert len(path.read_bytes()) <= 6000


def test_repack_changed_reports_preserves_previous_bundle(tmp_path):
    suite, _ = make_suite(tmp_path)
    first = suite_transfer.export_suite(suite)
    first_files = {p.name: p.read_bytes() for p in (suite / first['output']).iterdir()}
    (suite / 'PERFORMANCE_COMPARE.txt').write_text('New, complete target data')
    second = suite_transfer.export_suite(suite)
    assert first['output'] != second['output']
    assert first_files == {p.name: p.read_bytes() for p in (suite / first['output']).iterdir()}


def test_missing_cases_and_baseline_are_explicit(tmp_path):
    suite, _ = make_suite(tmp_path, profiles=('prefix',), bs=(1,))
    path = suite / 'node_000/plan.json'
    plan = json.loads(path.read_text())
    plan['matrix']['B'].append(32)
    path.write_text(json.dumps(plan))
    (suite / 'runs/off/summary.json').unlink()
    manifest = suite_transfer.export_suite(suite)
    assert manifest['complete'] is False
    payload = group_payloads(suite, manifest)[0]
    issues = '\n'.join(payload['coverage']['issues'])
    assert 'Missing OFF' in issues and 'prefix: B=[32]' in issues
    assert len(payload['cases']) == 1
    assert 'INCOMPLETE' in (suite / manifest['output'] / '00_overview.txt').read_text()


def test_unavailable_active_task_measurement_is_not_complete(tmp_path):
    suite, _ = make_suite(tmp_path, profiles=('prefix',), bs=(1,))
    active = suite / 'runs/prefix/case_00001/normal/active_capture_dispatch.json'
    value = json.loads(active.read_text())
    value['status'] = 'unavailable'
    active.write_text(json.dumps(value))
    manifest = suite_transfer.export_suite(suite)
    assert manifest['complete'] is False
    payload = group_payloads(suite, manifest)[0]
    assert any('measurement is unavailable' in issue for issue in payload['coverage']['issues'])


def test_restore_writes_original_reports_cases_without_paths_from_sources(tmp_path):
    suite, expected = make_suite(tmp_path, profiles=('prefix',), bs=(1,))
    manifest = suite_transfer.export_suite(suite)
    payload = group_payloads(suite, manifest)[0]
    output = tmp_path / 'restored'
    suite_transfer.restore_payload(payload, output, render=False)
    assert json.loads((output / 'cases/000/decoded_samples.json').read_text()) == [dict(metadata=m, decoded=d) for m, d in expected[0][3]]
    assert (output / 'reports/PERFORMANCE_COMPARE.txt').read_text() == 'OFF and ON full tasks: exact comparisons\n'
    assert json.loads((output / 'suite.json').read_text())['coverage']['complete'] is True
    with pytest.raises(ValueError, match='already exists'):
        suite_transfer.restore_payload(payload, output, render=False)
    payload['files']['../outside.txt'] = dict(kind='text', data='forbidden')
    bad_output = tmp_path / 'unsafe'
    with pytest.raises(ValueError, match='Unsafe report path'):
        suite_transfer.restore_payload(payload, bad_output, render=False)
    assert not bad_output.exists()


def test_suite_render_keeps_each_profile_and_case_trace_separate(tmp_path):
    suite, _ = make_suite(tmp_path)
    manifest = suite_transfer.export_suite(suite)
    payload = group_payloads(suite, manifest)[0]
    output = tmp_path / 'rendered'
    suite_transfer.restore_payload(payload, output)
    traces = [json.loads(path.read_text()) for path in sorted(output.glob('cases/*/trace.json'))]
    assert len(traces) == 4
    assert [trace['metadata']['timer_profile'] for trace in traces] == ['prefix', 'prefix', 'send', 'send']
    assert all(len([event for event in trace['traceEvents'] if event['ph'] == 'X']) == 2 for trace in traces)


def test_many_parts_bound_manifest_indexes_and_overview_too(tmp_path):
    suite, _ = make_suite(tmp_path, profiles=('prefix',), bs=(1,), noise_bytes=130000)
    manifest = suite_transfer.export_suite(suite, limit=4096)
    assert manifest['part_count'] > 20
    assert len(manifest['part_index']) > 1
    directory = suite / manifest['output']
    overview = (directory / '00_overview.txt').read_text()
    assert '{001..' in overview and 'every ordinal required' in overview
    for path in directory.iterdir():
        assert len(path.read_bytes()) <= 4096
    assert group_payloads(suite, manifest)[0]['coverage']['complete'] is True


def test_group_restore_rejects_missing_mixed_or_corrupt_groups_before_writing(tmp_path, monkeypatch):
    suite, _ = make_suite(tmp_path)
    monkeypatch.setattr(suite_transfer, 'GROUP_BYTES', 1000)
    manifest = suite_transfer.export_suite(suite)
    payloads = group_payloads(suite, manifest)
    def verified(payload):
        raw = suite_transfer._json(payload).encode()
        return dict(complete=True, raw_sha256=hashlib.sha256(raw).hexdigest()), raw
    groups = [verified(payload) for payload in payloads]
    missing = tmp_path / 'missing'
    with pytest.raises(ValueError, match='Missing or extra'):
        suite_transfer.restore_groups(groups[1:], missing, render=False)
    assert not missing.exists()
    bad = json.loads(json.dumps(payloads[0]))
    bad['suite_name'] = 'different-experiment'
    mixed = tmp_path / 'mixed'
    with pytest.raises(ValueError, match='hash mismatch'):
        suite_transfer.restore_groups([verified(bad), *groups[1:]], mixed, render=False)
    assert not mixed.exists()
    output = tmp_path / 'complete'
    suite_transfer.restore_groups(groups[::-1], output, render=False)
    index = json.loads((output / 'suite_index.json').read_text())
    assert index['group_count'] == len(groups)
    assert len(list(output.glob('group_*/cases/*/decoded_samples.json'))) == 4


def test_common_cli_restores_multiple_pasted_suite_bundles(tmp_path, monkeypatch):
    suite, _ = make_suite(tmp_path)
    monkeypatch.setattr(suite_transfer, 'GROUP_BYTES', 1000)
    manifest = suite_transfer.export_suite(suite)
    directory = suite / manifest['output']
    parts = sorted(directory.glob('suite_*_part_*.txt'))
    pasted = tmp_path / 'pasted.txt'
    pasted.write_text('\n'.join(path.read_text() for path in parts[::-1]))
    output = tmp_path / 'cli-restored'
    transfer = module('transfer_report')
    assert transfer.main(['restore', str(pasted), '--output-dir', str(output), '--no-render']) == 0
    assert json.loads((output / 'suite_index.json').read_text())['group_count'] == 4


def test_render_failure_in_last_streamed_group_leaves_no_output(tmp_path, monkeypatch):
    suite, _ = make_suite(tmp_path)
    monkeypatch.setattr(suite_transfer, 'GROUP_BYTES', 1000)
    manifest = suite_transfer.export_suite(suite)
    payloads = group_payloads(suite, manifest)
    transfer, codec = module('transfer_report'), module('text_codec')
    final_case = payloads[-1]['cases'][-1]
    context, samples = transfer.decode_samples(json.dumps(final_case['payload']).encode())
    samples[0][1]['intervals'][0]['last_tick'] += 1
    final_case['payload'] = json.loads(transfer.encode_samples(context, samples))
    block_groups = [codec.make_parts(suite_transfer._json(payload).encode())[1] for payload in payloads]
    output = tmp_path / 'malformed-final'
    with pytest.raises(ValueError, match='actual interval duration'):
        suite_transfer.restore_text_groups(block_groups, output)
    assert not output.exists()


def test_streamed_restore_renders_all_cases_after_validation(tmp_path, monkeypatch):
    suite, _ = make_suite(tmp_path)
    monkeypatch.setattr(suite_transfer, 'GROUP_BYTES', 1000)
    manifest = suite_transfer.export_suite(suite)
    directory = suite / manifest['output']
    codec = module('text_codec')
    blocks = codec.group_parts([path.read_text() for path in sorted(directory.glob('suite_*_part_*.txt'))[::-1]])
    output = tmp_path / 'streamed-render'
    suite_transfer.restore_text_groups(blocks, output)
    assert len(list(output.glob('group_*/cases/*/trace.json'))) == 4
    assert (output / 'suite_index.json').is_file()


def test_failed_case_write_never_publishes_suite_completion_marker(tmp_path, monkeypatch):
    suite, _ = make_suite(tmp_path, profiles=('prefix',), bs=(1,))
    manifest = suite_transfer.export_suite(suite)
    payload = group_payloads(suite, manifest)[0]
    transfer = module('transfer_report')
    def fail(*args, **kwargs):
        raise OSError('simulated output device full')
    monkeypatch.setattr(transfer, 'write_restored', fail)
    original = suite_transfer._module
    monkeypatch.setattr(suite_transfer, '_module', lambda name: transfer if name == 'transfer_report' else original(name))
    output = tmp_path / 'failed-write'
    with pytest.raises(OSError, match='device full'):
        suite_transfer.restore_payload(payload, output, render=False)
    assert not (output / 'suite.json').exists()


def test_show_prints_only_one_verified_suite_part(tmp_path,monkeypatch,capsys):
    suite,_=make_suite(tmp_path)
    monkeypatch.setattr(suite_transfer,'GROUP_BYTES',1000)
    manifest=suite_transfer.export_suite(suite)
    directory=suite/manifest['output'];transfer=module('transfer_report');codec=module('text_codec')
    with pytest.raises(SystemExit): transfer.main(['show',str(directory),'--part','1'])
    capsys.readouterr()
    assert transfer.main(['show',str(directory),'--group','2','--part','1'])==0
    printed=capsys.readouterr().out
    status,_=codec.inspect_parts([printed])
    assert printed.startswith(codec.BEGIN) and printed.endswith(codec.END+'\n')
    assert status['raw_sha256']==manifest['groups'][1]['payload_sha256']
    assert len(printed.replace('\n','\r\n').encode())<=60000


def test_suite_all_files_fit_crlf_at_small_limit(tmp_path):
    suite,_=make_suite(tmp_path)
    manifest=suite_transfer.export_suite(suite,limit=4096)
    for path in (suite/manifest['output']).iterdir():
        text=path.read_text(encoding='utf-8')
        assert len(text)<=4096 and len(text.replace('\n','\r\n').encode('utf-8'))<=4096
    assert manifest['transfer_bytes_crlf']>=manifest['transfer_bytes']
