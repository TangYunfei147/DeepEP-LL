"""Transfer bounds, complete reconstruction and absence/identity guarantees."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('ll_transfer_report', Path(__file__).with_name('transfer_report.py'))
transfer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transfer)


def fixture_samples(count=2, rows=6, unicode_metadata=False):
    samples = []
    for iteration in range(count):
        records = []
        intervals = []
        for core in range(2):
            for i in range(rows):
                stage = 'KERNEL' if i == 0 else ('PREFIX_COMPLETE' if i == 1 else f'STAGE_{i}')
                records.append(dict(core=core, role='send_and_recv', initial_role='send_and_recv',
                                    stage=stage, total_ticks=1000 if i == 0 else 10+core+i,
                                    calls=3 if i else 1, time_measured=True))
                intervals.append(dict(core=core, stage=stage, sample=0,
                                      first_tick=100, last_tick=1100 if i == 0 else 110+core+i,
                                      total_ticks=1000 if i == 0 else 10+core+i))
            records.append(dict(core=core, role='send_and_recv', stage='TX_PACKETS' if core == 0 else 'RX_PACKETS',
                                total_ticks=0, calls=7, time_measured=False))
        meta = dict(rank=0, iteration=iteration, config=dict(B=1, M=32), revision='abc123',
                    topology='目标A5/通路' if unicode_metadata else 'A5')
        decoded = dict(schema=13, operation='dispatch', active_aiv=2, send_cores=1,
                       boundary_mode='observe', frequency_hz=1000000000,
                       records=records, intervals=intervals,
                       cores=[dict(core=c, omitted_intervals=rows) for c in range(2)])
        samples.append((meta, decoded))
    report = dict(world_size=1, schema=13, sampling=dict(diagnostic_iters=count),
                  configurations=dict(dispatch=dict(B=1, M=32)), communication=dict(derived='excluded'))
    return report, samples


def read_series(directory, series):
    return [record for path in sorted(directory.glob(f'{series}_*.txt'))
            for record in json.loads(path.read_text(encoding='utf-8'))['records']]


def test_lossless_unicode_multiple_parts_and_bounds(tmp_path):
    report, samples = fixture_samples(count=4, rows=60, unicode_metadata=True)
    manifest = transfer.export_bundle(tmp_path, report, samples, limit=6000)
    directory = tmp_path/'transfer'
    assert manifest['series_counts']['full'] > 1
    assert manifest['export_complete'] is True
    assert manifest['source_sample_coverage']['dispatch']['complete'] is True
    for path in directory.iterdir():
        raw = path.read_bytes()
        assert len(raw) <= 6000
        assert len(raw.decode('utf-8')) <= 6000
        if path.suffix != '.md':
            json.loads(raw)
            assert raw.isascii()
    context, restored = transfer.restore_samples(directory)
    assert restored == samples
    assert context['configurations'] == report['configurations']
    assert restored[0][0]['topology'] == '目标A5/通路'
    assert restored[0][1]['cores'][0]['omitted_intervals'] == 60


def test_missing_stages_not_imputed_and_activity_not_header(tmp_path):
    report, samples = fixture_samples(count=1)
    samples[0][1]['records'][0]['optional_value'] = None
    # Preserve both explicit null and absent keys through the table representation.
    transfer.export_bundle(tmp_path, report, samples)
    _, restored = transfer.restore_samples(tmp_path/'transfer')
    assert restored == samples
    assert restored[0][1]['records'][0]['optional_value'] is None
    assert 'optional_value' not in restored[0][1]['records'][1]
    data = read_series(tmp_path/'transfer', 'priority')
    summary = next(r['data'] for r in data if r['kind'] == 'sample_summary')
    assert summary['observed_activity'] == dict(tx_observed=1, rx_observed=1)
    assert 'WAIT_RECV_COUNTS' not in summary['prefix_stages']
    assert summary['counters']['RELAY_PACKETS'] == dict(observed_cores=0, absent_cores=2)
    assert 'entry_to_first_send_ticks' not in summary['preparation']


def test_metadata_too_large_fails_before_writing(tmp_path):
    report, samples = fixture_samples(count=1)
    samples[0][0]['oversized'] = '长'*5000
    with pytest.raises(ValueError, match='one record exceeds'):
        transfer.export_bundle(tmp_path, report, samples, limit=4096)
    assert not (tmp_path/'transfer').exists()


@pytest.mark.parametrize('limit', [0, 4095, 65536, 60000.0, True])
def test_illegal_limits(limit, tmp_path):
    report, samples = fixture_samples(count=1)
    with pytest.raises(ValueError, match='limit'):
        transfer.export_bundle(tmp_path, report, samples, limit=limit)


def test_exact_maximum_bound_and_indivisible_record(tmp_path):
    report, samples = fixture_samples(count=1)
    samples[0][0]['long'] = 'x'*62000
    manifest = transfer.export_bundle(tmp_path, report, samples, limit=65535)
    assert manifest['limit'] == 65535
    assert transfer.restore_samples(tmp_path/'transfer')[1] == samples
    assert all(len(p.read_bytes()) <= 65535 for p in (tmp_path/'transfer').iterdir())


def test_corruption_missing_and_duplicate_parts_rejected(tmp_path):
    report, samples = fixture_samples(count=2, rows=80)
    transfer.export_bundle(tmp_path, report, samples, limit=6000)
    directory = tmp_path/'transfer'
    part = sorted(directory.glob('full_*.txt'))[0]
    original = part.read_bytes()
    part.write_bytes(original.replace(b'KERNEL', b'BROKEN', 1))
    with pytest.raises(ValueError, match='hash'):
        transfer.restore_samples(directory)
    part.write_bytes(original)
    part.unlink()
    with pytest.raises(FileNotFoundError):
        transfer.restore_samples(directory)


def test_source_completeness_separate_from_export(tmp_path):
    report, samples = fixture_samples(count=2)
    report['world_size'] = 2
    manifest = transfer.export_bundle(tmp_path, report, samples)
    assert manifest['export_complete'] is True
    assert manifest['source_sample_coverage']['dispatch']['complete'] is False
    assert manifest['all_observed_kernel_cores_complete'] is True
    # Duplicates are retained losslessly but explicitly do not form a complete source.
    report, samples = fixture_samples(count=1)
    samples.append(copy.deepcopy(samples[0]))
    manifest = transfer.export_bundle(tmp_path/'duplicates', report, samples)
    assert manifest['source_sample_coverage']['dispatch']['complete'] is False


def test_schema14_new_prefix_names_and_unknown_fields_retained(tmp_path):
    report, samples = fixture_samples(count=1)
    decoded = samples[0][1]
    decoded['schema'] = 14
    decoded['extra_parser_metadata'] = dict(any_field='kept')
    decoded['records'].append(dict(core=1, stage='PREFIX_READY_WAIT', calls=1,
                                   total_ticks=9321, time_measured=True))
    decoded['records'].append(dict(core=1, stage='PREFIX_TOTAL_WAIT', calls=1,
                                   total_ticks=4321, time_measured=True))
    transfer.export_bundle(tmp_path, report, samples)
    assert transfer.restore_samples(tmp_path/'transfer')[1] == samples
    data = read_series(tmp_path/'transfer', 'priority')
    summary = next(r['data'] for r in data if r['kind'] == 'sample_summary')
    assert summary['prefix_stages']['PREFIX_READY_WAIT']['maximum'] == 9321
    assert summary['prefix_stages']['PREFIX_TOTAL_WAIT']['maximum'] == 4321
    assert 'total_prefix_cost' not in summary


def test_same_input_stable_bundle_and_existing_other_bundle_rejected(tmp_path):
    report, samples = fixture_samples(count=1)
    first = transfer.export_bundle(tmp_path, report, samples)
    assert transfer.export_bundle(tmp_path, report, samples)['bundle_id'] == first['bundle_id']
    samples[0][0]['revision'] = 'new'
    with pytest.raises(ValueError, match='different bundle'):
        transfer.export_bundle(tmp_path, report, samples)


def test_large_part_index_is_itself_split_and_reconstructs(tmp_path):
    report, samples = fixture_samples(count=32, rows=128)
    manifest = transfer.export_bundle(tmp_path, report, samples, limit=6000)
    assert manifest['series_counts']['full'] > 100
    assert len(manifest['part_index']) > 1
    assert all(len(p.read_bytes()) <= 6000 for p in (tmp_path/'transfer').iterdir())
    assert transfer.restore_samples(tmp_path/'transfer')[1] == samples


def test_schema13_real_parser_records_round_trip(tmp_path):
    # Reuse the actual schema13 parser fixture, including counter-only rows and
    # core metadata describing interval omission; this needs no device runtime.
    source = Path(__file__).with_name('test_communication_report.py')
    spec = importlib.util.spec_from_file_location('transfer_parser_fixture', source)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    decoded = fixture.timer.parse_records(fixture.raw(), 'dispatch', 1e9)
    meta = dict(rank=0, iteration=0, config=dict(B=1, M=32), revision='schema13-old-data')
    samples = [(meta, decoded)]
    report = dict(world_size=1, sampling=dict(diagnostic_iters=1), schema=13)
    manifest = transfer.export_bundle(tmp_path, report, samples)
    assert manifest['stage_rows'] == len(decoded['records'])
    assert manifest['interval_rows'] == len(decoded['intervals'])
    assert transfer.restore_samples(tmp_path/'transfer')[1] == samples


def test_different_rank_profiles_are_not_silently_aggregated(tmp_path):
    report, samples = fixture_samples(count=2)
    samples[0][1].update(schema=14, timer_profile='prefix', timer_profile_id=3)
    samples[1][1].update(schema=14, timer_profile='recv', timer_profile_id=5)
    samples[1][0].update(rank=1, iteration=0)
    with pytest.raises(ValueError, match='Mixed timer profiles'):
        transfer.export_bundle(tmp_path, report, samples)
    assert not (tmp_path/'transfer').exists()
