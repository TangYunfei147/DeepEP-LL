"""Explicit small follow-up exports preserve raw values and never silently sample."""
import copy
import importlib.util
from pathlib import Path
import random

import pytest


spec = importlib.util.spec_from_file_location('ll_triage_extract', Path(__file__).with_name('triage_extract.py'))
extract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extract)
transfer = extract._module('transfer_report')


def fixture(suite, profile='prefix', bs=1, case='case_00000', noise=0):
    samples = []
    for rank in (0, 3):
        for iteration in (0, 1):
            records, intervals, cores = [], [], []
            for core in (0, 28, 55):
                records.append(dict(core=core, stage='KERNEL', total_ticks=2**60 + 37 + core,
                                    calls=1, time_measured=True, absent_test=None))
                records.append(dict(core=core, stage='PREFIX_READY_WAIT', total_ticks=123 + rank,
                                    calls=2, time_measured=True, future_field={'exact': [1, None]}))
                intervals.append(dict(core=core, stage='PREFIX_READY_WAIT', first_tick=2**60 + 15,
                                      last_tick=2**60 + 119, sample=0, total_ticks=104))
                cores.append(dict(core=core, omitted_intervals=17, future='原值'))
            metadata = dict(rank=rank, iteration=iteration, config=dict(B=bs, M=32), source_revision='commit')
            decoded = dict(schema=14, timer_profile=profile, timer_profile_id=3, active_aiv=56,
                           send_cores=28, recv_cores=28, operation='dispatch', future_identity='unchanged',
                           records=records, intervals=intervals, cores=cores)
            samples.append((metadata, decoded))
    if noise:
        samples[0][0]['exact_noise'] = random.Random(931).randbytes(noise).hex()
    context = dict(world_size=16, sampling=dict(diagnostic_iters=5), configurations=dict(dispatch=dict(B=bs, M=32)),
                   implementation='g4-small-b-parallel-prefix')
    destination = suite / 'analysis' / profile / case
    transfer.export_bundle(destination, context, samples)
    return destination, context, samples


def test_exact_selection_preserves_core_labels_launch_identity_and_values(tmp_path):
    suite = tmp_path / 'suite'
    _, _, source = fixture(suite)
    fixture(suite, bs=2, case='case_00001')
    fixture(suite, profile='recv')
    output = tmp_path / 'selected'
    manifest = extract.export_selection(suite, 'prefix', 1, output, rank=3, iteration=1, core=55)
    context, restored = transfer.restore_samples(output)
    expected = copy.deepcopy([source[-1]])
    for key in ('records', 'intervals', 'cores'):
        expected[0][1][key] = [row for row in expected[0][1][key] if row['core'] == 55]
    assert restored == expected
    assert restored[0][1]['active_aiv'] == 56
    assert restored[0][1]['send_cores'] == restored[0][1]['recv_cores'] == 28
    assert context['world_size'] == 16
    selected = context['export_selection']
    assert selected['explicit_subset'] and selected['complete_archive'] is False
    assert selected['selectors'] == dict(profile='prefix', B=1, rank=3, iteration=1, core=55,
                                         case='analysis/prefix/case_00000')
    assert manifest['complete_archive'] is False
    assert manifest['all_observed_kernel_cores_complete'] is False
    assert manifest['export_complete'] is True  # Complete within explicit selection.
    assert len(manifest['parts']) <= 3
    for path in output.iterdir():
        raw = path.read_bytes()
        text = raw.decode('utf-8')
        assert len(text) <= 60000 and len(raw) + text.count('\n') <= 60000


def test_null_selectors_preserve_every_case_sample_and_all_record_fields(tmp_path):
    suite = tmp_path / 'suite'
    _, _, source = fixture(suite)
    output = tmp_path / 'all-selected'
    extract.export_selection(suite, 'prefix', 1, output)
    context, restored = transfer.restore_samples(output)
    assert restored == source
    assert context['export_selection']['selectors']['rank'] is None
    assert context['export_selection']['complete_archive'] is False


@pytest.mark.parametrize('selector', [dict(bs=99), dict(rank=9), dict(iteration=99), dict(core=54)])
def test_missing_selection_rejected_before_directory_creation(tmp_path, selector):
    suite = tmp_path / 'suite'
    fixture(suite)
    options = dict(bs=1, **{k: v for k, v in selector.items() if k != 'bs'})
    if 'bs' in selector:
        options['bs'] = selector['bs']
    output = tmp_path / 'must-not-exist'
    with pytest.raises(ValueError, match='No samples|Core 54 is absent'):
        extract.export_selection(suite, 'prefix', output_dir=output, **options)
    assert not output.exists()


def test_existing_output_is_not_overwritten_or_read(tmp_path, monkeypatch):
    output = tmp_path / 'existing'
    output.mkdir()
    (output / 'keep.txt').write_text('original')
    monkeypatch.setattr(extract, '_module', lambda _: pytest.fail('Existing output must fail before source restore'))
    with pytest.raises(ValueError, match='already exists'):
        extract.export_selection(tmp_path / 'missing-suite', 'prefix', 1, output)
    assert (output / 'keep.txt').read_text() == 'original'
    assert len(list(output.iterdir())) == 1


def test_distinct_routes_require_explicit_case_even_with_rank_filter(tmp_path):
    suite = tmp_path / 'suite'
    destination, _, _ = fixture(suite, case='case_00000/route_a')
    fixture(suite, case='case_00000/route_b')
    output = tmp_path / 'one-route'
    with pytest.raises(ValueError, match='Multiple cases/routes match; supply --case'):
        extract.export_selection(suite, 'prefix', 1, output, rank=3)
    assert not output.exists()
    manifest = extract.export_selection(suite, 'prefix', 1, output, rank=3,
                                        case=destination.relative_to(suite))
    assert manifest['samples'] == 2
    assert manifest['export_selection']['selectors']['case'].endswith('/route_a')


def test_case_selector_must_stay_in_requested_profile(tmp_path):
    suite = tmp_path / 'suite'
    fixture(suite)
    with pytest.raises(ValueError, match='inside analysis/prefix'):
        extract.export_selection(suite, 'prefix', 1, tmp_path / 'out', case='../outside')
    assert not (tmp_path / 'out').exists()


def test_default_three_part_cap_rejects_exact_data_without_writing_then_explicit_limit_works(tmp_path):
    suite = tmp_path / 'suite'
    _, _, samples = fixture(suite, noise=180000)
    output = tmp_path / 'too-large'
    with pytest.raises(ValueError, match=r'exceeding --max-parts 3.*No files were written'):
        extract.export_selection(suite, 'prefix', 1, output)
    assert not output.exists()
    manifest = extract.export_selection(suite, 'prefix', 1, output, max_parts=10)
    assert 3 < len(manifest['parts']) <= 10
    assert transfer.restore_samples(output)[1] == samples


@pytest.mark.parametrize('max_parts', [0, 11, 3.0, True])
def test_illegal_part_limits_rejected_before_reading(tmp_path, max_parts):
    with pytest.raises(ValueError, match='integer in 1..10'):
        extract.export_selection(tmp_path / 'missing', 'prefix', 1, tmp_path / 'out', max_parts=max_parts)
    assert not (tmp_path / 'out').exists()


def test_cli_output_parts_restore_without_manifest(tmp_path, capsys):
    suite = tmp_path / 'suite'
    fixture(suite)
    output = tmp_path / 'extract'
    assert extract.main([str(suite), '--profile', 'prefix', '--bs', '1', '--rank', '3',
                         '--iteration', '1', '--core', '28', '--output-dir', str(output)]) == 0
    message = capsys.readouterr().out
    assert 'Explicit subset: 1 samples; send exactly 1 part files:' in message
    text = tmp_path / 'pasted.txt'
    text.write_text(''.join(path.read_text() for path in sorted(output.glob('part_*.txt'))))
    restored = tmp_path / 'restored'
    assert transfer.main(['restore', str(text), '--output-dir', str(restored), '--no-render']) == 0
    assert (restored / 'decoded_samples.json').is_file()
    assert '"complete_archive":false' in (restored / 'context.json').read_text()


def test_selected_rank_core_restores_trace_without_claiming_full_domain(tmp_path):
    suite=tmp_path/'suite'
    destination,context,samples=fixture(suite)
    for meta,decoded in samples:
        meta.update(world_size=16,correctness='passed')
        decoded['frequency_hz']=1e9
    # Fresh source fixture with the real complete-identity fields.
    source=destination.parent/'case_render'
    transfer.export_bundle(source,context,samples)
    output=tmp_path/'subset'
    extract.export_selection(suite,'prefix',1,output,rank=3,iteration=1,core=55,
                             case=source.relative_to(suite).as_posix())
    selected_context,selected_samples=transfer.restore_samples(output)
    restored=tmp_path/'restored'
    transfer.write_restored(restored,selected_context,selected_samples,trace_iteration=1)
    import json
    assert json.loads((restored/'analysis_status.json').read_text())['status']=='explicit_subset'
    trace=json.loads((restored/'trace.json').read_text())
    assert trace['metadata']['export_selection']['complete_archive'] is False
    assert all(e['args']['core']==55 for e in trace['traceEvents'] if e['ph']=='X')
    assert not (restored/'summary.json').exists()
