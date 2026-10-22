"""Explicit small follow-up exports preserve raw values and never silently sample."""
import copy
import hashlib
import importlib.util
import itertools
from pathlib import Path
import random

import pytest


spec = importlib.util.spec_from_file_location('ll_triage_extract', Path(__file__).with_name('triage_extract.py'))
extract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extract)
transfer = extract._module('transfer_report')
budget = extract._module('transfer_budget')


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


def test_optional_core_preserves_every_record_of_the_explicit_rank_and_iteration(tmp_path):
    suite = tmp_path / 'suite'
    _, _, source = fixture(suite)
    output = tmp_path / 'all-selected'
    extract.export_selection(suite, 'prefix', 1, output, rank=0, iteration=1)
    context, restored = transfer.restore_samples(output)
    assert restored == [source[1]]
    assert context['export_selection']['selectors']['rank'] == 0
    assert context['export_selection']['selectors']['iteration'] == 1
    assert context['export_selection']['selectors']['core'] is None
    assert context['export_selection']['complete_archive'] is False


@pytest.mark.parametrize('selector', [dict(bs=99), dict(rank=9), dict(iteration=99), dict(core=54)])
def test_missing_selection_rejected_before_directory_creation(tmp_path, selector):
    suite = tmp_path / 'suite'
    fixture(suite)
    options = dict(bs=1, rank=0, iteration=0)
    options.update(selector)
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
        extract.export_selection(suite, 'prefix', 1, output, rank=3, iteration=0)
    assert not output.exists()
    manifest = extract.export_selection(suite, 'prefix', 1, output, rank=3, iteration=0,
                                        case=destination.relative_to(suite))
    assert manifest['samples'] == 1
    assert manifest['export_selection']['selectors']['case'].endswith('/route_a')


def test_case_selector_must_stay_in_requested_profile(tmp_path):
    suite = tmp_path / 'suite'
    fixture(suite)
    with pytest.raises(ValueError, match='inside analysis/prefix'):
        extract.export_selection(suite, 'prefix', 1, tmp_path / 'out', rank=0, iteration=0,
                                 case='../outside')
    assert not (tmp_path / 'out').exists()


def test_default_three_part_cap_rejects_exact_data_without_writing_then_explicit_limit_works(tmp_path):
    suite = tmp_path / 'suite'
    _, _, samples = fixture(suite, noise=180000)
    output = tmp_path / 'too-large'
    with pytest.raises(ValueError, match=r'exceeding --max-parts 3.*No files were written'):
        extract.export_selection(suite, 'prefix', 1, output, rank=0, iteration=0)
    assert not output.exists()
    manifest = extract.export_selection(suite, 'prefix', 1, output, rank=0, iteration=0, max_parts=9)
    assert 3 < len(manifest['parts']) <= 9
    assert transfer.restore_samples(output)[1] == [samples[0]]
    assert manifest['send_budget']['issued'] == len(manifest['parts'])
    assert manifest['send_budget']['reserved'] == 1


@pytest.mark.parametrize('max_parts', [0, 10, 11, 3.0, True])
def test_illegal_part_limits_rejected_before_reading(tmp_path, max_parts):
    with pytest.raises(ValueError, match=r'integer in 1\.\.9'):
        extract.export_selection(tmp_path / 'missing', 'prefix', 1, tmp_path / 'out',
                                 rank=0, iteration=0, max_parts=max_parts)
    assert not (tmp_path / 'out').exists()


def test_cli_output_parts_restore_without_manifest(tmp_path, capsys):
    suite = tmp_path / 'suite'
    fixture(suite)
    output = tmp_path / 'extract'
    assert extract.main([str(suite), '--profile', 'prefix', '--bs', '1', '--rank', '3',
                         '--iteration', '1', '--core', '28', '--output-dir', str(output)]) == 0
    message = capsys.readouterr().out
    assert 'Explicit subset: 1 samples; send only 1 new part files:' in message
    assert 'Suite send budget: 1/10 issued; 8 remaining; 1 summary slot reserved.' in message
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


@pytest.mark.parametrize('selectors', [{}, {'rank': 0}, {'iteration': 0}])
def test_rank_and_iteration_are_required_before_loading_samples(tmp_path, monkeypatch, selectors):
    monkeypatch.setattr(extract, '_module', lambda _: pytest.fail('Missing selectors must fail before reading'))
    with pytest.raises(ValueError, match='explicit --rank and --iteration'):
        extract.export_selection(tmp_path / 'missing', 'prefix', 1, tmp_path / 'out', **selectors)
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('omitted', ['--rank', '--iteration'])
def test_cli_requires_explicit_rank_and_iteration(tmp_path, capsys, omitted):
    arguments = [str(tmp_path / 'suite'), '--profile', 'prefix', '--bs', '1',
                 '--output-dir', str(tmp_path / 'out')]
    if omitted != '--rank':
        arguments += ['--rank', '0']
    if omitted != '--iteration':
        arguments += ['--iteration', '0']
    with pytest.raises(SystemExit) as error:
        extract.main(arguments)
    assert error.value.code == 2
    assert omitted in capsys.readouterr().err
    assert not (tmp_path / 'out').exists()


def _issue_summary(suite):
    text = b'LL diagnosis initial summary\n'
    digest = hashlib.sha256(text).hexdigest()
    with budget.transaction(suite, 'summary', [digest]):
        (suite / 'SEND_FIRST.txt').write_bytes(text)
    return digest


def test_budget_accumulates_across_batches_and_different_output_directories(tmp_path):
    suite = tmp_path / 'suite'
    fixture(suite)
    selectors = [(0, 0, 0), (3, 1, 28), (0, 1, 55)]
    for number, (rank, iteration, core) in enumerate(selectors, 1):
        output = tmp_path / ('separate-location-' + str(number)) / 'selected'
        manifest = extract.export_selection(suite, 'prefix', 1, output,
                                            rank=rank, iteration=iteration, core=core)
        assert manifest['send_budget']['issued'] == number
        assert manifest['send_budget']['remaining'] == 9 - number
        assert manifest['send_budget']['reserved'] == 1
        assert manifest['send_first'] == manifest['parts']
    assert budget.budget_state(suite) == dict(limit=10, issued=3, reserved=1, remaining=6)
    assert (suite / 'triage/send_budget.json').is_file()
    assert not (suite / 'triage/send_budget.lock').exists()
    assert not list(tmp_path.glob('separate-location-*/selected/send_budget.json'))


def test_duplicate_parts_do_not_charge_or_appear_in_send_first(tmp_path):
    suite = tmp_path / 'suite'
    fixture(suite)
    first = extract.export_selection(suite, 'prefix', 1, tmp_path / 'first', rank=3, iteration=1, core=55)
    state = budget.budget_state(suite)
    second = extract.export_selection(suite, 'prefix', 1, tmp_path / 'second', rank=3, iteration=1, core=55)
    assert second['send_first'] == []
    assert second['parts'] == first['parts']
    assert second['send_budget']['new_count'] == 0
    assert second['send_budget']['already_issued']
    assert second['send_budget']['new_digests'] == []
    assert budget.budget_state(suite) == state
    assert transfer.restore_samples(tmp_path / 'first') == transfer.restore_samples(tmp_path / 'second')
    for name in first['parts']:
        assert (tmp_path / 'first' / name).read_bytes() == (tmp_path / 'second' / name).read_bytes()


def test_summary_and_nine_details_exhaust_suite_budget_then_tenth_detail_writes_nothing(tmp_path):
    suite = tmp_path / 'suite'
    fixture(suite)
    _issue_summary(suite)
    assert budget.budget_state(suite) == dict(limit=10, issued=1, reserved=0, remaining=9)
    selectors = list(itertools.product((0, 3), (0, 1), (0, 28, 55)))
    for number, (rank, iteration, core) in enumerate(selectors[:9], 1):
        manifest = extract.export_selection(suite, 'prefix', 1, tmp_path / f'batch_{number}',
                                            rank=rank, iteration=iteration, core=core)
        assert len(manifest['send_first']) == 1
        assert manifest['send_budget']['issued'] == number + 1
        assert manifest['send_budget']['remaining'] == 9 - number
        assert manifest['send_budget']['reserved'] == 0
    before = (suite / 'triage/send_budget.json').read_bytes()
    output = tmp_path / 'another_directory_cannot_reset_budget'
    rank, iteration, core = selectors[9]
    with pytest.raises(ValueError, match=r'Suite transmission limit is 10.*10 artifacts already issued'):
        extract.export_selection(suite, 'prefix', 1, output, rank=rank, iteration=iteration, core=core)
    assert not output.exists()
    assert (suite / 'triage/send_budget.json').read_bytes() == before
    assert budget.budget_state(suite) == dict(limit=10, issued=10, reserved=0, remaining=0)
    assert not (suite / 'triage/send_budget.lock').exists()
    # Identical content still restores after exhaustion, but is never requested again.
    repeat = extract.export_selection(suite, 'prefix', 1, tmp_path / 'repeat_after_exhaustion',
                                      rank=0, iteration=0, core=0)
    assert repeat['send_first'] == [] and repeat['send_budget']['issued'] == 10


def test_preexisting_summary_without_ledger_consumes_summary_slot(tmp_path):
    suite = tmp_path / 'suite'
    fixture(suite)
    (suite / 'SEND_FIRST.txt').write_bytes(b'Summary created by an earlier producer\n')
    assert not (suite / 'triage/send_budget.json').exists()
    manifest = extract.export_selection(suite, 'prefix', 1, tmp_path / 'first-detail',
                                        rank=0, iteration=0, core=0)
    assert manifest['send_budget']['issued'] == 2
    assert manifest['send_budget']['reserved'] == 0
    assert manifest['send_budget']['remaining'] == 8
    assert budget.budget_state(suite) == dict(limit=10, issued=2, reserved=0, remaining=8)


@pytest.mark.parametrize('fail_on', ['part', 'manifest'])
def test_failed_output_write_does_not_charge_budget(tmp_path, monkeypatch, fail_on):
    suite = tmp_path / 'suite'
    fixture(suite)
    _issue_summary(suite)
    before = budget.budget_state(suite)
    ledger = (suite / 'triage/send_budget.json').read_bytes()
    output = tmp_path / 'failed-output'
    original_write = Path.write_bytes

    def fail_selected_write(path, data):
        selected = path.name.startswith('part_') if fail_on == 'part' else path.name == 'manifest.json'
        if path.parent == output and selected:
            raise OSError('simulated output write failure')
        return original_write(path, data)

    monkeypatch.setattr(Path, 'write_bytes', fail_selected_write)
    with pytest.raises(OSError, match='simulated output write failure'):
        extract.export_selection(suite, 'prefix', 1, output, rank=0, iteration=0, core=0)
    assert budget.budget_state(suite) == before
    assert (suite / 'triage/send_budget.json').read_bytes() == ledger
    assert not (suite / 'triage/send_budget.lock').exists()
    retry = extract.export_selection(suite, 'prefix', 1, tmp_path / 'retry-fresh-output',
                                     rank=0, iteration=0, core=0)
    assert retry['send_budget']['issued'] == before['issued'] + 1


def test_cli_repeated_selection_reports_zero_new_transmissions(tmp_path, capsys):
    suite = tmp_path / 'suite'
    fixture(suite)
    args = [str(suite), '--profile', 'prefix', '--bs', '1', '--rank', '3',
            '--iteration', '1', '--core', '28', '--output-dir']
    assert extract.main(args + [str(tmp_path / 'first')]) == 0
    capsys.readouterr()
    assert extract.main(args + [str(tmp_path / 'same-content-new-directory')]) == 0
    message = capsys.readouterr().out
    assert 'send only 0 new part files:' in message
    assert 'Identical content already issued: do not resend it.' in message
    assert 'Suite send budget: 1/10 issued; 8 remaining; 1 summary slot reserved.' in message
