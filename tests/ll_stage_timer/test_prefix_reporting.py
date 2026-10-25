"""Recorded prefix configuration and same-core sparse preparation spans."""
import copy
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT/relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sparse = load('prefix_sparse_fixtures', 'tests/ll_stage_timer/test_sparse_timer.py')
triage = load('prefix_triage', 'tests/ll_stage_timer/triage_metrics.py')
parity = load('prefix_parity_fixtures', 'tests/ll_stage_timer/test_parity.py')
compare = parity.comparison


def captured(raw_p=6, effective=26):
    meta = sparse.metadata()
    meta['comparison'] = dict(small_b_prefix_cores=raw_p,
                              effective_small_b_prefix_cores=effective)
    raw = sparse.raw()
    decoded = sparse.timer.parse_records(raw, 'dispatch', 2e8)
    return meta, raw, decoded


def test_prefix_identity_survives_analysis_raw_and_restored_trace():
    meta, raw, decoded = captured()
    analyzer = load('prefix_analyzer', 'tests/ll_stage_timer/analyze.py')
    report = analyzer.analyze_samples([(meta, decoded)])
    assert report['small_b_prefix_cores'] == 6
    assert report['effective_small_b_prefix_cores'] == 26
    trace = sparse.timer.chrome_trace([dict(raw=raw, operation='dispatch',
        frequency_hz=2e8, metadata=meta)])
    transfer = load('prefix_transfer_trace', 'tests/ll_stage_timer/transfer_trace.py')
    restored = transfer.trace_from_decoded([(meta, decoded)])
    for output in (trace, restored):
        assert output['metadata']['cores'][0]['small_b_prefix_cores'] == 6
        assert output['metadata']['cores'][0]['effective_small_b_prefix_cores'] == 26
    summary = triage.summarize_case({}, [(meta, decoded)])
    assert summary['identity']['small_b_prefix_cores'] == 6
    assert summary['identity']['effective_small_b_prefix_cores'] == 26


def test_old_capture_prefix_configuration_is_unknown():
    meta, _, decoded = captured()
    del meta['comparison']
    result = triage.summarize_case({}, [(meta, decoded)])
    assert result['identity']['small_b_prefix_cores'] is None
    assert result['identity']['effective_small_b_prefix_cores'] is None


def test_triage_and_both_trace_paths_reject_mixed_recorded_prefix_configuration():
    meta, raw, decoded = captured()
    other = copy.deepcopy(meta)
    other['iteration'] = 1
    other['comparison']['small_b_prefix_cores'] = 1
    with pytest.raises(ValueError, match='Mixed'):
        triage.summarize_case({}, [(meta, decoded), (other, decoded)])
    with pytest.raises(ValueError, match='comparison.small_b_prefix_cores'):
        sparse.timer.chrome_trace([dict(raw=raw, operation='dispatch',frequency_hz=2e8,metadata=m)
                                   for m in (meta, other)])
    # Restored trace selects one iteration across ranks; no cross-rank clock alignment.
    other.update(rank=1,iteration=0,world_size=2)
    meta['world_size'] = 2
    transfer = load('prefix_mixed_transfer', 'tests/ll_stage_timer/transfer_trace.py')
    with pytest.raises(ValueError, match='Mixed'):
        transfer.trace_from_decoded([(meta, decoded), (other, decoded)])


def test_entry_to_prefix_ready_uses_same_core_points_and_saved_frequency():
    meta, data, _ = captured()
    # Move one core's whole clock domain far ahead. Its 450-tick span must not
    # become the distance from a different core's ENTRY to this PREFIX_READY.
    for index, row in enumerate(data[31]):
        for offset, value in enumerate(row):
            word = index*8 + offset
            if 8 <= word < 22 and value:
                sparse.setword(data,31,word,value+10**12)
    decoded = sparse.timer.parse_records(data, 'dispatch', 2e8)
    summary = triage.summarize_case({}, [(meta, decoded)])
    latency = {(name, group): values for name, group, values in summary['latencies']}
    assert latency['entry_to_prefix_ready', 'R'] == [1,2.25,2.25,2.25]
    assert ('entry_to_prefix_ready', 'S') not in latency
    report = load('prefix_summary_renderer', 'tests/ll_stage_timer/triage_report.py')
    text = report.render(dict(suite='prefix',cases=[dict(id='C001',metrics=summary,
        source='analysis/sparse/case',provenance={})],matrix={},off=[],performance=[],issues=[],complete=True))
    assert 'ENTRY->PREFIX_READY' in text
    assert 'P raw/effective' in text and '6/26' in text


def test_missing_or_reversed_prefix_milestone_is_not_invented():
    meta, _, decoded = captured()
    decoded['milestones'] = [p for p in decoded['milestones'] if p['point'] != 'PREFIX_READY']
    result = triage.summarize_case({}, [(meta, decoded)])
    assert not any(name == 'entry_to_prefix_ready' for name, _, _ in result['latencies'])
    decoded['milestones'].append(dict(core=30,point='PREFIX_READY',tick=1))
    with pytest.raises(ValueError, match='same-core entry to prefix-ready'):
        triage.summarize_case({}, [(meta, decoded)])


def stage_reports(raw_left=0, effective_left=26, raw_right=6, effective_right=6):
    left, right = parity.reports()
    for report, raw_p, effective in ((left,raw_left,effective_left),(right,raw_right,effective_right)):
        report['comparison'].update(small_b_prefix_cores=raw_p,
                                    effective_small_b_prefix_cores=effective)
    return left, right


def test_different_prefix_partition_has_no_per_core_stage_ratio():
    left, right = stage_reports()
    row = compare.compare(left,right,'stage')[0]
    assert row['status'] == 'prefix_partition_changed'
    assert row['candidate_over_baseline'] is None
    assert row['baseline_small_b_prefix_cores'] == 0
    assert row['candidate_effective_small_b_prefix_cores'] == 6


@pytest.mark.parametrize('raw_left,effective_left,raw_right,effective_right',
                         [(0,26,26,26),(6,None,1,None),(6,26,1,26)])
def test_same_effective_partition_or_inapplicable_prefix_keeps_stage_ratio(
        raw_left,effective_left,raw_right,effective_right):
    left,right = stage_reports(raw_left,effective_left,raw_right,effective_right)
    row = compare.compare(left,right,'stage')[0]
    assert row['status'] == 'matched'
    assert row['candidate_over_baseline'] == .8


def test_new_versus_old_stage_configuration_remains_unknown():
    left,right = stage_reports()
    left['comparison'].pop('small_b_prefix_cores')
    left['comparison'].pop('effective_small_b_prefix_cores')
    row = compare.compare(left,right,'stage')[0]
    assert row['status'] == 'prefix_configuration_unknown'
    assert row['candidate_over_baseline'] is None


def test_off_whole_kernel_ab_allows_and_reports_prefix_changes():
    left,right = stage_reports()
    def performance(report, duration):
        identity = report['comparison']
        identity.update(stage_timer=False,stage_capture=False,expert_capture=False)
        return dict(status='completed',results=[dict(case_id=0,correctness='passed',
            comparison=identity,dispatch=dict(ranks=[dict(rank=0,mean_us=duration)]))])
    row = compare.compare(performance(left,40),performance(right,30),'perf')[0]
    assert row['status'] == 'matched'
    assert row['candidate_over_baseline'] == .75
    assert row['baseline_effective_small_b_prefix_cores'] == 26
    assert row['candidate_effective_small_b_prefix_cores'] == 6


@pytest.mark.parametrize('field,value', [('small_b_prefix_cores',True),
    ('small_b_prefix_cores',-1),('small_b_prefix_cores',56),
    ('effective_small_b_prefix_cores',0),('effective_small_b_prefix_cores',True)])
def test_invalid_recorded_prefix_configuration_is_rejected(field,value):
    left,right = stage_reports()
    right['comparison'][field] = value
    with pytest.raises(ValueError,match='prefix producer identity'):
        compare.compare(left,right,'stage')
