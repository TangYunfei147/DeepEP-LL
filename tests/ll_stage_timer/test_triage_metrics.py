"""Triage uses all captures, same-core clocks and explicit missing evidence."""
import importlib.util
import json
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location('triage_metrics_tested', Path(__file__).with_name('triage_metrics.py'))
triage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(triage)


def sample(iteration=0, rank=0, *, hz=1e9, profile='prepare', send=8, cores=12,
           kernel=1000, init=100, active_send=7, init_slow_core=None):
    records, intervals = [], []
    def duration(core, stage, begin, end, calls=1):
        records.append(dict(core=core, stage=stage, total_ticks=end-begin, calls=calls, time_measured=True))
        intervals.append(dict(core=core, stage=stage, first_tick=begin, last_tick=end, sample=0))
    def counter(core, stage, count):
        records.append(dict(core=core, stage=stage, total_ticks=0, calls=count, time_measured=False))
    for core in range(cores):
        # The rank-dependent clock origin must not affect same-core durations.
        origin = rank*10**12
        local_init = init*20 if core == init_slow_core else init
        duration(core, 'INIT', origin+100, origin+100+local_init)
        duration(core, 'TIMER_SETUP', origin+80, origin+100)
        begin = origin+100+local_init
        duration(core, 'KERNEL', begin, begin+kernel+core)
        duration(core, 'INIT_LAYOUT', origin+100, origin+100+local_init//2)
        if core < active_send:
            duration(core, 'BEFORE_FIRST_SEND', begin, begin+40)
            duration(core, 'TOKEN_METADATA', begin+10, begin+20, calls=4)
            counter(core, 'TX_PACKETS', core+1)
        if core >= send:
            duration(core, 'BEFORE_FIRST_RECV_POLL', begin, begin+60)
            duration(core, 'RX_METADATA', begin+60, begin+90, calls=2)
            counter(core, 'RX_PACKETS', core-send+1)
            counter(core, 'RX_PROBED', (core-send+1)*3)
            counter(core, 'RX_READY', core-send+1)
    meta = dict(rank=rank, iteration=iteration, world_size=2, correctness='passed',
        config=dict(B=1, M=32, H=3584, num_topk=8, num_experts=384, dtype='fp8'),
        sampling=dict(diagnostic_iters=5))
    decoded = dict(operation='dispatch', schema=14, active_aiv=cores, send_cores=send,
        recv_cores=cores-send, frequency_hz=hz, timer_profile=profile, boundary_mode='observe',
        records=records, intervals=intervals)
    return meta, decoded


def latency(result, name, role):
    return next(stats for label, group, stats in result['latencies'] if label == name and group == role)


def activity(result, name, role):
    return next(stats for group, quantity, stats in result['activity'] if quantity == name and group == role)


def stage(result, name, role):
    return next(row for row in result['stages'] if row[:2] == [name, role])


def test_all_five_rounds_and_both_ranks_contribute():
    samples = [sample(i, r, kernel=1000 if i < 4 else 10000) for r in range(2) for i in range(5)]
    result = triage.summarize_case({}, samples)
    assert result['coverage']['samples'] == 10
    assert result['coverage']['complete_rank_iterations'] is True
    assert result['coverage']['complete_kernel_cores'] is True
    stats = latency(result, 'KERNEL', 'R')
    assert stats == [10, 1.011, 10.011, 10.011]
    assert stage(result, 'INIT_LAYOUT', 'S')[2][0] == 80
    assert stage(result, 'INIT_LAYOUT', 'S')[3][0] == 10
    assert result['slowest']['R']['iteration'] == 4
    # A far larger rank clock origin did not become elapsed time.
    assert result['slowest']['R']['us'] == pytest.approx(10.111)


def test_seven_observed_send_cores_and_packet_balance():
    result = triage.summarize_case({}, [sample()])
    assert activity(result, 'tx_active_observed', 'S') == [1, 7, 7, 7]
    assert activity(result, 'tx_counter_unknown_cores', 'S') == [1, 1, 1, 1]
    assert activity(result, 'tx_positive_min', 'S') == [1, 1, 1, 1]
    assert activity(result, 'tx_positive_median', 'S') == [1, 4, 4, 4]
    assert activity(result, 'tx_positive_max', 'S') == [1, 7, 7, 7]
    assert activity(result, 'rx_active_observed', 'R') == [1, 4, 4, 4]
    assert activity(result, 'probed_per_ready', 'R') == [1, 3, 3, 3]
    assert activity(result, 'ready_fraction', 'R') == [1, .333333, .333333, .333333]
    assert activity(result, 'tx_active_observed', 'R') == [0, None, None, None]


def test_missing_fields_are_not_zero_and_unknown_frequency_not_one_ghz():
    meta, decoded = sample(hz=None)
    decoded['records'] = [row for row in decoded['records'] if row['stage'] not in ('RX_READY', 'INIT_LAYOUT')]
    decoded['intervals'] = [row for row in decoded['intervals'] if row['stage'] not in ('INIT', 'INIT_LAYOUT')]
    result = triage.summarize_case({}, [(meta, decoded)])
    assert not result['latencies']
    assert all(row[0] != 'INIT_LAYOUT' for row in result['stages'])
    assert stage(result, 'KERNEL', 'S')[2] == [0, None, None, None]
    assert activity(result, 'ready_fraction', 'R') == [0, None, None, None]
    assert activity(result, 'probe_cores_without_ready_counter', 'R') == [1, 4, 4, 4]
    assert result['slowest']['S']['span_kind'] == 'KERNEL'
    assert result['slowest']['S']['us'] is None
    assert result['slowest']['S']['ticks'] == 1007


def test_non_ghz_units_and_init_dominant_slowest_core():
    result = triage.summarize_case({}, [sample(hz=2e8, init_slow_core=2)])
    assert result['slowest']['S']['core'] == 2
    assert result['slowest']['S']['span_kind'] == 'entry_to_kernel_end'
    assert result['slowest']['S']['us'] == pytest.approx(15.01)
    assert latency(result, 'KERNEL', 'S')[1] == pytest.approx(5.035)
    assert latency(result, 'INIT', 'S')[1] == 10
    assert latency(result, 'entry_to_first_send', 'S')[1] == 10.2
    assert result['slowest']['S']['top_stages'][0][0] == 'INIT'
    assert len(result['slowest']['S']['top_stages']) <= 12


def test_unmatched_counter_totals_do_not_produce_false_ratio():
    meta, decoded = sample()
    decoded['records'] = [row for row in decoded['records'] if not (row['core'] == 8 and row['stage'] == 'RX_PROBED')]
    result = triage.summarize_case({}, [(meta, decoded)])
    assert activity(result, 'ready_cores_without_probe_counter', 'R') == [1, 1, 1, 1]
    assert activity(result, 'ready_fraction', 'R') == [0, None, None, None]
    assert activity(result, 'probed_observed', 'R') == [1, 27, 27, 27]


def test_initial_roles_do_not_follow_activity_label():
    meta, decoded = sample()
    for row in decoded['records']:
        row['role'] = 'send_and_recv'
    # Initial S core also receives; it stays in S statistics.
    decoded['records'].append(dict(core=1, stage='RX_PACKETS', calls=8, total_ticks=0, time_measured=False))
    result = triage.summarize_case({}, [(meta, decoded)])
    assert activity(result, 'rx_active_observed', 'S') == [1, 1, 1, 1]
    assert activity(result, 'rx_active_observed', 'R') == [1, 4, 4, 4]
    assert activity(result, 'rx_active_observed', 'all') == [1, 5, 5, 5]
    assert result['slowest']['S']['core'] < 8
    assert result['slowest']['R']['core'] >= 8


@pytest.mark.parametrize('changed', ['profile', 'hz', 'send'])
def test_mixed_profile_frequency_or_split_rejected(changed):
    kwargs = {'profile': 'recv'} if changed == 'profile' else ({'hz': 2e8} if changed == 'hz' else {'send': 6})
    with pytest.raises(ValueError, match='Mixed'):
        triage.summarize_case({}, [sample(0), sample(1, **kwargs)])


def test_missing_core_and_missing_round_coverage():
    meta, decoded = sample()
    decoded['records'] = [row for row in decoded['records'] if row['core'] != 5]
    result = triage.summarize_case({}, [(meta, decoded)])
    assert result['coverage']['complete_rank_iterations'] is False
    assert result['coverage']['complete_kernel_cores'] is False
    assert result['coverage']['kernel_core_count'] == [1, 11, 11, 11]


def test_duplicate_sample_and_negative_entry_span_rejected():
    with pytest.raises(ValueError, match='Duplicate rank/iteration'):
        triage.summarize_case({}, [sample(), sample()])
    meta, decoded = sample()
    next(row for row in decoded['intervals'] if row['core'] == 0 and row['stage'] == 'INIT')['first_tick'] = 100000
    with pytest.raises(ValueError, match='Invalid same-core'):
        triage.summarize_case({}, [(meta, decoded)])


def test_compact_summary_independent_of_capture_count_and_json_safe():
    samples = [sample(i, r, cores=56, send=28) for r in range(16) for i in range(5)]
    result = triage.summarize_case({}, samples)
    encoded = json.dumps(result, separators=(',', ':'), allow_nan=False)
    assert len(encoded.encode('utf-8')) < 15000
    assert result['coverage']['samples'] == 80
    assert stage(result, 'KERNEL', 'S')[2][0] == 28*80
    assert stage(result, 'KERNEL', 'S')[3][0] == 80


def test_context_config_and_sampling_are_used_when_sample_metadata_minimal():
    meta, decoded = sample()
    meta.pop('config'); meta.pop('sampling'); meta.pop('world_size')
    context = dict(world_size=1, sampling=dict(diagnostic_iters=1),
                   configurations=dict(dispatch=dict(B=32, M=32, H=3584, num_topk=8, num_experts=384, dtype='fp8')))
    result = triage.summarize_case(context, [(meta, decoded)])
    assert result['identity']['B'] == 32
    assert result['identity']['K'] == 8
    assert result['coverage']['complete_rank_iterations'] is True
