"""Small first-pass timer summaries, computed from every decoded capture.

This is a lossy diagnostic view, not a trace archive. It never invents events,
aligns rank clocks, combines profiles, or subtracts separately aggregated values.
The original captures remain the source for subsequent, targeted questions.
"""
from collections import defaultdict
import math
import statistics


FORMAT = 'll-triage-case-v1'
STAT_COLUMNS = ['n', 'p50', 'p90', 'max']
STAGE_COLUMNS = ['stage', 'initial_group', 'core_total_us', 'rank_iteration_max_us', 'calls']
COUNTERS = frozenset((
    'POLL_CORE_SUMS_ITERATIONS', 'POLL_RECV_COUNTS_ITERATIONS',
    'WAIT_CUMSUM_ITERATIONS', 'RECV_READY_TOKENS', 'COMBINE_READY_SLOTS',
    'RELAY_PACKETS', 'OVERLAP_PATH', 'SERIAL_FALLBACK', 'SEND_LANES',
    'RECV_LANES', 'TX_PACKETS', 'RX_PROBED', 'RX_READY', 'RX_EMPTY_POLLS', 'RX_PACKETS',
))
LATENCIES = ('TIMER_SETUP', 'INIT', 'KERNEL', 'entry_to_first_send',
             'entry_to_first_probe', 'entry_to_kernel_end')
NOTES = (
    'All supplied samples contribute; statistics are a lossy first-pass summary.',
    'S/R are initial core groups from send_cores; S cores can also receive.',
    'Stage totals contain nested/overlapping parents and children; never add them.',
    'rank_iteration_max is max across observed cores in that group, then summarized across rank/iteration.',
    'Entry spans use two recorded timestamps from the same core/sample. No cross-rank clock alignment.',
    'Missing stages/counters are not zero. Sparse activity counts first-occurrence milestones, never packets or DMA completions.',
    'Profiles are separate experiments. Instrumentation may slow preparation and reduce subsequent waits.',
    'Capture spans do not replace OFF profiler time. Sparse spans exclude post-work export; older formats may also exclude diagnostic synchronization.',
)


def _rounded(value):
    return round(value, 6) if value is not None else None


def _stats(values):
    """Compact [n,p50,p90,max]; linear-interpolation quantiles, no zero fill."""
    values = sorted(value for value in values if value is not None)
    if not values:
        return [0, None, None, None]
    pos = (len(values) - 1) * .9
    low = math.floor(pos)
    high = math.ceil(pos)
    p90 = values[low] + (values[high] - values[low]) * (pos - low)
    return [len(values), _rounded(statistics.median(values)), _rounded(p90), _rounded(values[-1])]


def _pick(mapping, *names, default=None):
    for name in names:
        if mapping.get(name) is not None:
            return mapping[name]
    return default


def _identity(context, meta, decoded):
    operation = decoded.get('operation', 'dispatch')
    config = dict((context.get('configurations') or {}).get(operation) or context.get('config') or {})
    config.update(meta.get('config') or {})
    send = decoded.get('send_cores')
    cores = decoded.get('active_aiv')
    # P is a recorded build/run property, absent from the timer tensor header.
    # In particular, do not turn an old capture's missing P into today's default.
    comparison = meta.get('comparison') or context.get('comparison') or {}
    return dict(
        operation=operation, B=_pick(config, 'B', 'num_tokens'),
        M=_pick(config, 'M', 'num_max_tokens_per_rank'), H=_pick(config, 'H', 'hidden'),
        K=_pick(config, 'num_topk', 'K', 'topk'), E=_pick(config, 'num_experts', 'E', 'experts'),
        dtype=config.get('dtype'), world_size=meta.get('world_size', context.get('world_size')),
        schema=decoded.get('schema'), profile=decoded.get('timer_profile', 'full'),
        send_cores=send, recv_cores=decoded.get('recv_cores', cores-send if cores is not None and send is not None else None),
        small_b_prefix_cores=comparison.get('small_b_prefix_cores'),
        effective_small_b_prefix_cores=comparison.get('effective_small_b_prefix_cores'),
        active_aiv=cores, frequency_hz=decoded.get('frequency_hz', context.get('frequency_hz')),
        boundary_mode=decoded.get('boundary_mode', context.get('boundary_mode', 'observe')),
    )


def _group(core, identity):
    split = identity['send_cores']
    return 'S' if split is not None and core < split else ('R' if split is not None else 'unknown')


def _us(ticks, frequency):
    return ticks * 1e6 / frequency if ticks is not None and frequency else None


def _duration(row, frequency):
    if row is None or row.get('time_measured', True) is False or row.get('stage') in COUNTERS:
        return None
    # Derive units from ticks and this capture's frequency, never an assumed GHz.
    return _us(row.get('total_ticks'), frequency)


def _entry_span(intervals, target):
    init = intervals.get('INIT')
    end = intervals.get(target)
    if not init or not end:
        return None
    start_tick = init.get('first_tick')
    end_tick = end.get('last_tick')
    if start_tick is None or end_tick is None:
        return None
    if end_tick < start_tick:
        raise ValueError('Invalid same-core entry span')
    return end_tick - start_tick


def _counter(rows, name):
    row = rows.get(name)
    value = row.get('calls') if row else None
    if value is not None and (not isinstance(value, (int, float)) or value < 0):
        raise ValueError('Invalid counter value')
    return value


def summarize_case(context, samples):
    """Summarize one operation/profile/shape using every (metadata,decoded) pair.

    Returns JSON-compatible data without I/O. ``stages`` rows and all distribution
    arrays have explicit column headers. Unknown values stay ``None``. Rows are
    only present for measured stages; there is no schema-sized zero-filled table.
    """
    samples = list(samples)
    if not samples:
        raise ValueError('No samples for triage')
    identity = None
    pairs = set()
    ranks, iterations, kernel_cores = set(), set(), set()
    stage_values, stage_calls, stage_maxima = defaultdict(list), defaultdict(list), defaultdict(list)
    latency_maxima = defaultdict(list)
    activity_values = defaultdict(list)
    counter_rows = defaultdict(int)
    kernel_counts = []
    slowest = {}
    correct = defaultdict(int)
    expected_iters = (context.get('sampling') or {}).get('diagnostic_iters')
    for meta, decoded in samples:
        current = _identity(context, meta, decoded)
        if identity is not None and current != identity:
            raise ValueError('Mixed shape, profile, operation, core split, prefix configuration or frequency in triage case')
        identity = current
        frequency = identity['frequency_hz']
        if frequency is not None and (not math.isfinite(frequency) or frequency <= 0):
            raise ValueError('Invalid timer frequency')
        pair = (meta.get('rank'), meta.get('iteration'))
        if pair in pairs:
            raise ValueError('Duplicate rank/iteration in triage case')
        pairs.add(pair)
        ranks.add(pair[0]); iterations.add(pair[1])
        correct[meta.get('correctness', 'unknown')] += 1
        sample_iters = (meta.get('sampling') or {}).get('diagnostic_iters')
        if expected_iters is None:
            expected_iters = sample_iters
        elif sample_iters is not None and sample_iters != expected_iters:
            raise ValueError('Mixed diagnostic iteration counts')
        rows_by_core = defaultdict(dict)
        for row in decoded.get('records', []):
            core, stage = row['core'], row['stage']
            if stage in rows_by_core[core]:
                raise ValueError('Duplicate core/stage row')
            rows_by_core[core][stage] = row
        interval_by_core = defaultdict(dict)
        for interval in decoded.get('intervals', []):
            if interval.get('sample', 0) == 0:
                core, stage = interval['core'], interval['stage']
                if stage in interval_by_core[core]:
                    raise ValueError('Duplicate first core/stage interval')
                interval_by_core[core][stage] = interval
        milestones_by_core = defaultdict(dict)
        for milestone in decoded.get('milestones', []):
            core, point = milestone['core'], milestone['point']
            if point in milestones_by_core[core]:
                raise ValueError('Duplicate core milestone')
            milestones_by_core[core][point] = milestone['tick']
        present_kernel = {core for core, rows in rows_by_core.items() if 'KERNEL' in rows}
        kernel_cores.update(present_kernel)
        kernel_counts.append(len(present_kernel))
        per_stage = defaultdict(list)
        per_latency = defaultdict(list)
        per_activity = defaultdict(list)
        for core, rows in rows_by_core.items():
            group = _group(core, identity)
            for stage, row in rows.items():
                duration = _duration(row, frequency)
                if row.get('time_measured', True) is not False and stage not in COUNTERS:
                    key = (stage, group)
                    if duration is not None:
                        stage_values[key].append(duration)
                        per_stage[key].append(duration)
                    if row.get('calls') is not None:
                        stage_calls[key].append(row['calls'])
                elif stage in COUNTERS:
                    counter_rows[stage] += 1
            spans = dict(zip(LATENCIES[:3], (_duration(rows.get(name), frequency) for name in LATENCIES[:3])))
            raw_spans = {}
            for target, label in (('BEFORE_FIRST_SEND', 'entry_to_first_send'),
                                  ('BEFORE_FIRST_RECV_POLL', 'entry_to_first_probe'),
                                  ('KERNEL', 'entry_to_kernel_end')):
                raw_spans[label] = _entry_span(interval_by_core[core], target)
                spans[label] = _us(raw_spans[label], frequency)
            if identity['schema'] == 15:
                points = milestones_by_core[core]
                if 'ENTRY' in points and 'PREFIX_READY' in points:
                    ticks = points['PREFIX_READY'] - points['ENTRY']
                    if ticks < 0:
                        raise ValueError('Invalid same-core entry to prefix-ready span')
                    spans['entry_to_prefix_ready'] = _us(ticks, frequency)
                if 'FIRST_RECV_READY' in points:
                    spans['entry_to_first_ready'] = _us(points['FIRST_RECV_READY']-points['ENTRY'], frequency)
                    spans['init_to_first_ready'] = _us(points['FIRST_RECV_READY']-points['INIT_DONE'], frequency)
            for label, value in spans.items():
                if value is not None:
                    per_latency[(label, group)].append(value)
                    per_latency[(label, 'all')].append(value)
            # Clock comparison here is duration-only; start timestamps from
            # different ranks never enter selection or arithmetic.
            span_ticks = raw_spans['entry_to_kernel_end']
            span_kind = 'entry_to_kernel_end'
            if span_ticks is None:
                span_kind = 'KERNEL'
                span_ticks = rows.get('KERNEL', {}).get('total_ticks')
            if span_ticks is not None and (group not in slowest or span_ticks > slowest[group]['ticks']):
                phases = [[stage, _rounded(_duration(row, frequency)), row.get('calls'), row.get('total_ticks')]
                          for stage, row in rows.items() if stage not in COUNTERS and row.get('time_measured', True)]
                phases.sort(key=lambda row: (-(row[3] or 0), row[0]))
                slowest[group] = dict(rank=pair[0], iteration=pair[1], core=core,
                    span_kind=span_kind, ticks=span_ticks, us=_rounded(_us(span_ticks, frequency)),
                    top_stages=phases[:12], phase_totals=phases)
            if core in present_kernel:
                per_activity[group].append(rows)
                per_activity['all'].append(rows)
        for key, values in per_stage.items():
            stage_maxima[key].append(max(values))
        for key, values in per_latency.items():
            latency_maxima[key].append(max(values))
        if identity['schema'] == 15 and identity['operation'] == 'dispatch':
            for group in ('S', 'R', 'all'):
                observed_cores = [core for core in present_kernel if group == 'all' or _group(core, identity) == group]
                for point, label in (('FIRST_SEND', 'first_send_cores'), ('FIRST_RECV_PROBE', 'first_probe_cores'), ('FIRST_RECV_READY', 'first_ready_cores')):
                    activity_values[(group, label)].append(sum(point in milestones_by_core[core] for core in observed_cores))
        for group, core_rows in per_activity.items():
            for name, label in (('TX_PACKETS', 'tx'), ('RX_PACKETS', 'rx')):
                values = [_counter(rows, name) for rows in core_rows]
                observed = [v for v in values if v is not None]
                positive = [v for v in observed if v > 0]
                activity_values[(group, label+'_active_observed')].append(len(positive) if observed else None)
                activity_values[(group, label+'_counter_cores')].append(len(observed))
                activity_values[(group, label+'_counter_unknown_cores')].append(len(values)-len(observed))
                activity_values[(group, label+'_packets_observed')].append(sum(observed) if observed else None)
                for field, value in (('positive_min', min(positive) if positive else None),
                                     ('positive_median', statistics.median(positive) if positive else None),
                                     ('positive_max', max(positive) if positive else None)):
                    activity_values[(group, label+'_'+field)].append(value)
            probed = [_counter(rows, 'RX_PROBED') for rows in core_rows]
            ready = [_counter(rows, 'RX_READY') for rows in core_rows]
            probe_total = sum(v for v in probed if v is not None) if any(v is not None for v in probed) else None
            ready_total = sum(v for v in ready if v is not None) if any(v is not None for v in ready) else None
            incomplete = sum(p is not None and r is None for p, r in zip(probed, ready))
            missing_probe = sum(r is not None and p is None for p, r in zip(probed, ready))
            activity_values[(group, 'probed_observed')].append(probe_total)
            activity_values[(group, 'ready_observed')].append(ready_total)
            activity_values[(group, 'probe_cores_without_ready_counter')].append(incomplete)
            activity_values[(group, 'ready_cores_without_probe_counter')].append(missing_probe)
            complete = probe_total is not None and ready_total is not None and not incomplete and not missing_probe
            activity_values[(group, 'probed_per_ready')].append(probe_total/ready_total if complete and ready_total else None)
            activity_values[(group, 'ready_fraction')].append(ready_total/probe_total if complete and probe_total else None)
    world = identity['world_size']
    expected = {(r, i) for r in range(world) for i in range(expected_iters)} if type(world) is int and type(expected_iters) is int else None
    # Lists of observed rank/iteration IDs retain holes without an 80-row table.
    coverage = dict(samples=len(samples), ranks=sorted(ranks, key=lambda x: (x is None, x)),
        iterations=sorted(iterations, key=lambda x: (x is None, x)), rank_iterations=len(pairs),
        expected_rank_iterations=len(expected) if expected is not None else None,
        complete_rank_iterations=pairs == expected if expected is not None else None,
        kernel_cores=sorted(kernel_cores), kernel_core_count=_stats(kernel_counts),
        complete_kernel_cores=all(n == identity['active_aiv'] for n in kernel_counts) if identity['active_aiv'] is not None else None,
        correctness=dict(sorted(correct.items())), counter_rows=dict(sorted(counter_rows.items())))
    stages = [[stage, group, _stats(stage_values[(stage, group)]), _stats(stage_maxima[(stage, group)]),
               _stats(stage_calls[(stage, group)])] for stage, group in sorted(set(stage_values) | set(stage_calls))]
    return dict(format=FORMAT, identity=identity, coverage=coverage,
        stat_columns=STAT_COLUMNS, stage_columns=STAGE_COLUMNS, stages=stages,
        latency_columns=['span', 'initial_group', 'rank_iteration_max_us'],
        latencies=[[label, group, _stats(values)] for (label, group), values in sorted(latency_maxima.items())],
        activity_columns=['initial_group', 'quantity', 'rank_iteration_distribution'],
        activity=[[group, name, _stats(values)] for (group, name), values in sorted(activity_values.items())],
        slowest_columns=['stage', 'total_us', 'calls', 'total_ticks'], slowest=slowest,
        notes=list(NOTES))
