"""Build a Chrome trace from losslessly transferred, decoded timer samples.

Only saved interval rows become events. Accumulated stage metrics cannot provide
a packet timeline and are deliberately never used to synthesize one. Every
rank/operation/iteration has its own clock origin and its own process track.
"""
import math


_PROFILES = ('full', 'overview', 'prepare', 'prefix', 'send', 'recv', 'sparse')
_CONTEXT_FIELDS = ('world_size', 'config', 'run_id', 'seed', 'case_id', 'sampling',
                   'comparison', 'active_capture_profile')


def _frequency(value):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or value <= 0):
        raise ValueError('Chrome trace requires explicit finite positive frequency_hz')
    return value


def _integer(value, description):
    if type(value) is not int or value < 0:
        raise ValueError(f'Invalid {description}')
    return value


def _profile(decoded):
    # Schema 13 has no profile field in its device header: its probes are full.
    name = decoded.get('timer_profile', 'full')
    profile_id = decoded.get('timer_profile_id', 0)
    if (type(profile_id) is not int or not 0 <= profile_id < len(_PROFILES) or
            name != _PROFILES[profile_id] or
            (decoded['schema'] == 13 and profile_id != 0) or
            (decoded['schema'] == 14 and profile_id == 6) or
            (decoded['schema'] == 15 and profile_id != 6)):
        raise ValueError('Invalid timer profile identity')
    return name, profile_id


def _interval(row, decoded, profile):
    core = _integer(row.get('core'), 'interval core')
    if core >= decoded['active_aiv']:
        raise ValueError('Interval core exceeds active_aiv')
    if not isinstance(row.get('stage'), str) or not row['stage']:
        raise ValueError('Invalid interval stage')
    _integer(row.get('sample'), 'interval sample')
    begin = _integer(row.get('first_tick'), 'interval first_tick')
    end = _integer(row.get('last_tick'), 'interval last_tick')
    total = _integer(row.get('total_ticks'), 'interval total_ticks')
    if end < begin or total != end - begin:
        raise ValueError('Invalid actual interval duration')
    if (row.get('timer_profile', profile[0]) != profile[0] or
            row.get('timer_profile_id', profile[1]) != profile[1]):
        raise ValueError('Mixed timer profiles in intervals')


def trace_from_decoded(samples, trace_iteration=0, frequency_hz=None):
    """Return Chrome trace JSON data, without reading raw tensors or writing files.

    ``samples`` contains ``(metadata, decoded_report)`` pairs from the transfer
    bundle. Only ``trace_iteration`` is displayed; other iterations remain in the
    restored analysis data. Explicit ``frequency_hz`` overrides saved frequencies,
    as in the normal raw-tensor trace generator. No cross-rank alignment is made.

    A separate numeric AIV/stage track keeps KERNEL and parent milestones apart
    from their child probes. This changes presentation, never interval timing.
    """
    _integer(trace_iteration, 'trace iteration')
    if frequency_hz is not None:
        _frequency(frequency_hz)
    samples = list(samples)
    selected = [(meta, decoded) for meta, decoded in samples
                if meta.get('iteration') == trace_iteration]
    if not selected:
        raise ValueError(f'No diagnostic iteration {trace_iteration} in transferred samples')

    identity = None
    seen = set()
    checked = []
    stages = set()
    for meta, decoded in selected:
        if decoded.get('schema') not in (13, 14, 15):
            raise ValueError('Restored trace requires schema 13 or 14 intervals, or schema15 sparse milestones')
        if decoded.get('operation') not in ('dispatch', 'combine'):
            raise ValueError('Expected dispatch or combine operation')
        active = _integer(decoded.get('active_aiv'), 'active_aiv')
        if not active:
            raise ValueError('Missing active AIV cores')
        rank = _integer(meta.get('rank'), 'rank')
        iteration = _integer(meta.get('iteration'), 'iteration')
        if meta.get('world_size') is not None:
            world = _integer(meta['world_size'], 'world_size')
            if rank >= world:
                raise ValueError('Rank exceeds world_size')
        profile = _profile(decoded)
        frequency = _frequency(frequency_hz if frequency_hz is not None
                               else decoded.get('frequency_hz'))
        common = (tuple(meta.get(key) for key in _CONTEXT_FIELDS),
                  decoded['schema'], decoded.get('boundary_mode'), profile,
                  active, decoded.get('send_cores'), decoded.get('recv_cores'),
                  decoded.get('dispatch_recv_cores'), frequency)
        if identity is not None and common != identity:
            raise ValueError('Mixed configurations, schemas, timer profiles, modes or frequencies')
        identity = common
        key = (decoded['operation'], rank, iteration)
        if key in seen:
            raise ValueError('Duplicate rank/operation/iteration trace')
        seen.add(key)
        rows = decoded.get('intervals')
        if not isinstance(rows, list) or not rows:
            raise ValueError('No actual intervals in selected transferred sample')
        row_keys = set()
        for row in rows:
            _interval(row, decoded, profile)
            row_key = (row['core'], row['stage'], row['sample'])
            if row_key in row_keys:
                raise ValueError('Duplicate sampled interval')
            row_keys.add(row_key)
            stages.add(row['stage'])
        checked.append((meta, decoded, profile, frequency))

    # Named stage tracks also distinguish overlapping parent/child intervals.
    first = {'KERNEL': -3, 'TIMER_SETUP': -2, 'INIT': -1}
    stage_order = sorted(stages, key=lambda stage: (first.get(stage, 0), stage))
    stage_ids = {stage: index for index, stage in enumerate(stage_order)}
    stride = len(stage_ids)
    checked.sort(key=lambda item: (item[0]['rank'], item[1]['operation']))
    events, info = [], []
    for pid, (meta, decoded, profile, frequency) in enumerate(checked, 1):
        rank, iteration, operation = meta['rank'], meta['iteration'], decoded['operation']
        rows = decoded['intervals']
        # One origin across all AIVs, including work before KERNEL (INIT/setup).
        origin = min(row['first_tick'] for row in rows)
        events.extend([
            dict(ph='M', name='process_name', pid=pid, tid=0,
                 args=dict(name=f'rank {rank} / {operation} / iteration {iteration} / {profile[0]}')),
            dict(ph='M', name='process_sort_index', pid=pid, tid=0,
                 args=dict(sort_index=pid)),
        ])
        core_roles = {row['core']: row.get('role', row.get('initial_role', 'unknown'))
                      for row in decoded.get('cores', [])}
        lanes = set()
        for row in sorted(rows, key=lambda row: (row['core'], stage_ids[row['stage']],
                                                row['first_tick'], -row['last_tick'])):
            core = row['core']
            tid = core * stride + stage_ids[row['stage']]
            role = row.get('role', core_roles.get(core, 'unknown'))
            if tid not in lanes:
                events.extend([
                    dict(ph='M', name='thread_name', pid=pid, tid=tid,
                         args=dict(name=f"AIV {core:02d} {role} / {row['stage']}")),
                    dict(ph='M', name='thread_sort_index', pid=pid, tid=tid,
                         args=dict(sort_index=tid)),
                ])
                lanes.add(tid)
            events.append(dict(
                ph='X', name=row['stage'], cat=operation, pid=pid, tid=tid,
                ts=(row['first_tick'] - origin) * 1e6 / frequency,
                dur=row['total_ticks'] * 1e6 / frequency,
                args=dict(rank=rank, iteration=iteration, core=core, role=role,
                          sample=row['sample'], timer_profile=profile[0],
                          timer_profile_id=profile[1],
                          raw_begin_cycle=row['first_tick'], raw_end_cycle=row['last_tick'],
                          raw_duration_cycles=row['total_ticks'],
                          interval_kind=row.get('interval_kind', 'sampled actual interval'))))
        for point in decoded.get('milestones', []):
            core = _integer(point.get('core'), 'milestone core')
            tick = _integer(point.get('tick'), 'milestone tick')
            if core >= decoded['active_aiv']:
                raise ValueError('Milestone core exceeds active_aiv')
            events.append(dict(ph='i', s='t', name=point['point'], cat='milestone', pid=pid, tid=core*stride,
                               ts=(tick-origin)*1e6/frequency,
                               args=dict(rank=rank, iteration=iteration, core=core, raw_cycle=tick)))
        info.append(dict(operation=operation, rank=rank, iteration=iteration, pid=pid,
                         origin_tick=origin, frequency_hz=frequency,
                         cores=decoded.get('cores', []), timer_profile=profile[0],
                         timer_profile_id=profile[1]))
    decoded, profile = checked[0][1:3]
    return dict(traceEvents=events, displayTimeUnit='us', metadata=dict(
        format='sparse milestones restored from lossless text' if decoded['schema']==15 else 'dc_seg sampled intervals restored from lossless text',
        schema=decoded['schema'], boundary_mode=decoded.get('boundary_mode'),
        timer_profile=profile[0], timer_profile_id=profile[1],
        sampled_calls_per_stage=None if decoded['schema']==15 else 2, all_calls_in_metrics=decoded['schema']!=15,
        trace_iteration=trace_iteration, source_samples=len(samples),
        selected_samples=len(selected), cores=info,
        lane_layout='AIV/stage; parent milestones and KERNEL use separate tracks',
        clock='Rank-local origins; no cross-rank alignment',
        interpretation='Only recorded actual intervals, including waits. No timeline is reconstructed from accumulated totals.'))
