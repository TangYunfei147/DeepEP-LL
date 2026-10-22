"""Render restored receiver evidence without reconstructing unsampled activity.

Transfer exports contain only three replay envelopes, not the complete raw trace.
Each rank stays in its own file and each capture has an independent local origin.
The raw-run path delegates event interpretation to the existing timer parser.
"""
import importlib.util
import json
import math
from pathlib import Path

from receiver_export import CORE_COLUMNS, verify
from receiver_report import CASES


def _inside(root, path):
    path = path.resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'Input path escapes restored bundle: {path}')
    return path


def _integer(value, label):
    if type(value) is not int or value < 0:
        raise ValueError(f'{label} must be a nonnegative integer')
    return value


def _frequency(value):
    if (type(value) not in (int, float) or not math.isfinite(value) or value <= 0):
        raise ValueError('Trace rendering requires a finite positive frequency_hz')
    return value


def _recover(root):
    # Verify the full dataset before interpreting any records. Also prevent a
    # restored directory containing symlinks from reading outside that bundle.
    index = _inside(root, root / '000-index.json')
    manifest = json.loads(index.read_bytes())
    for entry in manifest.get('files', []):
        _inside(root, root / entry['file'])
    verify(root)
    groups = {}
    for entry in manifest['files']:
        part = json.loads((root / entry['file']).read_bytes())
        groups.setdefault(part['kind'], []).extend(part['records'])
    for kind, records in groups.items():
        recovered = []
        cursor = 0
        while cursor < len(records):
            item = records[cursor]
            if 'fragment_record_sha256' not in item:
                recovered.append(item)
                cursor += 1
                continue
            count = item['fragments']
            recovered.append(json.loads(''.join(
                piece['json_text'] for piece in records[cursor:cursor + count])))
            cursor += count
        groups[kind] = recovered
    return manifest, groups


def _transfer_plans(root):
    manifest, groups = _recover(root)
    meta = [item['data'] for item in groups['summary']
            if item.get('section') == 'metadata']
    if len(meta) != 1 or meta[0].get('core_columns') != CORE_COLUMNS:
        raise ValueError('Missing or unsupported transfer core columns')
    case_keys = meta[0].get('case_keys', {})
    cases = sorted({row['case'] for row in groups['cores']})
    case_numbers = {case: index for index, case in enumerate(cases)}
    owners = {}
    seen = set()
    diagnostics = []
    for row in groups['cores']:
        case = row['case']
        rank = _integer(row['rank'], 'rank')
        iteration = _integer(row['iteration'], 'iteration')
        identity = (case, rank, iteration)
        if identity in seen:
            raise ValueError('Duplicate transfer rank/capture')
        seen.add(identity)
        frequency = _frequency(row['frequency_hz'])
        order = row['order']
        if type(order) is not int or order not in range(3):
            raise ValueError('Invalid replay order')
        values = {}
        for value in row['cores']:
            if len(value) != len(CORE_COLUMNS):
                raise ValueError('Invalid transfer core record')
            core = _integer(value[0], 'core')
            if core in values:
                raise ValueError('Duplicate core record')
            for field, number in zip(CORE_COLUMNS[1:], value[1:]):
                _integer(number, field)
            values[core] = value
        intervals = {}
        for value in row.get('optional_sampled_intervals', []):
            if len(value) != 4 or value[0] not in values or value[0] in intervals:
                raise ValueError('Invalid or duplicate sampled core intervals')
            intervals[value[0]] = value[1:]
        events = []
        missing = []
        for core, value in sorted(values.items()):
            for arm_index, arm in enumerate(CASES):
                pair = intervals.get(core, [[None, None]] * 3)[arm_index]
                if not isinstance(pair, list) or len(pair) != 2:
                    raise ValueError('Invalid sampled interval pair')
                begin, end = pair
                if begin is None or end is None:
                    missing.append(dict(core=core, arm=arm,
                                        reason='missing_sampled_interval'))
                    continue
                _integer(begin, 'first_tick')
                _integer(end, 'last_tick')
                if end < begin or end - begin != value[arm_index + 1]:
                    raise ValueError('Replay interval differs from single-call ticks')
                events.append(dict(ph='X', name=CASES[arm][0], cat='receiver replay',
                    pid=iteration + 1, tid=core,
                    args=dict(rank=rank, iteration=iteration, case=case, arm=arm,
                              execution_position=(arm_index - order) % 3 + 1,
                              raw_begin_cycle=begin, raw_end_cycle=end,
                              raw_duration_ticks=value[arm_index + 1],
                              frequency_hz=frequency,
                              interval_kind='sampled actual replay envelope')))
        origin = min((event['args']['raw_begin_cycle'] for event in events), default=None)
        for event in events:
            args = event['args']
            event.update(ts=(args['raw_begin_cycle'] - origin) * 1e6 / frequency,
                         dur=args['raw_duration_ticks'] * 1e6 / frequency)
        diagnostic = dict(case=case, rank=rank, iteration=iteration,
                          origin_tick=origin, frequency_hz=frequency,
                          sampled_intervals=len(events), missing_intervals=missing,
                          status='missing' if not events else 'partial' if missing else 'rendered')
        diagnostics.append(diagnostic)
        owners.setdefault((case, rank), []).append((iteration, events, diagnostic))

    plans = []
    skipped = []
    for (case, rank), captures in sorted(owners.items()):
        if not any(events for _, events, _ in captures):
            skipped.append(dict(case=case, rank=rank,
                                reason='No actual sampled intervals in any capture'))
            continue
        events = []
        capture_info = []
        for iteration, sampled, diagnostic in sorted(captures):
            if not sampled:
                continue
            events.append(dict(ph='M', name='process_name', pid=iteration + 1,
                               args=dict(name=f'rank {rank}, capture {iteration} (independent origin)')))
            for core in sorted({event['tid'] for event in sampled}):
                events.append(dict(ph='M', name='thread_name', pid=iteration + 1,
                                   tid=core, args=dict(name=f'AIV {core:02d}')))
            events.extend(sampled)
            capture_info.append({key: value for key, value in diagnostic.items()
                                 if key != 'missing_intervals'})
        trace = dict(traceEvents=events, displayTimeUnit='us', metadata=dict(
            format='receiver transfer sampled replay envelopes',
            dataset=manifest['dataset'], rank=rank, case=case,
            original_case=case_keys.get(case), captures=capture_info,
            scope='Only the three recorded replay envelopes; not a complete dispatch trace.',
            clock='Each capture has an independent rank-local origin. No cross-rank or cross-capture alignment.',
            missing_policy='Missing intervals are omitted, never synthesized from accumulated ticks.'))
        plans.append((f'case{case_numbers[case]:03d}-rank{rank:03d}-trace.json', trace))
    return plans, dict(dataset=manifest['dataset'], captures=diagnostics, skipped_ranks=skipped)


def _timer():
    source = Path(__file__).resolve().parents[2] / 'deep_ep/utils/ll_stage_timer.py'
    spec = importlib.util.spec_from_file_location('receiver_trace_timer', source)
    timer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(timer)
    return timer


def _run_plans(root):
    timer = _timer()
    plans, captures, identities = [], [], set()
    for path in sorted(root.rglob('*.json')):
        path = _inside(root, path)
        saved = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(saved, dict) or 'raw' not in saved or saved.get('operation') != 'dispatch':
            continue
        meta = saved.get('metadata', {})
        rank = _integer(meta.get('rank'), 'rank')
        iteration = _integer(meta.get('iteration'), 'iteration')
        frequency = _frequency(saved.get('frequency_hz'))
        case = json.dumps([meta.get('run_id'), meta.get('config'), meta.get('case_id'),
                           meta.get('route'), meta.get('seed')], sort_keys=True)
        identity = (case, rank, iteration)
        if identity in identities:
            raise ValueError('Duplicate raw rank/capture')
        identities.add(identity)
        decoded = timer.parse_records(saved['raw'], 'dispatch', frequency)
        trace = timer.chrome_trace([saved])
        trace.update(displayTimeUnit='us')
        trace['metadata'].update(source=path.relative_to(root).as_posix(),
            frequency_hz=frequency,
            origin_tick=min(item['first_tick'] for item in decoded['intervals']),
            scope='All actual sampled intervals from this raw capture; cumulative metrics do not generate events.',
            clock='Independent origin for this rank/capture. No cross-rank or cross-capture alignment.')
        events = trace['traceEvents']
        events.append(dict(ph='M', name='process_name', pid=1,
                           args=dict(name=f'rank {rank}, capture {iteration} (independent origin)')))
        name = f'capture{len(plans):04d}-rank{rank:03d}-iteration{iteration:03d}-trace.json'
        plans.append((name, trace))
        captures.append(dict(source=path.relative_to(root).as_posix(), file=name,
                             rank=rank, iteration=iteration, schema=decoded['schema'],
                             frequency_hz=frequency, sampled_intervals=len(decoded['intervals'])))
    if not plans:
        raise ValueError('No raw dispatch captures in restored receiver run')
    return plans, dict(captures=captures, skipped_ranks=[])


def render_bundle(root, source_kind):
    """Return new trace/report paths below ``root/rendered``; never overwrite.

    The report is always emitted, including when every transfer interval is
    absent. It makes skipped ranks and partial captures explicit to the caller.
    No input is modified and rendered files are not part of the transfer input.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError('Restored bundle directory does not exist')
    output = root / 'rendered'
    if output.exists():
        raise ValueError('Rendered output exists; use a new restore directory')
    if source_kind == 'receiver-transfer':
        plans, details = _transfer_plans(root)
    elif source_kind == 'receiver-run':
        plans, details = _run_plans(root)
    else:
        raise ValueError(f'Unknown receiver source kind: {source_kind}')
    report = dict(source_kind=source_kind, rendered_trace_files=[name for name, _ in plans],
                  interpretation='Only actual sampled intervals are rendered; missing intervals and unmeasured gaps remain unknown.',
                  **details)
    plans.append(('render-report.json', report))
    output.mkdir()
    paths = []
    for name, value in plans:
        path = output / name
        with path.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
            stream.write('\n')
        paths.append(path)
    return paths
