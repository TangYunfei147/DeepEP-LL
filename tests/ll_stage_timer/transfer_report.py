"""Bounded, lossless timer exports for terminals with a 65535-character limit.

The priority series is deliberately smaller than the full series. Tables carry
their own field names and never fill absent metrics with zero. Every text part is
an independent ASCII JSON document; concatenate its ``records`` arrays, not the
files themselves. ``restore_samples`` verifies and reconstructs the full series.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics

FORMAT = 'll-timer-transfer-v1'
DEFAULT_LIMIT = 60000
MAX_LIMIT = 65535
PREFIX_STAGES = ('SEND_COUNTS', 'COUNT_COMPUTE', 'COUNT_SEND_ISSUE',
                 'WAIT_RECV_COUNTS', 'GATHER_RECV_COUNTS', 'GET_CUMSUM',
                 'BUILD_RECV_CUMSUM', 'PREFIX_COMPLETE', 'POLL_CORE_SUMS',
                 'REDUCE_CORE_PREFIX', 'BROADCAST_PREFIX', 'PUBLISH_PREFIX_FLAGS',
                 'PREFIX_PUBLISH_WAIT', 'WAIT_CUMSUM', 'PREFIX_LOAD',
                 'RECV_PREPARE', 'BUILD_EXPERT_TASKS', 'WRITE_COUNTS_OUTPUT',
                 'BEFORE_FIRST_SEND', 'BEFORE_FIRST_RECV_POLL', 'SEND_SETUP',
                 'SEND_COMPLETE_WAIT', 'SEND_RETIRE_WAIT', 'RECV_COMPLETE_WAIT',
                 'FINAL_DRAIN', 'ALL_CORE_WAIT', 'TIMER_FLUSH')
COUNTERS = ('TX_PACKETS', 'RX_PACKETS', 'RELAY_PACKETS', 'RX_PROBED',
            'RX_READY', 'RX_EMPTY_POLLS')
NOTES = (
    'Raw ticks and calls are retained. Missing stages/counters are absent, never imputed as zero.',
    'Stage totals may nest or overlap; do not sum them. Submission is not DMA completion.',
    'Only recorded real intervals are exported; typically first two calls, not a full packet timeline.',
    'No cross-rank clock alignment or subtraction. Kernel spans exclude uninstrumented final synchronization/export.',
    'Activity comes from observed TX/RX/relay counters; header roles alone do not establish activity.',
    'ON diagnostics are not an OFF performance baseline; low ready fraction alone does not prove congestion.',
)


def _json(value):
    return json.dumps(value, ensure_ascii=True, separators=(',', ':'), sort_keys=True,
                      allow_nan=False)


def _hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _stats(values):
    values = list(values)
    if not values:
        return dict(observed=0)
    return dict(observed=len(values), minimum=min(values), p50=statistics.median(values),
                maximum=max(values))


def _activity(stages):
    tx = stages.get('TX_PACKETS', {}).get('calls')
    rx = stages.get('RX_PACKETS', {}).get('calls')
    if tx and rx:
        return 'tx_rx_observed'
    if tx:
        return 'tx_observed'
    if rx:
        return 'rx_observed'
    return 'no_tx_rx_counter_record'


def _partition(core, decoded):
    split = decoded.get('send_cores')
    return ('S' if core < split else 'R') if split is not None else 'unknown'


def _tables(kind, sample_id, rows, budget):
    """Keep exact key presence/order of rows, splitting tables before size limits."""
    fields = None
    table = None
    for row in rows:
        keys = sorted(row)
        if keys != fields or (table and len(table['rows']) >= 64):
            if table:
                yield table
            fields = keys
            table = dict(kind=kind, sample=sample_id, fields=fields, rows=[])
        values = [row[key] for key in fields]
        candidate = dict(table, rows=table['rows'] + [values])
        if len(_json(candidate)) > budget:
            if table['rows']:
                yield table
                table = dict(kind=kind, sample=sample_id, fields=fields, rows=[])
                candidate = dict(table, rows=[values])
            if len(_json(candidate)) > budget:
                raise ValueError(f'{kind}: one row exceeds transfer limit; no data was truncated')
        table = candidate
    if table:
        yield table


def _full_records(report, samples, budget):
    derived = {'communication', 'overlap_receive', 'input_wait_bottleneck', 'relay_timing',
               'per_rank_core_stage', 'per_rank_stage_max', 'per_expert',
               'prefix_diagnosis', 'prefix_diagnostics'}
    yield dict(kind='report_context', data={k: v for k, v in report.items() if k not in derived})
    for sample_id, (meta, decoded) in enumerate(samples):
        yield dict(kind='sample', sample=sample_id, metadata=meta,
                   decoded={k: v for k, v in decoded.items() if k not in ('records', 'intervals')},
                   expected_stage_rows=len(decoded.get('records', [])),
                   expected_interval_rows=len(decoded.get('intervals', [])),
                   records_present='records' in decoded, intervals_present='intervals' in decoded)
        yield from _tables('stages', sample_id, decoded.get('records', []), budget)
        yield from _tables('intervals', sample_id, decoded.get('intervals', []), budget)


def _priority(samples, report):
    per_sample = []
    aggregates = defaultdict(list)
    core_details = []
    completeness = []
    contexts = []
    context_ids = {}
    for sample_id, (meta, decoded) in enumerate(samples):
        context = dict(metadata={k: v for k, v in meta.items() if k not in ('rank', 'iteration')},
                       decoded={k: v for k, v in decoded.items() if k not in ('records', 'intervals', 'cores')})
        context_key = _json(context)
        if context_key not in context_ids:
            context_ids[context_key] = len(contexts)
            contexts.append(dict(kind='sample_context', context=len(contexts), **context))
        cores = defaultdict(dict)
        for row in decoded.get('records', []):
            cores[row['core']][row['stage']] = row
        identity = dict(sample=sample_id, operation=decoded.get('operation'),
                        rank=meta.get('rank'), iteration=meta.get('iteration'),
                        schema=decoded.get('schema'), frequency_hz=decoded.get('frequency_hz'),
                        context=context_ids[context_key])
        spans = {c: st['KERNEL']['total_ticks'] for c, st in cores.items() if 'KERNEL' in st}
        longest = max(spans, key=spans.get) if spans else None
        activity = Counter(_activity(st) for st in cores.values())
        packet_counts = {}
        for name in COUNTERS:
            values = [st[name]['calls'] for st in cores.values() if name in st]
            packet_counts[name] = dict(observed_cores=len(values), absent_cores=len(cores)-len(values))
            if values:
                packet_counts[name]['observed_total'] = sum(values)
        item = dict(**identity, kernel_span_ticks=_stats(spans.values()),
                    longest_core=longest, observed_activity=dict(activity), counters=packet_counts,
                    expected_cores=decoded.get('active_aiv'), observed_cores=len(cores),
                    prefix_stages={}, preparation={},
                    interval_coverage=dict(recorded=len(decoded.get('intervals', [])),
                                           omitted_per_core=_stats(c['omitted_intervals'] for c in decoded.get('cores', [])
                                                                   if 'omitted_intervals' in c)))
        names = set(PREFIX_STAGES)
        names.update(name for st in cores.values() for name in st
                     if name.startswith(('PREFIX_', 'COUNT_SNAPSHOT_', 'COUNT_READY_', 'COUNT_VALUE_')))
        for name in sorted(names):
            rows = [st[name] for st in cores.values() if name in st]
            if rows:
                item['prefix_stages'][name] = _stats(r['total_ticks'] for r in rows)
        for core, stages in cores.items():
            category = _activity(stages)
            part = _partition(core, decoded)
            for name, row in stages.items():
                key = (decoded.get('operation'), part, category, name)
                aggregates[key].append(row)
            if core == longest:
                core_details.append((spans[core], sample_id, core))
        # INIT is outside KERNEL. Only subtract timestamps from this same core/sample.
        intervals = defaultdict(dict)
        for row in decoded.get('intervals', []):
            if row.get('sample', 0) == 0:
                intervals[row['core']][row['stage']] = row
        for label,start,finish in (
                ('entry_to_first_send_ticks','INIT','BEFORE_FIRST_SEND'),
                ('entry_to_first_probe_ticks','INIT','BEFORE_FIRST_RECV_POLL'),
                ('entry_to_kernel_end_ticks','INIT','KERNEL'),
                ('timer_entry_to_kernel_end_ticks','TIMER_SETUP','KERNEL')):
            values=[]
            for core,trace in intervals.items():
                if start in trace and finish in trace:
                    value=trace[finish]['last_tick']-trace[start]['first_tick']
                    if value<0: raise ValueError('Non-monotonic same-core preparation interval')
                    values.append(value)
            if values: item['preparation'][label]=_stats(values)
        item['active_capture_profile']=meta.get('active_capture_profile')
        per_sample.append(item)
        expected = decoded.get('active_aiv')
        completeness.append(dict(**identity, observed_kernel_cores=len(spans), expected_cores=expected,
                                 all_kernel_cores_present=(len(spans) == expected) if expected is not None else None))
    records = [dict(kind='priority_context', notes=list(NOTES),
                    detail_availability='Schema13 lacks schema14 prefix subphase splits; absent values remain missing.',
                    schemas=sorted({d.get('schema') for _, d in samples if d.get('schema') is not None}),
                    configurations=report.get('configurations'), sampling=report.get('sampling'))]
    records.extend(contexts)
    records.extend(dict(kind='sample_summary', data=row) for row in per_sample)
    aggregate_rows = []
    for (operation, partition, activity, stage), rows in sorted(aggregates.items()):
        row = dict(operation=operation, partition=partition, observed_activity=activity, stage=stage,
                   calls=_stats(r['calls'] for r in rows),
                   total_ticks=_stats(r['total_ticks'] for r in rows),
                   time_measured=all(r.get('time_measured', True) for r in rows))
        aggregate_rows.append(row)
    # One longest-core sample per rank/iteration is ranked; send only the top four
    # in the initial series. Every other core is preserved in the full series.
    selected = sorted(core_details, reverse=True)[:4]
    return records, aggregate_rows, selected, per_sample, completeness


def _pack(series, records, bundle_id, limit):
    def encode(part, parts, items):
        return _json(dict(format=FORMAT, bundle_id=bundle_id, series=series,
                          part=part, parts=parts, table_encoding='zip(fields,row); preserve absent keys',
                          records=items)) + '\n'
    groups = []
    group = []
    for record in records:
        # Reserve maximum ten-digit part counts while packing; final headers shrink.
        if len(encode(9999999999, 9999999999, group + [record])) > limit:
            if not group:
                raise ValueError(f"{series}/{record.get('kind')}: one record exceeds {limit} characters; no data was truncated")
            groups.append(group)
            group = []
            if len(encode(9999999999, 9999999999, [record])) > limit:
                raise ValueError(f"{series}/{record.get('kind')}: one record exceeds {limit} characters; no data was truncated")
        group.append(record)
    if group:
        groups.append(group)
    return [encode(i + 1, len(groups), group) for i, group in enumerate(groups)]


def _overview(bundle_id, samples, per_sample, aggregate, limit, priority_names, full_names):
    lines = ['# LL timer transfer overview', '', f'Bundle: `{bundle_id}`',
             f'Send this file first, then {len(priority_names)} priority part(s).',
             'The full series is lossless backup; send selected parts only when needed.', '',
             f'Samples: {len(samples)}; full parts: {len(full_names)}.',
             'Profile: '+str(sorted({d.get('timer_profile','full') for _,d in samples})),
             'Units below are raw ticks. Frequency and exact metadata are in priority/full parts.', '',
             '## Longest core in each rank/iteration', '',
             '| operation | rank | iter | core | KERNEL ticks | TX cores | RX cores | PREFIX_COMPLETE max | first Probe p50 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    omitted = 0
    for row in per_sample:
        activity = row['observed_activity']
        tx = activity.get('tx_observed', 0) + activity.get('tx_rx_observed', 0)
        rx = activity.get('rx_observed', 0) + activity.get('tx_rx_observed', 0)
        prefix = row['prefix_stages'].get('PREFIX_COMPLETE', {}).get('maximum', 'missing')
        probe = row['prefix_stages'].get('BEFORE_FIRST_RECV_POLL', {}).get('p50', 'missing')
        line = f"| {row['operation']} | {row['rank']} | {row['iteration']} | {row['longest_core']} | {row['kernel_span_ticks'].get('maximum', 'missing')} | {tx} | {rx} | {prefix} | {probe} |"
        if len(('\n'.join(lines + [line])).encode('utf-8')) < limit - 3500:
            lines.append(line)
        else:
            omitted += 1
    if omitted:
        lines.extend(['', f'{omitted} summary rows continue in priority files; raw data are not truncated.'])
    if len('\n'.join(lines).encode('utf-8')) < limit - 3500:
        lines.extend(['', '## Observed per-core/per-sample stage totals', '',
                      'Each row is a separate distribution; parent and child stages must not be added.', '',
                      '| operation | partition | observed activity | stage | n | p50 ticks | max ticks |',
                      '|---|---|---|---|---:|---:|---:|'])
    skipped = 0
    for row in aggregate:
        name = row['stage']
        if name not in (*PREFIX_STAGES, 'INIT', 'INIT_COUNT_BUFFERS', 'KERNEL', 'TOKEN_METADATA',
                        'TOKEN_PACK', 'RX_METADATA','TIMER_SETUP','SEND_SETUP','SEND_COUNTS','FINAL_DRAIN') and not name.startswith(('PREFIX_', 'COUNT_', 'INIT_', 'SENDER_')):
            continue
        stats = row['total_ticks']
        line = f"| {row['operation']} | {row['partition']} | {row['observed_activity']} | {name} | {stats['observed']} | {stats['p50']} | {stats['maximum']} |"
        if len(('\n'.join(lines + [line])).encode('utf-8')) < limit - 3500:
            lines.append(line)
        else:
            skipped += 1
    if skipped:
        lines.extend(['', f'{skipped} stage summary rows continue in priority files.'])
    lines.extend(['', '## How to read the data', '', *['- '+note for note in NOTES], '',
                  '- Prefix stage statistics use only observed records; compare PREFIX_COMPLETE, its children, first Probe and KERNEL together.',
                  '- Per-sample activity counts are observed positive packet counters. Missing counters do not prove an idle core.',
                  '- Priority aggregate medians are medians of observed per-core/per-sample values, not differences of medians.',
                  '- Full sample headers retain configuration, revision/topology fields when recorded, schema, frequency and interval omission counts.',
                  '- If revision/topology metadata was never recorded, the exporter cannot recover it.',
                  '- Manifest hashes detect missing, duplicated, reordered or modified full parts.', ''])
    result = '\n'.join(lines)
    if len(result) > limit or len(result.encode('utf-8')) > limit:
        raise ValueError('Overview exceeds transfer limit')
    return result


def export_bundle(output_dir, report, samples, limit=DEFAULT_LIMIT):
    """Write bounded files under output_dir/transfer; return the manifest dict.

    An oversized indivisible metadata value is an explicit error before writing
    any file. The optional limit must be between 4096 and 65535 inclusive.
    """
    if type(limit) is not int or not 4096 <= limit <= MAX_LIMIT:
        raise ValueError('transfer limit must be an integer in [4096, 65535]')
    samples = list(samples)
    profiles = defaultdict(set)
    for _, decoded in samples:
        profile = decoded.get('timer_profile', 'full' if decoded.get('schema', 0) <= 13 else 'unknown')
        profiles[decoded.get('operation')].add(profile)
    if any(len(values) > 1 for values in profiles.values()):
        raise ValueError('Mixed timer profiles cannot be aggregated; export each capture profile separately')
    budget = limit - 1024
    full = list(_full_records(report, samples, budget))
    bundle_id = _hash(_json(full))[:20]
    priority, aggregate, selected, summaries, completeness = _priority(samples, report)
    priority.extend(_tables('stage_aggregate', None, aggregate, budget))
    for ticks, sample_id, core in selected:
        meta, decoded = samples[sample_id]
        priority.append(dict(kind='selected_core', sample=sample_id, core=core, kernel_ticks=ticks,
                             rank=meta.get('rank'), iteration=meta.get('iteration'),
                             reason='top four per-sample longest kernel spans; full series contains all cores'))
        priority.extend(_tables('selected_stages', sample_id,
                                [r for r in decoded.get('records', []) if r['core'] == core], budget))
        priority.extend(_tables('selected_intervals', sample_id,
                                [r for r in decoded.get('intervals', []) if r['core'] == core], budget))
    full_parts = _pack('full', full, bundle_id, limit)
    priority_parts = _pack('priority', priority, bundle_id, limit)
    files = {}
    manifest_parts = []
    for series, parts in (('priority', priority_parts), ('full', full_parts)):
        for index, content in enumerate(parts, 1):
            name = f'{series}_{index:03d}_of_{len(parts):03d}.txt'
            files[name] = content
            parsed = json.loads(content)
            manifest_parts.append(dict(file=name, series=series, part=index, parts=len(parts),
                                       chars=len(content), utf8_bytes=len(content.encode('utf-8')),
                                       sha256=_hash(content), records=len(parsed['records']),
                                       sample_ids=sorted({r['sample'] for r in parsed['records']
                                                          if r.get('sample') is not None})))
    priority_names = [p['file'] for p in manifest_parts if p['series'] == 'priority']
    full_names = [p['file'] for p in manifest_parts if p['series'] == 'full']
    files['00_overview.md'] = _overview(bundle_id, samples, summaries, aggregate, limit, priority_names, full_names)
    expected_world = report.get('world_size')
    actual = defaultdict(list)
    for meta, decoded in samples:
        actual[decoded.get('operation')].append((meta.get('rank'), meta.get('iteration')))
    expected_iters = (report.get('sampling') or {}).get('diagnostic_iters')
    if expected_iters is None:
        values = [i for pairs in actual.values() for _, i in pairs if isinstance(i, int)]
        expected_iters = max(values) + 1 if values else None
    expected = ({(r, i) for r in range(expected_world) for i in range(expected_iters)}
                if isinstance(expected_world, int) and isinstance(expected_iters, int) else None)
    coverage = {op: dict(observed_samples=len(pairs), unique_samples=len(set(pairs)),
                        expected_samples=len(expected) if expected is not None else None,
                        complete=(set(pairs) == expected and len(pairs) == len(set(pairs)))
                        if expected is not None else None) for op, pairs in actual.items()}
    manifest = dict(format=FORMAT, bundle_id=bundle_id, limit=limit,
                    send_first=['00_overview.md', *priority_names],
                    full_payload_sha256=_hash(_json(full)), full_record_count=len(full),
                    samples=len(samples), stage_rows=sum(len(d.get('records', [])) for _, d in samples),
                    interval_rows=sum(len(d.get('intervals', [])) for _, d in samples),
                    export_complete=True, source_sample_coverage=coverage,
                    all_observed_kernel_cores_complete=all(x['all_kernel_cores_present'] is True for x in completeness) if completeness else None,
                    completeness_note='Export is lossless for supplied decoded samples. Expected rank/iteration coverage is separate; intervals were sampled on device.',
                    series_counts=dict(priority=len(priority_parts), full=len(full_parts)),
                    part_index=[], notes=list(NOTES))
    # The index itself may exceed one terminal download for large captures.
    indexes = _pack('index', [dict(kind='part_index', data=p) for p in manifest_parts], bundle_id, limit)
    for index, content in enumerate(indexes, 1):
        name = f'part_index_{index:03d}_of_{len(indexes):03d}.txt'
        files[name] = content
        manifest['part_index'].append(dict(file=name, sha256=_hash(content), chars=len(content),
                                           utf8_bytes=len(content.encode('utf-8'))))
    files['manifest.json'] = _json(manifest) + '\n'
    # Validate every byte before writing; no silent partial/truncated bundle.
    for name, content in files.items():
        if len(content) > limit or len(content.encode('utf-8')) > limit:
            raise ValueError(f'{name} exceeds transfer limit; no data was truncated')
    output = Path(output_dir) / 'transfer'
    output.mkdir(parents=True, exist_ok=True)
    # Do not overwrite a different bundle or leave old parts pretending to be current.
    existing = output / 'manifest.json'
    if existing.exists() and json.loads(existing.read_text(encoding='utf-8')).get('bundle_id') != bundle_id:
        raise ValueError('transfer contains a different bundle; choose a new analysis output directory')
    for name, content in files.items():
        (output / name).write_bytes(content.encode('utf-8'))
    return manifest


def restore_samples(directory):
    """Verify all full parts and return exact (report context, decoded samples)."""
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    records = []
    all_parts = []
    for ordinal, entry in enumerate(manifest['part_index'], 1):
        raw = (directory / entry['file']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry['sha256']:
            raise ValueError('Part index hash mismatch')
        index = json.loads(raw)
        if (index['bundle_id'] != manifest['bundle_id'] or index['part'] != ordinal
                or index['parts'] != len(manifest['part_index']) or index['series'] != 'index'):
            raise ValueError('Part index identity mismatch')
        all_parts.extend(r['data'] for r in index['records'])
    parts = [p for p in all_parts if p['series'] == 'full']
    if len(parts) != manifest['series_counts']['full']:
        raise ValueError('Full part count mismatch')
    if [p['part'] for p in parts] != list(range(1, len(parts) + 1)):
        raise ValueError('Missing, duplicate or reordered full parts')
    for expected in parts:
        raw = (directory / expected['file']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected['sha256']:
            raise ValueError('Full part hash mismatch')
        part = json.loads(raw)
        if (part['bundle_id'] != manifest['bundle_id'] or part['part'] != expected['part']
                or part['parts'] != len(parts) or part['series'] != 'full'):
            raise ValueError('Full part identity mismatch')
        records.extend(part['records'])
    if len(records) != manifest['full_record_count'] or _hash(_json(records)) != manifest['full_payload_sha256']:
        raise ValueError('Full payload incomplete or modified')
    headers = {}
    rows = defaultdict(lambda: defaultdict(list))
    context = None
    for record in records:
        if record['kind'] == 'report_context':
            context = record['data']
        elif record['kind'] == 'sample':
            if record['sample'] in headers:
                raise ValueError('Duplicate sample header')
            headers[record['sample']] = record
        elif record['kind'] in ('stages', 'intervals'):
            for values in record['rows']:
                if len(values) != len(record['fields']):
                    raise ValueError('Malformed table row')
                rows[record['sample']][record['kind']].append(dict(zip(record['fields'], values)))
    result = []
    for sample_id in range(manifest['samples']):
        h = headers[sample_id]
        decoded = dict(h['decoded'])
        if (len(rows[sample_id]['stages']) != h['expected_stage_rows']
                or len(rows[sample_id]['intervals']) != h['expected_interval_rows']):
            raise ValueError('Sample row count mismatch')
        if h['records_present']:
            decoded['records'] = rows[sample_id]['stages']
        if h['intervals_present']:
            decoded['intervals'] = rows[sample_id]['intervals']
        result.append((h['metadata'], decoded))
    return context, result
