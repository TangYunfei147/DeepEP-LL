"""One lossless compressed delivery for every case/profile in a diagnosis run.

Only parsed timer bundles and explicitly named performance reports are included.
Raw logs, environments, wheels, profiler databases and caches are never scanned.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re

FORMAT = 'll-timer-suite-v1'
LIMIT = 60000
GROUP_BYTES = 64 * 1024 * 1024


def _module(name):
    spec = importlib.util.spec_from_file_location('ll_suite_' + name, Path(__file__).with_name(name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(',', ':'), allow_nan=False)


def _fits(text,limit):
    return len(text)<=limit and len(text.encode('utf-8'))+text.count('\n')<=limit


def _load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _case_b(context, samples):
    values = set()
    for meta, _ in samples:
        value = (meta.get('config') or {}).get('B')
        if isinstance(value, int):
            values.add(value)
    for config in (context.get('configurations') or {}).values():
        if isinstance(config, dict) and isinstance(config.get('B'), int):
            values.add(config['B'])
    return sorted(values)


def collect_suite(suite, on_case=None):
    """Collect a deterministic, allowlisted payload; retain incomplete evidence."""
    suite = Path(suite)
    transfer = _module('transfer_report')
    plan_file = suite / 'node_000/plan.json'
    plan = _load(plan_file) if plan_file.is_file() else {}
    selected_plan = {key: plan[key] for key in ('format', 'matrix', 'nodes') if key in plan}
    selected_plan['stages'] = [
        {key: stage[key] for key in ('name', 'timer', 'profile') if key in stage}
        for stage in plan.get('stages', [])]
    issues, cases, files = [], [], {}
    if not plan:
        issues.append('Missing node_000/plan.json: expected matrix/profile coverage is unknown.')
    off = suite / 'runs/off/summary.json'
    if off.is_file():
        files['runs/off/summary.json'] = dict(kind='json', data=_load(off))
    else:
        issues.append('Missing OFF formal summary; no OFF baseline can be reconstructed.')
    performance = suite / 'PERFORMANCE_COMPARE.txt'
    if performance.is_file():
        files['PERFORMANCE_COMPARE.txt'] = dict(kind='text', data=performance.read_text(encoding='utf-8'))
    active_cases = {}
    for path in sorted((suite / 'runs').glob('*/case_*/*/active_capture_dispatch.json')):
        relative = path.relative_to(suite).as_posix()
        data = _load(path)
        files[relative] = dict(kind='json', data=data)
        profile = path.relative_to(suite / 'runs').parts[0]
        if data.get('status') == 'available':
            active_cases.setdefault(profile, set()).add((data.get('config') or {}).get('B'))
        else:
            issues.append(f'Active-capture full-task measurement is unavailable: {relative}')
    seen = {}
    for path in sorted((suite / 'analysis').rglob('transfer/manifest.json')):
        relative = path.parent.relative_to(suite).as_posix()
        profile = path.relative_to(suite / 'analysis').parts[0]
        try:
            manifest = _load(path)
            context, samples = transfer.restore_samples(path.parent)
            encoded = json.loads(transfer.encode_samples(context, samples))
        except (ValueError, KeyError, TypeError, OSError) as exc:
            issues.append(f'Unreadable/incomplete case {relative}: {type(exc).__name__}: {exc}')
            continue
        observed_profiles={decoded.get('timer_profile','full') for _,decoded in samples}
        if observed_profiles!={profile}:
            issues.append(f'Timer profile identity differs from directory {profile}: {sorted(observed_profiles)}; data retained.')
        # A case can be present more than once after offline re-analysis. Store
        # the same exact source payload only once, preserving every source name.
        identity = (profile, hashlib.sha256(_json(encoded).encode()).hexdigest())
        if identity in seen:
            seen[identity]['sources'].append(relative)
            continue
        coverage = manifest.get('source_sample_coverage') or {}
        coverage_complete = bool(coverage) and all(item.get('complete') is True for item in coverage.values())
        cores_complete = manifest.get('all_observed_kernel_cores_complete')
        if not coverage_complete:
            issues.append(f'Rank/iteration coverage incomplete or unknown: {relative}')
        if cores_complete is not True:
            issues.append(f'Kernel-core coverage incomplete or unknown: {relative}')
        case = dict(sources=[relative], profile=profile,
                          source_bundle_id=manifest.get('bundle_id'), B=_case_b(context, samples),
                          sample_count=len(samples), source_sample_coverage=coverage,
                          all_observed_kernel_cores_complete=cores_complete, payload=encoded)
        if on_case is not None:
            on_case(case, selected_plan)
            case = {key: value for key, value in case.items() if key != 'payload'}
        cases.append(case)
        seen[identity] = case
    expected_bs = set((selected_plan.get('matrix') or {}).get('B', []))
    profiles = [stage['name'] for stage in selected_plan['stages'] if stage.get('timer')]
    if not expected_bs or not profiles:
        issues.append('Expected B matrix or timer profiles are missing; suite coverage is unknown.')
    off_data = files.get('runs/off/summary.json', {}).get('data', {})
    off_bs = {(row.get('config') or {}).get('B') for row in off_data.get('results', [])
              if (row.get('dispatch') or {}).get('max_rank_mean_kernel_us') is not None}
    missing_off = sorted(expected_bs - off_bs)
    if missing_off:
        issues.append(f'Missing OFF full-task measurements: B={missing_off}')
    for profile in profiles:
        present = {value for case in cases if case['profile'] == profile for value in case['B']}
        missing = sorted(expected_bs - present)
        if missing:
            issues.append(f'Missing parsed timer cases for {profile}: B={missing}')
        missing_active = sorted(expected_bs - active_cases.get(profile, set()))
        if missing_active:
            issues.append(f'Missing active-capture full-task reports for {profile}: B={missing_active}')
    if not cases:
        issues.append('No valid parsed timer case is available yet.')
    return dict(format=FORMAT, suite_name=suite.name, plan=selected_plan, files=files, cases=cases,
                coverage=dict(complete=not issues, issues=issues),
                notes=[
                    'Each profile and case is an independent capture; do not sum profiles or merge their timelines.',
                    'Only recorded intervals are retained; cumulative stage totals do not reconstruct omitted intervals.',
                    'OFF profiler full-task durations are the performance baseline. ON full-task durations quantify perturbation.',
                    'Packaging preserves every supplied decoded field; missing source ranks/iterations/stages stay missing.',
                ])


def _overview(payload, manifest, directory, suite, limit):
    coverage = payload['coverage']
    lines = ['LL diagnosis: one compressed suite', '',
             f"Bundle: {manifest['bundle_id']}",
             f"Coverage: {'complete' if coverage['complete'] else 'INCOMPLETE / unknown; see issues below'}",
             f"Cases: {len(payload['cases'])}; samples: {sum(case['sample_count'] for case in payload['cases'])}",
             f"Payload: {manifest['uncompressed_bytes']} bytes; transferable parts: {manifest['transfer_bytes']} bytes in {manifest['part_count']} files / {len(manifest['groups'])} bundles.",
             'Send this overview and only the part files below. No per-case priority/full files or separate performance CSV are needed.',
             'OFF and active-capture performance reports, identities, raw decoded metrics and recorded intervals are already included.',
             'Do not add costs from different profiles or merge their timestamps.', '', 'Files to send:']
    for group in manifest['groups']:
        ordinal, count = group['group'], group['part_count']
        if count <= 8:
            names = [f'suite_{ordinal:03d}_part_{index:03d}_of_{count:03d}.txt' for index in range(1, count+1)]
        else:
            names = [f'suite_{ordinal:03d}_part_{{001..{count:03d}}}_of_{count:03d}.txt ({count} files; every ordinal required)']
        lines.extend((directory / name).relative_to(suite).as_posix() for name in names)
    if coverage['issues']:
        lines += ['', 'Coverage issues (full list also retained in payload):']
        for issue in coverage['issues']:
            if len(('\n'.join(lines + [issue]) + '\n').encode('utf-8')) > limit - 256:
                lines.append('Additional issues are retained in the compressed payload.')
                break
            lines.append(issue)
    lines += ['', 'Parts accept any order and duplicate pastes. SHA256 detects missing, modified and mixed bundles.',
              'Copy every listed group. One restore command verifies all groups before writing; the final group includes performance reports.',
              'Restore offline with: python tests/ll_stage_timer/transfer_report.py restore <pasted-parts.txt> --output-dir <fresh-directory>', '']
    text = '\n'.join(lines)
    if not _fits(text,limit):
        raise ValueError('Suite overview exceeds transfer limit; reduce the requested matrix')
    return text


def export_suite(suite, limit=LIMIT):
    """Stream cases into <=64 MiB groups, then expose one suite transfer index.

    A single large case stays indivisible and remains subject to the codec's
    explicit maximum. Only one raw group is retained in memory at a time.
    """
    suite = Path(suite)
    codec = _module('text_codec')
    pending, pending_bytes, groups, entries, files = [], 0, [], [], {}
    def flush(value):
        raw = _json(value).encode('utf-8')
        metadata, parts = codec.make_parts(raw, limit=limit)
        ordinal = len(groups) + 1
        for index, text in enumerate(parts, 1):
            name = f'suite_{ordinal:03d}_part_{index:03d}_of_{len(parts):03d}.txt'
            entry = dict(file=name, group=ordinal, chars=len(text), utf8_bytes=len(text.encode('utf-8')),
                         sha256=hashlib.sha256(text.encode('utf-8')).hexdigest(),crlf_bytes=len(text.encode('utf-8'))+text.count('\n'))
            entries.append(entry)
            files[name] = text
        groups.append(dict(group=ordinal, payload_sha256=hashlib.sha256(raw).hexdigest(),
                           uncompressed_bytes=len(raw), codec=metadata, part_count=len(parts)))
    def accept(case, selected_plan):
        nonlocal pending, pending_bytes
        size = len(_json(case).encode('utf-8'))
        if pending and pending_bytes + size > GROUP_BYTES:
            flush(dict(format=FORMAT, suite_name=suite.name, group=len(groups) + 1,
                       plan=selected_plan, cases=pending, files={},
                       coverage=dict(complete=None, issues=['Partial suite bundle; final bundle contains overall coverage and performance reports.'])))
            pending, pending_bytes = [], 0
        pending.append(case)
        pending_bytes += size
    payload = collect_suite(suite, on_case=accept)
    final = dict(payload, cases=pending, group=len(groups) + 1,
                 suite_case_index=payload['cases'], group_count=len(groups) + 1,
                 preceding_groups=[dict(group=group['group'], payload_sha256=group['payload_sha256']) for group in groups])
    flush(final)
    bundle_id = hashlib.sha256(_json(groups).encode('utf-8')).hexdigest()[:20]
    directory = suite / 'transfer' / bundle_id
    manifest = dict(format=FORMAT, bundle_id=bundle_id, groups=groups, limit=limit,
                    uncompressed_bytes=sum(group['uncompressed_bytes'] for group in groups),
                    transfer_bytes=sum(entry['utf8_bytes'] for entry in entries),
                    transfer_bytes_crlf=sum(entry['crlf_bytes'] for entry in entries),part_count=len(entries),
                    complete=payload['coverage']['complete'], output=directory.relative_to(suite).as_posix(),
                    send_first=['00_overview.txt', 'suite_*_part_*.txt'], part_index=[])
    # A large suite can have many transfer parts. Its local hash index must
    # obey the same terminal bound without duplicating every filename in the
    # readable overview or top-level manifest.
    index_rows = []
    def index_flush():
        name = f'part_index_{len(manifest["part_index"])+1:03d}.json'
        content = _json(dict(format=FORMAT, bundle_id=bundle_id, parts=index_rows)) + '\n'
        files[name] = content
        manifest['part_index'].append(dict(file=name, sha256=hashlib.sha256(content.encode()).hexdigest(), entries=len(index_rows)))
    for entry in entries:
        if index_rows and len(_json(dict(format=FORMAT, bundle_id=bundle_id, parts=index_rows+[entry])).encode()) > limit-2:
            index_flush()
            index_rows = []
        index_rows.append(entry)
    if index_rows:
        index_flush()
    files['00_overview.txt'] = _overview(payload, manifest, directory, suite, limit)
    files['manifest.json'] = _json(manifest) + '\n'
    for name, text in files.items():
        if not _fits(text,limit):
            raise ValueError(f'{name} exceeds transfer limit')
        path = directory / name
        if path.exists() and path.read_bytes() != text.encode('utf-8'):
            raise ValueError(f'Existing suite export differs; not overwriting {path}')
    directory.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (directory / name).write_bytes(text.encode('utf-8'))
    return manifest


def _report_path(value):
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or '\\' in value or ':' in value:
        raise ValueError('Unsafe report path in suite payload')
    if value in ('runs/off/summary.json', 'PERFORMANCE_COMPARE.txt'):
        return path
    if re.fullmatch(r'runs/[A-Za-z0-9_.-]+/case_[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/active_capture_dispatch\.json', value):
        return path
    raise ValueError('Unrecognized report path in suite payload')


def restore_payload(payload, output, trace_iteration=0, render=True, _validated=False):
    """Restore CPU analysis inputs and observed traces into a fresh directory."""
    if isinstance(payload, (bytes, str)):
        payload = json.loads(payload)
    if payload.get('format') != FORMAT:
        raise ValueError('Unsupported suite payload format')
    output = Path(output)
    if output.exists():
        raise ValueError('Restore output already exists; choose a fresh directory')
    transfer = _module('transfer_report')
    files = {}
    for name, item in payload.get('files', {}).items():
        relative = _report_path(name)
        if item.get('kind') == 'json':
            content = _json(item['data']) + '\n'
        elif item.get('kind') == 'text' and name == 'PERFORMANCE_COMPARE.txt':
            content = item['data']
        else:
            raise ValueError('Unsupported suite report encoding')
        files[Path('reports') / relative] = content
    case_index, decoded_cases = [], []
    for index, case in enumerate(payload.get('cases', [])):
        context, samples = transfer.decode_samples(_json(case['payload']).encode('utf-8'))
        if not _validated:
            # Run exactly the renderer/analysis that will be written before
            # creating any destination, including a late failing case.
            transfer.prepare_restored(context, samples, trace_iteration, render)
        directory = Path('cases') / f'{index:03d}'
        decoded_cases.append((directory, context, samples))
        case_index.append({key: value for key, value in case.items() if key != 'payload'} | dict(output=directory.as_posix()))
    output.mkdir(parents=True, exist_ok=False)
    for relative, content in files.items():
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode('utf-8'))
    for directory, context, samples in decoded_cases:
        transfer.write_restored(output / directory, context, samples,
                                trace_iteration=trace_iteration, render=render)
    # A completeness marker is published only after all case files succeed.
    (output / 'suite.json').write_text(_json({key: value for key, value in payload.items() if key not in ('cases', 'files')} |
                                            dict(cases=case_index)) + '\n', encoding='utf-8')
    return output


def _validate_groups(reader, input_count, trace_iteration, render):
    """First pass: retain identities/indexes only, never decoded matrix payloads."""
    parsed = {}
    transfer = _module('transfer_report')
    for input_index in range(input_count):
        status, raw = reader(input_index)
        if raw is None:
            raise ValueError('Missing suite text parts')
        digest = hashlib.sha256(raw).hexdigest()
        if not status.get('complete') or status.get('raw_sha256') != digest:
            raise ValueError('Unverified or modified suite group')
        payload = json.loads(raw)
        ordinal = payload.get('group')
        if payload.get('format') != FORMAT or type(ordinal) is not int or ordinal < 1:
            raise ValueError('Invalid suite group identity')
        if ordinal in parsed and parsed[ordinal][0] != digest:
            raise ValueError('Conflicting suite groups with the same ordinal')
        for name, item in payload.get('files', {}).items():
            _report_path(name)
            if item.get('kind') not in ('json', 'text') or (item.get('kind') == 'text' and name != 'PERFORMANCE_COMPARE.txt'):
                raise ValueError('Unsupported suite report encoding')
            if item.get('kind') == 'text' and not isinstance(item.get('data'), str):
                raise ValueError('Invalid suite report text')
        case_index = []
        for case in payload.get('cases', []):
            context, samples = transfer.decode_samples(_json(case['payload']).encode('utf-8'))
            transfer.prepare_restored(context, samples, trace_iteration, render)
            case_index.append({key: value for key, value in case.items() if key != 'payload'})
        metadata = {key: value for key, value in payload.items() if key not in ('cases', 'files')}
        metadata.update(cases=case_index, has_files=bool(payload.get('files')), input_index=input_index)
        parsed[ordinal] = (digest, metadata)
    finals = [(ordinal, item) for ordinal, item in parsed.items() if 'group_count' in item[1]]
    if len(finals) != 1:
        raise ValueError('Missing or conflicting final suite bundle; all bundle groups are required')
    final_ordinal, (_, final) = finals[0]
    count = final.get('group_count')
    if type(count) is not int or count < 1 or count != len(parsed) or final_ordinal != count or set(parsed) != set(range(1, count+1)):
        raise ValueError('Missing or extra suite bundle groups')
    expected_preceding = [dict(group=ordinal, payload_sha256=parsed[ordinal][0]) for ordinal in range(1, count)]
    if final.get('preceding_groups') != expected_preceding:
        raise ValueError('Suite predecessor hash mismatch; do not mix exports')
    all_cases = []
    for ordinal in range(1, count+1):
        _, payload = parsed[ordinal]
        if payload.get('suite_name') != final.get('suite_name') or payload.get('plan') != final.get('plan'):
            raise ValueError('Mixed suite name or matrix/profile identity')
        if ordinal < count and payload.get('has_files'):
            raise ValueError('Performance reports must appear only in the final suite bundle')
        for case in payload.get('cases', []):
            all_cases.append(case)
    index = final.get('suite_case_index')
    if not isinstance(index, list) or len(index) != len(all_cases):
        raise ValueError('Suite case index count mismatch')
    for actual, expected in zip(all_cases, index):
        if ({key: value for key, value in actual.items() if key != 'sources'} !=
                {key: value for key, value in expected.items() if key != 'sources'} or
                not set(actual.get('sources', [])).issubset(expected.get('sources', []))):
            raise ValueError('Suite case index identity mismatch')
    return parsed, final


def _restore_reader(reader, input_count, output, trace_iteration, render):
    output = Path(output)
    if output.exists():
        raise ValueError('Restore output already exists; choose a fresh directory')
    parsed, final = _validate_groups(reader, input_count, trace_iteration, render)
    count = final['group_count']
    for ordinal in range(1, count+1):
        _, raw = reader(parsed[ordinal][1]['input_index'])
        target = output if count == 1 else output / f'group_{ordinal:03d}'
        restore_payload(raw, target, trace_iteration, render, _validated=True)
    if count == 1:
        return output
    index = dict(format=FORMAT, suite_name=final.get('suite_name'), group_count=count,
                 coverage=final['coverage'], cases=final['suite_case_index'],
                 reports=f'group_{count:03d}/reports')
    (output / 'suite_index.json').write_text(_json(index) + '\n', encoding='utf-8')
    return output


def restore_groups(groups, output, trace_iteration=0, render=True):
    """Compatibility API for callers already holding a small decoded suite."""
    return _restore_reader(lambda index: groups[index], len(groups), output, trace_iteration, render)


def restore_text_groups(block_groups, output, trace_iteration=0, render=True):
    """Verify all groups first, then restore one decompressed group at a time.

    Raw decoded payloads are discarded between groups in both passes. Only the
    compressed paste blocks and small suite/case identities survive a pass.
    """
    codec = _module('text_codec')
    return _restore_reader(lambda index: codec.inspect_parts(block_groups[index]),
                           len(block_groups), output, trace_iteration, render)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('suite', type=Path)
    parser.add_argument('--limit', type=int, default=LIMIT)
    args = parser.parse_args(argv)
    manifest = export_suite(args.suite, args.limit)
    print(args.suite / manifest['output'] / '00_overview.txt')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
