"""One bounded, readable first-pass diagnosis; full archives stay on the target."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIMIT = 60000
FORMAT = 'll-triage-report-v1'
PROFILE_ORDER = ('overview', 'prepare', 'prefix', 'send', 'recv', 'full')
ENVELOPES = frozenset(('KERNEL', 'INIT', 'TIMER_SETUP', 'BEFORE_FIRST_SEND',
    'BEFORE_FIRST_RECV_POLL', 'SEND_TOKENS', 'PREPARE_INPUT', 'SEND_COUNTS',
    'CAL_CUMSUM', 'BUILD_RECV_CUMSUM', 'WAIT_CUMSUM', 'PREFIX_COMPLETE',
    'SEND_SETUP', 'RECV_COPY', 'RECV_POLL', 'RECV_PREPARE'))


def _module(name):
    spec = importlib.util.spec_from_file_location('ll_triage_'+name, Path(__file__).with_name(name+'.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def _json(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _fits(text, limit=LIMIT):
    return len(text) <= limit and len(text.encode('utf-8'))+text.count('\n') <= limit


def _num(value):
    return '-' if value is None else format(value, '.6g')


def _pair(stats):
    # All missing values remain visibly unmeasured. n is an observed-sample count.
    return '-' if not stats or not stats[0] else f'{_num(stats[1])}/{_num(stats[3])}[{stats[0]}]'


def _lat(case, span, group):
    return next((row[2] for row in case['metrics']['latencies'] if row[:2] == [span, group]), None)


def _act(case, quantity, group):
    return next((row[2] for row in case['metrics']['activity'] if row[:2] == [group, quantity]), None)


def collect(suite):
    """Decode one existing case at a time; never unpack the large suite archive."""
    suite = Path(suite)
    metrics, transfer = _module('triage_metrics'), _module('transfer_report')
    plan_path = suite/'node_000/plan.json'
    plan = _read(plan_path) if plan_path.is_file() else {}
    cases, issues, seen = [], [], set()
    for path in sorted((suite/'analysis').rglob('transfer/manifest.json')):
        relative = path.parent.relative_to(suite).as_posix()
        profile = path.relative_to(suite/'analysis').parts[0]
        try:
            manifest = _read(path)
            source_id = (profile, manifest.get('bundle_id'), manifest.get('format'))
            if source_id[1] and source_id in seen:
                continue
            context, samples = transfer.restore_samples(path.parent)
            summary = metrics.summarize_case(context, samples)
            if summary['identity']['profile'] != profile:
                issues.append(f'Profile directory/record mismatch at {relative}')
            provenance = {key: (context.get('comparison') or {}).get(key) for key in (
                'implementation', 'transport', 'logical_group', 'core_protocol',
                'source_revision', 'source_sha256')}
            cases.append(dict(source=relative, source_bundle_id=manifest.get('bundle_id'),
                              metrics=summary, provenance=provenance))
            seen.add(source_id)
            del samples
        except (ValueError, OSError, KeyError, TypeError) as error:
            issues.append(f'Unreadable case {relative}: {type(error).__name__}: {error}')
    cases.sort(key=lambda c: (c['metrics']['identity'].get('B') or -1,
        PROFILE_ORDER.index(c['metrics']['identity']['profile']) if c['metrics']['identity']['profile'] in PROFILE_ORDER else 99,
        c['source']))
    for index, case in enumerate(cases, 1):
        case['id'] = f'C{index:03d}'
        coverage = case['metrics']['coverage']
        if coverage['complete_rank_iterations'] is not True or coverage['complete_kernel_cores'] is not True:
            issues.append(f"{case['id']}: rank/iteration/core coverage incomplete or unknown")
        if coverage.get('correctness', {}).get('passed') != coverage['samples']:
            issues.append(f"{case['id']}: correctness evidence incomplete")
    expected_bs = (plan.get('matrix') or {}).get('B', [])
    expected_profiles = [p['name'] for p in plan.get('stages', []) if p.get('timer')]
    for b in expected_bs:
        for profile in expected_profiles:
            if not any(c['metrics']['identity']['B'] == b and c['metrics']['identity']['profile'] == profile for c in cases):
                issues.append(f'Missing parsed case: B={b}, profile={profile}')
    if not expected_bs or not expected_profiles:
        issues.append('Expected B/profile matrix unknown; plan missing or incomplete')
    off_path = suite/'runs/off/summary.json'
    off = _read(off_path) if off_path.is_file() else {}
    off_rows = [dict(B=row.get('config', {}).get('B'), route=row.get('route'),
                    us=row.get('dispatch', {}).get('max_rank_mean_kernel_us'),
                    correctness=row.get('correctness'), comparison=row.get('comparison'))
                for row in off.get('results', []) if 'dispatch' in row]
    performance = _module('diagnose')._performance_rows(suite)
    for b in expected_bs:
        if not any(row['B'] == b and row['us'] is not None for row in off_rows):
            issues.append(f'Missing OFF performance: B={b}')
        for profile in expected_profiles:
            matches = [r for r in performance if r['B'] == b and r['profile'] == profile]
            if not matches or not any(r['status'] == 'available' for r in matches):
                issues.append(f'Missing/incomparable active-capture performance: B={b}, profile={profile}')
    return dict(format=FORMAT, suite=suite.name, matrix=plan.get('matrix'), cases=cases,
                off=off_rows, performance=performance, issues=issues,
                complete=bool(cases) and not issues)


def _essential(data):
    lines = ['LL FIRST-PASS DIAGNOSIS -- SEND THIS ONE FILE ONLY',
        f"Suite: {data['suite']}; cases={len(data['cases'])}; source coverage={'complete' if data['complete'] else 'INCOMPLETE/UNKNOWN'}",
        'This is a lossy analytical SUMMARY over ALL supplied samples, not a restorable trace archive.',
        'Full captures/archives stay on the target. No suite_* part files are needed for the first pass.',
        'Units: us; a/b[n] = p50/max over n observed rank-rounds. Missing=-, never measured zero.',
        'Span/stage tables: per-rank-round max across observed cores of the initial group (S/R).',
        'Activity tables: observed group counts or ratios per rank-round (not core-time maxima).',
        'INIT and TIMER_SETUP precede KERNEL; entry spans use actual same-core timestamps.',
        'OFF/ON rows are full profiler kernel tasks; internal spans exclude final uninstrumented sync/export.',
        'Different profiles are independent runs. Do not add costs, parents/children, or subtract cross-rank clocks.',
        'Probe overhead may slow preparation and reduce later waits. Check ON/OFF before interpreting waits.',
        'Active-core counts are positive packet counters observed; absent counter rows are not invented.',
        'Matrix: '+_json(data.get('matrix')),
        'Selection: '+_json(data.get('selection','all cases'))+'; coverage above refers to the source matrix.', '',
        'OFF PERFORMANCE (maximum of rank means, full device task)', 'B | route | us']
    for row in data['off']:
        lines.append(f"{row['B']} | {row.get('route')} | {_num(row['us'])}")
    lines += ['', 'ACTIVE-CAPTURE PERTURBATION (same-source identity checked)',
              'B | profile | OFF us | active ON us | ON/OFF | delta us | status']
    for row in sorted(data['performance'], key=lambda r: (r.get('B') or -1, r['profile'])):
        status = 'ok' if row['status'] == 'available' else 'unavailable'
        lines.append(f"{row['B']} | {row['profile']} | {_num(row.get('off_max_rank_mean_us'))} | "
                     f"{_num(row.get('active_max_rank_mean_us'))} | {_num(row.get('ratio'))} | {_num(row.get('delta_us'))} | {status}")
    lines += ['', 'CASE COVERAGE / SHAPE', 'id | B | profile | EP | M,H,K,E,dtype | S/R | samples/expected | cores | hz']
    for case in data['cases']:
        identity, coverage = case['metrics']['identity'], case['metrics']['coverage']
        shape = ','.join(str(identity.get(k)) for k in ('M', 'H', 'K', 'E', 'dtype'))
        lines.append(f"{case['id']} | {identity['B']} | {identity['profile']} | {identity['world_size']} | {shape} | "
                     f"{identity['send_cores']}/{identity['recv_cores']} | {coverage['samples']}/{coverage['expected_rank_iterations']} | "
                     f"{_pair(coverage['kernel_core_count'])} | {_num(identity['frequency_hz'])}")
    lines += ['', 'PREPARATION AND KERNEL SPANS (p50/max[n], not additive)',
              'id/group | KERNEL | INIT | TIMER_SETUP | INIT->first send | INIT->first probe | INIT->KERNEL end']
    for case in data['cases']:
        for group in ('S', 'R'):
            values = [_pair(_lat(case, name, group)) for name in (
                'KERNEL', 'INIT', 'TIMER_SETUP', 'entry_to_first_send', 'entry_to_first_probe', 'entry_to_kernel_end')]
            lines.append(f"{case['id']}/{group} | "+' | '.join(values))
    lines += ['', 'OBSERVED WORK AND READY RATIO (p50/max[n])',
              'id | S TX cores | R RX cores | S TX packets | R RX packets | S max packets/core | R max packets/core | R ready/probed | R probes/ready']
    for case in data['cases']:
        values = [_pair(_act(case, name, group)) for name, group in (
            ('tx_active_observed', 'S'), ('rx_active_observed', 'R'), ('tx_packets_observed', 'S'),
            ('rx_packets_observed', 'R'), ('tx_positive_max','S'), ('rx_positive_max','R'), ('ready_fraction', 'R'), ('probed_per_ready', 'R'))]
        lines.append(case['id']+' | '+' | '.join(values))
    return lines


def _details(case, count):
    if not count:
        return []
    lines = [f"{case['id']}: B={case['metrics']['identity']['B']} profile={case['metrics']['identity']['profile']}"]
    for group in ('S', 'R'):
        rows = [r for r in case['metrics']['stages'] if r[1] == group and r[0] not in ENVELOPES and r[3][0]]
        rows.sort(key=lambda r: (-r[3][1], -r[3][3], r[0]))
        lines.append(f"  {group} stage rank-round max p50/max[n]: "+
                     '; '.join(f'{r[0]}={_pair(r[3])}' for r in rows[:count]))
        slow = case['metrics']['slowest'].get(group)
        if slow:
            selected = [r for r in slow['phase_totals'] if r[0] not in ENVELOPES][:count]
            lines.append(f"  {group} slow core: rank={slow['rank']} iter={slow['iteration']} core={slow['core']} "
                f"{slow['span_kind']}={_num(slow['us'])}; "+
                '; '.join(f'{r[0]}={_num(r[1])}(calls={r[2]})' for r in selected))
    return lines


def render(data, limit=LIMIT):
    """Hard one-file cap. Reduce selected details fairly across every case."""
    if type(limit) is not int or not 4096 <= limit <= 60000:
        raise ValueError('Triage limit must be 4096..60000')
    essential = _essential(data)
    issues = data['issues'][:12]
    tail = ['', 'SOURCE ISSUES (full list retained locally):', *issues]
    if len(data['issues']) > len(issues):
        tail.append(f"{len(data['issues'])-len(issues)} more issues in local triage_summary.json")
    identities = {}
    for case in data['cases']:
        encoded = _json(case['provenance'])
        identities.setdefault(encoded, []).append(case['id'])
    # Preserve exact build/transport identity once, instead of 2800 repeated copies.
    tail += ['', 'BUILD IDENTITIES:']
    for encoded, ids in identities.items():
        tail.append(','.join(ids)+': '+encoded)
    tail += ['', 'LOCAL CASE LOCATORS (for a targeted follow-up, not more files to send):']
    tail.extend(case['id']+' '+case['source'] for case in data['cases'])
    tail += ['', 'Exact follow-up only if requested: triage_extract.py <suite> --profile <p> --bs <B> --rank <r> --iteration <i> --core <c> --output-dir <newdir>',
             'Follow-up default ceiling:3 parts. Full archive is optional via --transfer-mode archive.', '']
    for count in (4, 3, 2, 1, 0):
        lines = essential + ['', f'SELECTED COSTS: top {count} non-envelope stages by median rank-round maximum per group/case; remaining stages stay locally.']
        for case in data['cases']:
            lines += _details(case, count)
        text = '\n'.join(lines+tail)
        if _fits(text, limit):
            return text
    # Unusual much larger matrices still get a single usable report, explicitly
    # retaining the OFF/ON table; no unbounded file splitting or silent truncation.
    lines = essential[:essential.index('CASE COVERAGE / SHAPE')] + [
        '', f"Full case detail for {len(data['cases'])} cases exceeds this report budget.",
        'Only the performance overview is shown; all case statistics remain in local triage_summary.json.',
        'Rerun triage_report.py with --bs and/or --profile for a narrower first-pass report.',
        *issues, '']
    text = '\n'.join(lines)
    if not _fits(text, limit):
        raise ValueError('Performance overview exceeds one file; select --bs/--profile explicitly')
    return text


def export_report(suite, *, bs=None, profiles=None, limit=LIMIT):
    suite = Path(suite)
    data = collect(suite)
    if bs is not None or profiles is not None:
        selected = lambda b, p: (bs is None or b in bs) and (profiles is None or p in profiles)
        data['cases'] = [c for c in data['cases'] if selected(c['metrics']['identity']['B'], c['metrics']['identity']['profile'])]
        data['performance'] = [r for r in data['performance'] if selected(r['B'], r['profile'])]
        data['off'] = [r for r in data['off'] if bs is None or r['B'] in bs]
        if not data['cases']:
            raise ValueError('No parsed cases match --bs/--profile; existing report was not changed')
        data['selection'] = dict(B=bs, profiles=profiles)
    content = render(data, limit)
    summary = _json(data)+'\n'
    directory = suite/'triage'/hashlib.sha256(summary.encode()).hexdigest()[:20]
    directory.mkdir(parents=True, exist_ok=True)
    (directory/'triage_summary.json').write_text(summary, encoding='utf-8')
    (directory/'SEND_FIRST.txt').write_bytes(content.encode('utf-8'))
    target = suite/'SEND_FIRST.txt'
    target.write_bytes(content.encode('utf-8'))
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('suite', type=Path)
    parser.add_argument('--bs', nargs='+', type=int)
    parser.add_argument('--profile', nargs='+', choices=PROFILE_ORDER)
    args = parser.parse_args(argv)
    suite = args.suite if args.suite.is_absolute() else ROOT/args.suite
    print(export_report(suite, bs=args.bs, profiles=args.profile))


if __name__ == '__main__': main()
