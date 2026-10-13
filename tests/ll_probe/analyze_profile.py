"""Audit all-rank profiler reports; never combine distinct task identities."""
import argparse
import json
import math
from pathlib import Path

from profile_common import describe


def analyze(paths):
    if not paths:
        raise ValueError('No reports')
    by_rank = {}
    identity = None
    fields = ('world', 'kernel', 'warmup', 'capture_steps', 'sample_window', 'delay_shape',
              'reference_defaults', 'measurement', 'git_revision', 'configuration')
    for path in paths:
        report = json.loads(path.read_text())
        rank = report['rank']
        if rank in by_rank:
            raise ValueError('Duplicate rank')
        current = {k: report[k] for k in fields}
        current['source_sha256'] = report.get('source_sha256')
        if identity is not None and current != identity:
            raise ValueError('Mixed configurations or source revisions')
        identity = current
        # Resolve adjacent copied logs, not the source host's original absolute path.
        log = (path.parent/'native.log').read_text()
        if report.get('correctness_passed') is not True or 'PASS' not in log or 'FAIL' in log:
            raise ValueError('Missing native correctness/cleanup evidence')
        for group in report['kernel_statistics']:
            raw = group['kernel_us']
            if len(raw) != report['capture_steps'] or any(not math.isfinite(v) or v < 0 for v in raw):
                raise ValueError('Invalid raw capture samples')
            tail = report['sample_window']
            selected = raw[-tail:] if tail else raw
            if selected != group['selected_kernel_us']:
                raise ValueError('Incorrect tail selection')
        by_rank[rank] = report
    if sorted(by_rank) != list(range(identity['world'])):
        raise ValueError('Missing ranks')
    kind = identity['configuration']['kind']
    senders = [0] if kind == 'shmem' else list(by_rank)
    keys = [[(g['kernel_name'], g['task_type']) for g in by_rank[r]['kernel_statistics']] for r in senders]
    if not keys[0] or any(key != keys[0] for key in keys):
        raise ValueError('Missing/mismatched named task groups')
    if kind == 'shmem' and by_rank[1]['kernel_statistics']:
        raise ValueError('Receiver-only rank must not report a sender kernel')
    tasks = []
    for i, (name, task_type) in enumerate(keys[0]):
        per_rank = {r: by_rank[r]['kernel_statistics'][i]['selected_kernel_us'] for r in senders}
        maximum = [max(samples[j] for samples in per_rank.values()) for j in range(len(per_rank[senders[0]]))]
        tasks.append(dict(kernel=name, task_type=task_type,
                          per_rank={r: describe(samples) for r, samples in per_rank.items()},
                          iteration_rank_max=describe(maximum), rank_max_samples=maximum))
    cfg, world = identity['configuration'], identity['world']
    wire = None
    if kind == 'comm' and cfg['layer'] == 3:
        edges = {1: 1, 2: world-1, 3: world-1, 4: world*(world-1)}[cfg['scenario']]
        wire = cfg['bytes']*edges
    elif kind in ('peer', 'shmem') and cfg['mode'] >= 2:
        wire = cfg['bytes']
    result = dict(**identity, all_rank_audit='passed', kernel_task_reports=tasks,
                  binary_sha256_by_rank={r: report['binary_sha256'] for r, report in by_rank.items()},
                  timing_note='Local named task intervals. Rank-MAX is not globally synchronized makespan; API/Event diagnostics are excluded.')
    if len({report['binary_sha256'] for report in by_rank.values()}) > 1:
        result['build_warning'] = 'Rank binaries differ; inspect per-rank build/compiler identity before comparing performance.'
    if wire is not None and len(tasks) == 1:
        duration = tasks[0]['iteration_rank_max']['mean_us']
        if duration <= 0:
            raise ValueError('Cannot calculate bandwidth from zero task duration')
        result.update(job_wire_bytes=wire, aggregate_wire_gbps=wire/(duration*1000))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--peak-gbps', type=float)
    parser.add_argument('--peak-description')
    args = parser.parse_args()
    result = analyze(args.reports)
    if args.peak_gbps is not None:
        if not math.isfinite(args.peak_gbps) or args.peak_gbps <= 0 or not args.peak_description:
            parser.error('Require positive peak and matching topology/direction description')
        if 'aggregate_wire_gbps' not in result:
            parser.error('This report does not define one communication bandwidth')
        result.update(reference_peak_gbps=args.peak_gbps, peak_description=args.peak_description,
                      attainment=result['aggregate_wire_gbps']/args.peak_gbps)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)


if __name__ == '__main__':
    main()
