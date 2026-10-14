"""Audit all-rank profiler reports; never combine distinct task identities."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json
import math
from paths import project_path

from profile_common import describe


def audit_deployment(by_rank, deployment):
    """Validate actual node placement; output paths are deliberately not identity."""
    if deployment is None:
        return
    topology = deployment.get('topology', {})
    ips = topology.get('ips', [])
    local = topology.get('rank_num_per_server', 0)
    world = topology.get('world', 0)
    if (not deployment.get('run_id') or not isinstance(local, int) or local < 1
            or not isinstance(ips, list) or not ips or len(set(ips)) != len(ips)
            or world != local * len(ips) or len(by_rank) != world):
        raise ValueError('Invalid deployment identity')
    if world == 32 and len(ips) < 2:
        raise ValueError('32-rank run requires multiple physical hosts')
    boots = {}
    for rank, report in by_rank.items():
        provenance = report.get('rank_provenance', {})
        node = provenance.get('node_index')
        boot = provenance.get('boot_id')
        if (node != rank // local or report.get('device') != rank % local
                or not boot or report['world'] != world):
            raise ValueError('Rank/device/node placement mismatch')
        if node in boots and boots[node] != boot:
            raise ValueError('One node index spans multiple physical hosts')
        boots[node] = boot
    if set(boots) != set(range(len(ips))) or len(set(boots.values())) != len(boots):
        raise ValueError('Missing nodes or multiple node indices on the same physical host')


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
        # Output locations differ across machines; they are provenance, not workload identity.
        current['configuration'] = {k: v for k, v in report['configuration'].items() if k != 'output_dir'}
        current['source_sha256'] = report.get('source_sha256')
        current['deployment_identity'] = report.get('deployment_identity')
        current['measurement_scope'] = report.get('measurement_scope')
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
    audit_deployment(by_rank, identity['deployment_identity'])
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
        fanout = cfg.get('fanout')
        fanout = world-1 if fanout is None else fanout
        if not 1 <= fanout < world or (cfg['scenario'] != 3 and fanout != world-1):
            raise ValueError('Invalid communication fanout in report')
        edges = {1: 1, 2: world-1, 3: fanout, 4: world*(world-1)}[cfg['scenario']]
        wire = cfg['bytes']*edges
    elif kind in ('peer', 'shmem') and cfg['mode'] >= 2:
        wire = cfg['bytes']
    result = dict(**identity, all_rank_audit='passed', kernel_task_reports=tasks,
                  binary_sha256_by_rank={r: report['binary_sha256'] for r, report in by_rank.items()},
                  rank_provenance_by_rank={r: report.get('rank_provenance') for r, report in by_rank.items()},
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
    parser.add_argument('reports', nargs='+', type=project_path)
    parser.add_argument('--output', required=True, type=project_path)
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
