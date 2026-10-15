"""Audit all-rank profiler reports; never combine distinct task identities."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import csv
import json
import math
from paths import project_path

from profile_common import describe
from bandwidth_model import load_hardware, comm_edges, volume, theoretical_time, bandwidth_statistics


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


def analyze(paths, hardware_override=None, peak_override=None, peak_description=None):
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
        current['configuration'] = {k: v for k, v in report['configuration'].items() if k not in ('output_dir','hardware_config')}
        current['source_sha256'] = report.get('source_sha256')
        current['deployment_identity'] = report.get('deployment_identity')
        current['measurement_scope'] = report.get('measurement_scope')
        current['hardware_profile'] = report.get('hardware_profile')
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
        edges=len(comm_edges(world,cfg['scenario'],cfg.get('fanout'),cfg.get('targets'),cfg.get('sources')))
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
    if wire is not None and len(tasks)==1 and kind=='comm' and cfg['layer']==3:
        deployment=identity.get('deployment_identity') or {}
        local=(deployment.get('topology') or {}).get('rank_num_per_server')
        vol,edges=volume(world,cfg,local)
        hardware=hardware_override if hardware_override is not None else identity.get('hardware_profile')
        if peak_override is not None:
            hardware=dict(specification=dict(label=peak_description,model=dict(aggregate_GBps=peak_override)),
                          source='explicit aggregate peak override')
        theory=theoretical_time(vol,edges,local,hardware)
        stats=bandwidth_statistics(tasks[0]['rank_max_samples'],vol,theory)
        result['bandwidth_summary']=dict(volume=vol,theory=theory,reference=hardware,statistics=stats,
            capture_samples_per_rank=identity['capture_steps'],used_samples=len(tasks[0]['rank_max_samples']),
            captured_sent_bytes=wire*identity['capture_steps'],selected_window_sent_bytes=wire*len(tasks[0]['rank_max_samples']),
            volume_note='Payload only; excludes warmup, correctness checks, Exp and all_reduce traffic',
            sample_window=identity['sample_window'],task_time_basis='per-iteration rank-MAX of named kernel/task; not globally aligned job makespan')
    return result


def write_table(result, path):
    summary=result.get('bandwidth_summary',{})
    vol=summary.get('volume',{});stats=summary.get('statistics',{})
    columns=['kernel','task_type','samples','captured_samples','bs','hidden_size','element_bytes','reference_kind','reference_label','bytes_per_edge','job_sent_bytes','intra_host_sent_bytes',
             'inter_host_sent_bytes','theoretical_time_us','latency_percentile','time_us',
             'effective_GBps','reference_utilization_percent']
    with path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for task in result['kernel_task_reports']:
            latency=task['iteration_rank_max']
            for q in (50,90,99):
                row=dict(kernel=task['kernel'],task_type=task['task_type'],samples=latency['sample_count'],
                         latency_percentile=f'P{q}',time_us=latency[f'p{q}_us'])
                if summary:
                    row.update(captured_samples=summary['capture_samples_per_rank'],
                               reference_kind=summary['theory']['status'],
                               reference_label=summary['theory'].get('reference_label'))
                    row.update({k:(vol.get('shape') or {}).get(k) for k in ('bs','hidden_size','element_bytes')})
                    row.update({k:vol.get(k) for k in ('bytes_per_edge','job_sent_bytes','intra_host_sent_bytes','inter_host_sent_bytes')})
                    row['theoretical_time_us']=stats['theoretical_time_us']
                    row.update(next(x for x in stats['at_latency_percentiles'] if x['latency_percentile']==f'P{q}'))
                writer.writerow(row)
    print(path.read_text(),end='')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', nargs='+', type=project_path)
    parser.add_argument('--output', required=True, type=project_path)
    parser.add_argument('--peak-gbps', type=float)
    parser.add_argument('--peak-description')
    parser.add_argument('--hardware-config',type=project_path,help='Override captured hardware-reference snapshot')
    parser.add_argument('--csv-output',type=project_path)
    args = parser.parse_args()
    if args.peak_gbps is not None and (not math.isfinite(args.peak_gbps) or args.peak_gbps<=0 or not args.peak_description):
        parser.error('Require positive peak and matching topology/direction description')
    hardware=load_hardware(args.hardware_config) if args.hardware_config else None
    result = analyze(args.reports,hardware,args.peak_gbps,args.peak_description)
    if args.peak_gbps is not None:
        if not math.isfinite(args.peak_gbps) or args.peak_gbps <= 0 or not args.peak_description:
            parser.error('Require positive peak and matching topology/direction description')
        if 'aggregate_wire_gbps' not in result:
            parser.error('This report does not define one communication bandwidth')
        result.update(reference_peak_gbps=args.peak_gbps, peak_description=args.peak_description,
                      attainment=result['aggregate_wire_gbps']/args.peak_gbps)
    table=args.csv_output or args.output.with_suffix('.csv')
    if table==args.output:table=args.output.with_name(args.output.stem+'.table.csv')
    if args.output.exists() or table.exists():raise ValueError('Use fresh JSON and CSV output paths')
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
    write_table(result,table)


if __name__ == '__main__':
    main()
