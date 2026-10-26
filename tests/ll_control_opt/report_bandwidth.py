"""Bounded Sending-stage bandwidth evidence, matched by seed/rank/core and actual tasks."""
import argparse
from collections import defaultdict
import importlib.util
import math
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[2]
MAX_BYTES = 65535
CONDITIONS = ('off', 'observe', 'complete')
MODE = dict(observe='observe', complete='bandwidth-complete', off='observe')


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT/relative)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


profile = load('bandwidth_report_profile', 'tests/ll_control_opt/profile_bandwidth.py')
base = load('bandwidth_report_common', 'tests/ll_control_opt/report_baseline.py')
timer = load('bandwidth_report_timer', 'deep_ep/utils/ll_stage_timer_bandwidth.py')
read, expect, table, median = base.read, base.expect, base.table, base.median
MEASUREMENT = base.MEASUREMENT


def validate_plan(plan):
    for key, value in dict(schema=1, revision=profile.DIAGNOSTIC_REVISION,
            baseline_reference=profile.BASELINE_REFERENCE, send_cores=30, recv_cores=26,
            shape=profile.suite.SHAPE).items():
        expect(plan.get(key), value, key)
    if not re.fullmatch('[0-9a-f]{40}', plan['revision']):
        raise ValueError('Diagnostic source must be an immutable commit')
    if plan.get('timer_iters') not in (3, 5):
        raise ValueError('Expected 3 or 5 complete diagnostic iterations')
    seeds = profile.suite.selected_seeds(argparse.Namespace(seeds=plan.get('seeds', [])))
    if len(seeds) > 3 or [run.get('seed') for run in plan.get('runs', [])] != seeds:
        raise ValueError('Expected one to three distinct matching run seeds')
    for variant in ('off', 'on'):
        build = plan['build_metadata'][variant]
        for key, value in dict(revision=plan['revision'], timer='OFF' if variant == 'off' else 'ON',
                timer_schema=17, timer_profile='bandwidth', timer_profile_id=8, send_cores=30,
                recv_cores=26, logical_group=4, high_throughput=False,timer_bytes_per_core=1024).items():
            expect(build.get(key), value, variant+'.'+key)
        for key in ('native_sha256', 'kernel_sha256', 'sdk_library_sha256',
                    'sdk_manifest_sha256', 'wheel_sha256', 'source_sha256'):
            if not re.fullmatch('[0-9a-f]{64}', build.get(key, '')):
                raise ValueError('Missing build identity: '+key)
        expect(build.get('shmem_revision'), plan['dependency']['shmem_revision'], 'SDK revision')
    for field in ('source_sha256', 'dependency_id', 'sdk_library_sha256', 'sdk_manifest_sha256', 'shmem_revision'):
        expect(plan['build_metadata']['off'][field], plan['build_metadata']['on'][field],
               'OFF/ON matching '+field)
    expect(plan['build_metadata']['off']['dependency_id'], plan['dependency']['dependency_id'], 'planned dependency identity')
    for run in plan['runs']:
        order = run.get('condition_order', [])
        if len(order) != 3 or set(order) != set(CONDITIONS):
            raise ValueError('Condition execution order must contain each condition once')


def validate_summary(directory, seed, plan, condition):
    saved = read(directory/'summary.json'); enabled = condition != 'off'
    for key, value in dict(status='completed', world_size=16, case_count=1, result_count=1).items():
        expect(saved.get(key), value, 'summary.'+key)
    config = saved['config']
    for key, value in dict(seed=seed, warmup=10, num_iters=100, perf_tail=50, route='normal',
            ll_stage_timer=enabled, ll_timer_mode=MODE[condition], skip_check=False,
            skip_perf_test=enabled, dispatch_only=True, correctness_repeats=3,
            use_1g_huge_pages=0, ll_profile_capture=enabled).items():
        expect(config.get(key), value, 'config.'+key)
    if enabled:
        expect(config.get('ll_timer_iters'), plan['timer_iters'], 'capture count')
        expect(config.get('ll_timer_frequency_hz'), 1_000_000_000, 'frequency')
    if len(saved['results']) != 1:
        raise ValueError('Expected one fixed-shape result')
    result = saved['results'][0]
    for key, value in dict(config=profile.suite.SHAPE, correctness='passed', case_id=0, route='normal').items():
        expect(result.get(key), value, 'case.'+key)
    identity = result['comparison']; build = plan['build_metadata']['on' if enabled else 'off']
    for key, value in dict(source_revision=plan['revision'], control_revision=plan['revision'],
            source_sha256=build['source_sha256'], stage_timer=enabled, stage_timer_schema=17,
            timer_profile='bandwidth', timer_profile_id=8, send_cores=30, recv_cores=26,
            active_aiv=56, logical_group=4, transport='mte', runtime_session='ll-mte',
            high_throughput_enabled=False, seed=seed, world_size=16,
            implementation='g4-r0-prefix', overlap_stage=3, core_protocol=19030,
            timer_mode=MODE[condition] if enabled else None, dispatch_only=True,
            stage_timer_slots=16, stage_timer_fields=8, stage_timer_recording='stage-bandwidth',
            stage_timer_points=list(timer.POINTS), stage_timer_bandwidth_stages=list(timer.STAGES),
            stage_timer_bandwidth_fields=list(timer.FIELDS), stage_timer_bandwidth_word_offset=32,
            stage_timer_bandwidth_record_words=16).items():
        expect(identity.get(key), value, 'runtime.'+key)
    if not enabled:
        perf = result['dispatch']; ranks = perf['ranks']; tail = []
        if len(ranks) != 16 or {row['rank'] for row in ranks} != set(range(16)):
            raise ValueError('Incomplete OFF profiler rank set')
        for row in ranks:
            values = row['kernel_us']
            if row['samples'] != 50 or len(values) != 100 or any(not math.isfinite(v) or v <= 0 for v in values):
                raise ValueError('Invalid OFF profiler samples')
            tail.extend(values[-50:])
            if not math.isclose(statistics.mean(values[-50:]), row['mean_us'], rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError('OFF rank mean does not match its raw tail')
        if not math.isclose(statistics.mean(tail), perf['mean_kernel_us'], rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError('OFF pooled mean does not match raw tails')
    return result, config


def core_rows(meta, decoded):
    points = {core: {} for core in range(56)}
    metrics = {core: {} for core in range(56)}
    for mark in decoded['milestones']:
        points[mark['core']][mark['point']] = mark['tick']
    for row in decoded['bandwidth_metrics']:
        metrics[row['core']][row['stage']] = row
    result = []
    for core, marks in points.items():
        work = (marks['WORK_DONE']-marks['INIT_DONE'])/1000
        if work <= 0:
            raise ValueError('Nonpositive bandwidth work span')
        result.append(dict(rank=meta['rank'], iteration=meta['iteration'], core=core,
                           points=marks, work_us=work, metrics=metrics[core], lanes=decoded['cores'][core]))
    return result

def read_condition(directory, seed, plan, condition):
    profile.validate_loaded_ranks(directory, plan['build_metadata']['off' if condition == 'off' else 'on'])
    result, config = validate_summary(directory, seed, plan, condition)
    if condition == 'off':
        if list(directory.rglob('sample_*.json')):
            raise ValueError('OFF run unexpectedly contains capture samples')
        return [], result, None
    active = base.validate_active(directory, 'dispatch', result, plan['timer_iters'])
    sample_root = directory/'case_00000/normal/ll_stage_timer/dispatch'
    expected = {sample_root/f'rank_{rank:03d}'/f'sample_{iteration:04d}.json'
                for rank in range(16) for iteration in range(plan['timer_iters'])}
    if set(directory.rglob('sample_*.json')) != expected:
        raise ValueError('Missing, extra or misplaced bandwidth rank/iteration samples')
    rows = []
    for path in sorted(expected):
        saved = read(path); meta = saved['metadata']
        rank = int(path.parent.name[5:]); iteration = int(path.stem[7:])
        for key, value in dict(rank=rank, world_size=16, iteration=iteration, run_id=directory.name,
                seed=seed, case_id=0, correctness='passed',
                config={**profile.suite.SHAPE, 'logical_group_size':4}, comparison=result['comparison']).items():
            expect(meta.get(key), value, 'sample.'+key)
        for key, value in dict(after_perf=False, warmup=10, settle_iters=100,
                diagnostic_iters=plan['timer_iters'], delay_rows=config['delay_rows'],
                delay_cols=config['delay_cols'], group_backend='hccl', expert_timer=False).items():
            expect(meta['sampling'].get(key), value, 'sampling.'+key)
        measured = active['ranks'][rank]; task = meta['active_capture_profile']
        for key, value in dict(status=measured['status'], all_ranks_status=active['status'],
                measurement=MEASUREMENT,
                kernel_us=measured['kernel_us'][iteration] if measured['status']=='available' else None,
                sample_index=iteration, expected_samples=plan['timer_iters'],
                sample_order='unique_finite_profiler_start_time').items():
            expect(task.get(key), value, 'sample active task.'+key)
        expect(saved['operation'], 'dispatch', 'operation')
        expect(saved['frequency_hz'], 1_000_000_000, 'frequency')
        decoded = timer.parse_records(saved['raw'], 'dispatch', saved['frequency_hz'])
        for key, value in dict(schema=17, active_aiv=56, send_cores=30,
                boundary_mode=MODE[condition], timer_profile='bandwidth', timer_profile_id=8,
                recording='stage-bandwidth').items():
            expect(decoded.get(key), value, 'decoded.'+key)
        rows.extend(core_rows(meta, decoded))
    return rows, result, active


BYTE_FIELDS = ('peer_bytes', 'self_bytes', 'peer_payload_bytes', 'self_payload_bytes',
               'peer_operations', 'self_operations')


def expected_control(rank, core, experts=384, ranks=16, send=30, recv=26, group=4):
    """Mirror the static ownership, counting actual copy calls at rank boundaries."""
    fields = dict.fromkeys(BYTE_FIELDS, 0)
    if core < send:
        return fields
    worker = core-send
    local = experts//ranks*group
    first = rank//group*local
    remote = experts-local
    begin = worker*(remote//recv)+min(worker, remote%recv)
    end = begin+remote//recv+int(worker < remote%recv)
    ranges = []
    if begin < min(end, first):
        ranges.append((begin, min(end, first)))
    if max(begin, first) < end:
        ranges.append((max(begin, first)+local, end+local))
    lo = worker*(local//recv)+min(worker, local%recv)
    hi = lo+local//recv+int(worker < local%recv)
    if lo < hi:
        ranges.append((first+lo, first+hi))
    per_rank = experts//ranks
    for begin, end in ranges:
        while begin < end:
            destination = begin//per_rank
            count = min(end-begin, per_rank-begin%per_rank)
            target = 'self' if destination == rank else 'peer'
            fields[target+'_bytes'] += count*32
            fields[target+'_operations'] += 1
            begin += count
    return fields


def data_layout(shape):
    align = lambda value, base: (value+base-1)//base*base
    scales = (shape['H']+127)//128*4
    useful = shape['H']+scales
    header = align(align(shape['H'], 256)+scales, 32)
    indices = header+32
    index_bytes = shape['num_topk']*8
    if indices%480+index_bytes > 480:
        indices = align(indices, 480)
    packet = (align(indices+index_bytes, 32)+479)//480*512
    return useful, packet


def validate_traffic(conditions, shape):
    """Validate exact per-core traffic before performing any time aggregation."""
    reference = None
    for condition in ('observe', 'complete'):
        snapshots = defaultdict(set)
        for row in conditions[condition][0]:
            rank, core = row['rank'], row['core']
            lanes = row['lanes']
            if not lanes['overlap_enabled'] or max(lanes['send_lanes'], lanes['recv_lanes']) > 6:
                raise ValueError('Actual lanes disagree with fixed-shape overlap layout')
            if lanes['send_lanes'] and lanes['recv_lanes'] and lanes['send_lanes'] != lanes['recv_lanes']:
                raise ValueError('Source and receiver overlap lane layout mismatch')
            snapshots[rank, core, 'lanes'].add((lanes['send_lanes'], lanes['recv_lanes'], lanes['overlap_enabled']))
            for stage, metric in row['metrics'].items():
                snapshots[rank, core, stage].add(tuple(metric[field] for field in BYTE_FIELDS))
                # Roles describe the initial partition; S cores join receive and relay.
                if stage in ('REMOTE_FIRST_HOP', 'SOURCE_DIRECT') and core >= 30 and metric['total_operations']:
                    raise ValueError('Remote source stage outside S core')
                if stage == 'GROUP_LOCAL' and core < 30 and metric['total_operations']:
                    raise ValueError('Group local stage outside R core')
                if stage == 'PREFIX_LOCAL' and core != 30 and any(metric[field] for field in BYTE_FIELDS):
                    raise ValueError('Prefix publishing outside R0')
                if stage == 'PREFIX_LOCAL' and metric['peer_bytes']:
                    raise ValueError('Prefix local traffic cannot be cross-rank')
                if stage in ('REMOTE_FIRST_HOP', 'SOURCE_DIRECT') and metric['self_bytes']:
                    raise ValueError('Remote source sends cannot target self')
                if stage == 'COUNT_CONTROL':
                    for field, expected in expected_control(rank, core).items():
                        expect(metric[field], expected, 'Exact count ownership '+field)
                if stage == 'PREFIX_LOCAL':
                    for field, expected in dict(self_bytes=2*shape['num_experts']*4+32 if core == 30 else 0,
                                               self_operations=3 if core == 30 else 0).items():
                        expect(metric[field], expected, 'Exact prefix publishing '+field)
                if stage in timer.STAGES[2:]:
                    payload, packet = data_layout(shape)
                    for target in ('peer', 'self'):
                        expect(metric[target+'_payload_bytes'], metric[target+'_operations']*payload,
                               stage+' exact useful payload bytes')
                        expect(metric[target+'_bytes'], metric[target+'_operations']*packet,
                               stage+' exact wire bytes')
                    if metric['total_operations'] and not (lanes['recv_lanes'] if stage == 'RELAY' else lanes['send_lanes']):
                        raise ValueError('Sending stage has no actual lanes')
        if any(len(value) != 1 for value in snapshots.values()):
            raise ValueError('Per-core issued traffic or lanes changed across fixed-route captures')
        snapshots = {key: next(iter(value)) for key, value in snapshots.items()}
        total_packets=sum(value[4]+value[5] for (_,_,stage),value in snapshots.items() if stage in timer.STAGES[2:])
        expect(total_packets,16*shape['B']*shape['num_topk'],'Global route-conserving data packet count')
        if reference is None:
            reference = snapshots
        elif snapshots != reference:
            raise ValueError('Per-core issued traffic or lanes changed across observe/completion conditions')


def active_mean(condition):
    active = condition[2]
    return active['summary']['mean_kernel_us'] if active['status'] == 'available' else None


def percent(value, reference):
    return 100*(value/reference-1) if value is not None and reference else None


def mean(values):
    values = [value for value in values if value is not None]
    return statistics.mean(values) if values else None


def stage_core_means(rows):
    grouped = defaultdict(list)
    for row in rows:
        for stage, metric in row['metrics'].items():
            grouped[row['rank'], row['core'], stage].append(metric)
    result = {}
    for key, group in grouped.items():
        entry = {field: statistics.mean(row[field] for row in group) for field in BYTE_FIELDS}
        entry.update(operations=entry['peer_operations']+entry['self_operations'],
                     intervals=statistics.mean(row['completed_intervals'] for row in group),
                     issue_us=mean(row['issue_span_us'] for row in group),
                     elapsed_us=mean(row['completed_elapsed_us'] for row in group),
                     pre_us=mean(row['pre_drain_us'] for row in group),
                     end_us=mean(row['end_drain_us'] for row in group),
                     control_us=mean(row['empty_control_us'] for row in group))
        for target in ('peer', 'self'):
            for payload in ('', '_payload'):
                size = sum(row[target+payload+'_bytes'] for row in group)
                elapsed = sum(row['completed_elapsed_ticks'] for row in group)
                entry[target+payload+'_gbps'] = size/elapsed if size and elapsed else None  # pinned 1GHz
        result[key] = entry
    return result


def normalized_rows(conditions):
    """Only complete same-rank task durations are aggregate throughput windows."""
    result = []
    for label in CONDITIONS:
        rows = conditions['observe' if label == 'off' else label][0]
        traffic = defaultdict(lambda: dict(peer_bytes=0, self_bytes=0, peer_payload_bytes=0, self_payload_bytes=0))
        for row in rows:
            for stage, metric in row['metrics'].items():
                total = traffic[row['rank'], row['iteration'], stage]
                for field in total:
                    total[field] += metric[field]
        active = conditions[label][2]
        rank_perf = {row['rank']: row['mean_us'] for row in conditions['off'][1]['dispatch']['ranks']}
        for stage in timer.STAGES:
            rates = defaultdict(list)
            for (rank, iteration, name), sizes in traffic.items():
                if name != stage:
                    continue
                task = rank_perf[rank] if label == 'off' else (
                    active['ranks'][rank]['kernel_us'][iteration] if active['ranks'][rank]['status'] == 'available' else None)
                for field, size in sizes.items():
                    rates[field].append(size/task/1000 if task is not None else None)
            # A partial rank set must not be silently pooled into an aggregate.
            available = label == 'off' or active['status'] == 'available'
            result.append((label, stage, *(mean(rates[field]) if available and any(rates[field]) else None for field in
                         ('peer_bytes', 'self_bytes', 'peer_payload_bytes', 'self_payload_bytes'))))
    return result


def describe_seed(run, conditions, shape):
    validate_traffic(conditions, shape)
    lines = ['', f'SEED requested={run["seed"]}, measured_route={run["measured_route_seed"]}',
             'Execution order: '+','.join(run['condition_order'])]
    baseline = conditions['off'][1]['dispatch']['mean_kernel_us']
    observed = active_mean(conditions['observe'])
    table(lines, 'ACTUAL FULL DEVICE TASKS: OFF rank-tail pooled mean; ON actual attached captures',
          ('condition', 'task_us', 'vs_OFF_pct', 'vs_OBSERVE_pct', 'status'),
          [('off', baseline, 0, None, 'available')]+
          [(label, active_mean(conditions[label]), percent(active_mean(conditions[label]), baseline),
            percent(active_mean(conditions[label]), observed), conditions[label][2]['status']) for label in ('observe', 'complete')])
    for label in ('observe', 'complete'):
        if conditions[label][2]['status'] != 'available':
            lines.append(label+' profiler unavailable: '+'; '.join(
                str(row['rank'])+':'+str(row.get('reason', 'unavailable'))[:160]
                for row in conditions[label][2]['ranks'] if row['status'] != 'available'))
    means = {label: stage_core_means(conditions[label][0]) for label in ('observe', 'complete')}
    traffic, rates, overhead, issue = [], [], [], []
    for stage in timer.STAGES:
        for role, selected in (('S0..29', lambda core: core < 30), ('R0..25', lambda core: core >= 30)):
            group = [row for (_, core, name), row in means['complete'].items() if name == stage and selected(core)]
            present = [row for row in group if row['operations']]
            if not present:
                continue
            traffic.append((stage, role, len(present), len(group),
                            *(sum(row[field] for row in group) for field in BYTE_FIELDS)))
            duration = sum(row['elapsed_us'] or 0 for row in present)
            peer_size = sum(row['peer_bytes'] for row in present)
            self_size = sum(row['self_bytes'] for row in present)
            peer_useful = sum(row['peer_payload_bytes'] for row in present)
            self_useful = sum(row['self_payload_bytes'] for row in present)
            rates.append((stage, role, median(row['elapsed_us'] for row in present),
                          median(row['intervals'] for row in present),
                          median(row['peer_gbps'] for row in present), peer_size/duration/1000 if duration and peer_size else None,
                          median(row['self_gbps'] for row in present), self_size/duration/1000 if duration and self_size else None,
                          peer_useful/duration/1000 if duration and peer_useful else None,
                          self_useful/duration/1000 if duration and self_useful else None))
            overhead.append((stage, role, median(row['pre_us'] for row in present),
                             median(row['end_us'] for row in present), median(row['control_us'] for row in present)))
            observed_group = [row for (_, core, name), row in means['observe'].items()
                              if name == stage and selected(core) and row['operations']]
            issue.append((stage, role, median(row['issue_us'] for row in observed_group)))
    table(lines, 'EXACT ISSUED TRAFFIC: total over16 ranks per capture; observe/completion per-core invariance checked',
          ('stage', 'initial_role', 'active_cores', 'selected_cores', *BYTE_FIELDS), traffic)
    table(lines, 'COMPLETION EFFECTIVE MOVEMENT GB/s: intervention only; per-core elapsed includes self and peer work',
          ('stage', 'initial_role', 'elapsed_us_p50', 'intervals_p50', 'peer_GBps_p50', 'peer_GBps_time_weighted',
           'self_GBps_p50', 'self_GBps_time_weighted', 'peer_useful_GBps_time_weighted', 'self_useful_GBps_time_weighted'), rates)
    table(lines, 'COMPLETION BOUNDARY COSTS: raw per-core means then median; no subtraction',
          ('stage', 'initial_role', 'pre_drain_us_p50', 'end_drain_us_p50', 'empty_control_us_p50'), overhead)
    table(lines, 'OBSERVE SUBMISSION SPANS: first-to-last issue only; NOT completed bandwidth',
          ('stage', 'initial_role', 'issue_span_us_p50'), issue)
    table(lines, 'FULL-TASK-NORMALIZED GB/s: per-rank total phase bytes / SAME rank complete task, then arithmetic mean',
          ('condition', 'stage', 'peer_GBps_mean', 'self_GBps_mean', 'peer_useful_GBps_mean', 'self_useful_GBps_mean'),
          normalized_rows(conditions))
    lines.append('Control useful payload=NA. Weighted phase rates=sum_bytes/sum_core_elapsed: average per-core effective movement, not rank/link aggregate.')
    lines.append('OFF normalized numerators come from the matched observe route; OFF denominators are each rank\'s actual tail50 mean, not instrumented phase time.')
    lines.append('Completion adds synchronization and changes overlap. Its phase rates cannot be assigned to uninstrumented baseline or used as a peak-attainment verdict.')
    return lines


def render(directory):
    directory = Path(directory).resolve(); plan = read(directory/'profile-plan.json'); validate_plan(plan)
    lines = ['DISPATCH SENDING-STAGE BANDWIDTH REPORT',
        f'Diagnostic source={plan["revision"]}; business reference={plan["baseline_reference"]}',
        f'Runner={plan.get("runner_revision", "unknown")}; SDK={plan["dependency"]["shmem_revision"]}',
        'EP16 B=M256 H3584 K8 E384 prequantized FP8 dispatch only; G4 S30/R26; HT OFF; schema17/profile8; 1GHz.',
        f'OFF=10 warmup/100 tasks/rank tail50. Each ON mode=10 warmup/100 settle/{plan["timer_iters"]} actual attached captures.',
        'Exp + scalar all_reduce precedes every measured/settle/capture iteration; warmup has no prelude.',
        'All16 ranks/captures/correctness/loaded native+kernel+SDK hashes and per-core exact traffic across modes validated.',
        'Stages: COUNT_CONTROL=count-record publishing; PREFIX_LOCAL=R0 local prefix workspace/ready publishing;',
        'REMOTE_FIRST_HOP=S first-hop packets; SOURCE_DIRECT=S direct terminal packets; GROUP_LOCAL=R local-group source packets; RELAY=receiver onward packets.',
        'Peer means destination rank differs from issuing rank; self means same-rank write. Same physical host is not the same as self.',
        'Wire bytes are actual DataCopy lengths, including packet flags/header/padding. Useful data bytes=H FP8 bytes+packed E8M0 scales.',
        'Data operations count packets; COUNT/PREFIX operations count DataCopy API calls, which can contain multiple32B records.',
        'Excluded: local final output, flag clearing, source mask/weight metadata output, receive reads. This is not all MTE3 traffic.',
        'OBSERVE adds no pipeline wait and records only issued traffic/submission spans. Issue spans are not communication completion time.',
        'COMPLETE pre-drains MTE3, timestamps first issue, waits MTE3 at interval end, then repeats an empty control.',
        'Source/control phases have at most one nonempty interval per core. Relay accumulates distinct forwarding intervals.',
        'Completed durations include packet production, scheduling, submission and MTE3 waiting, not only DMA or network travel.',
        'GB/s uses decimal1e9 bytes/s. Core rates use exact bytes/completed_elapsed; peer and self share the actual phase interval denominator.',
        'Report first averages captures per rank/core; duration p50 uses active phase cores; each rate p50 uses cores with that nonzero byte class. Weighted rates use sum bytes/sum all active phase core durations, never sum core rates.',
        'Rank normalization averages all16 rank rates, including zero contributions when a byte class exists. Different phase normalized rates can be added because they share a denominator.',
        'No cross-rank clocks are subtracted; no maximum core span is used as a fabricated rank phase window. Zero-byte/zero-time rate=NA.',
        'Pre-drain and empty-control costs are outside completed intervals; end-drain is inside. Controls are never subtracted.',
        'Full task perturbation includes timer export. No automatic bottleneck, optimization gain, or physical-link peak-attainment verdict.',
        'Raw ticks, per-core rows and raw task samples remain on target. This single bounded report retains exact traffic, coverage and defined summary statistics.']
    for variant in ('off', 'on'):
        lines += ['', variant.upper()+' build identity:']
        lines.extend(f'{key}={plan["build_metadata"][variant][key]}' for key in
                     ('source_sha256', 'native_sha256', 'kernel_sha256', 'sdk_library_sha256', 'sdk_manifest_sha256'))
    seen = set()
    for run in plan['runs']:
        expect(run.get('measured_route_seed'), run['seed']+2, 'measured route seed')
        conditions = {}
        for label in CONDITIONS:
            source = (directory/Path(run[label+'_directory'])).resolve()
            if not source.is_relative_to(directory) or source == directory or source in seen:
                raise ValueError('Run paths must be distinct children of the requested profile directory')
            seen.add(source)
            conditions[label] = read_condition(source, run['seed'], plan, label)
        lines.extend(describe_seed(run, conditions, plan['shape']))
    data = ('\n'.join(lines)+'\n').encode('utf-8')
    if len(data) > MAX_BYTES:
        raise ValueError(f'Report is {len(data)} bytes, above {MAX_BYTES}; no output written or evidence truncated')
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path); parser.add_argument('--output', type=Path)
    args = parser.parse_args(); directory = (ROOT/args.directory).resolve()
    output = (ROOT/args.output).resolve() if args.output else directory/'bandwidth-report.txt'
    try:
        data = render(directory)
        with output.open('xb') as stream:
            stream.write(data)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f'Bandwidth report rejected: {error}\n')
    print(f'Send only {output}; 1/10 files, {len(data)}/{MAX_BYTES} UTF-8 bytes')


if __name__ == '__main__':
    main()


