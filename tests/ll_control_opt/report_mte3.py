"""Bounded MTE3 backpressure evidence, matched by seed/rank/core and actual tasks."""
import argparse
from collections import defaultdict
import importlib.util
import math
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[2]
MAX_BYTES = 65535
CONDITIONS = ('off', 'observe', 'tx', 'rx')
MODE = dict(observe='observe', tx='mte3-tx', rx='mte3-rx', off='observe')


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT/relative)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


profile = load('mte3_report_profile', 'tests/ll_control_opt/profile_mte3.py')
base = load('mte3_report_common', 'tests/ll_control_opt/report_baseline.py')
timer = load('mte3_report_timer', 'deep_ep/utils/ll_stage_timer_mte3.py')
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
                timer_schema=16, timer_profile='mte3', timer_profile_id=7, send_cores=30,
                recv_cores=26, logical_group=4, high_throughput=False).items():
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
        if len(order) != 4 or set(order) != set(CONDITIONS):
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
            source_sha256=build['source_sha256'], stage_timer=enabled, stage_timer_schema=16,
            timer_profile='mte3', timer_profile_id=7, send_cores=30, recv_cores=26,
            active_aiv=56, logical_group=4, transport='mte', runtime_session='ll-mte',
            high_throughput_enabled=False, seed=seed, world_size=16,
            implementation='g4-r0-prefix', overlap_stage=3, core_protocol=19030,
            timer_mode=MODE[condition] if enabled else None, dispatch_only=True,
            stage_timer_slots=8, stage_timer_fields=8, stage_timer_recording='mte3-backpressure',
            stage_timer_points=list(timer.POINTS), stage_timer_wait_metrics=list(timer.METRICS),
            stage_timer_wait_metric_fields=['sum_ticks','max_ticks','calls','empty_control_ticks'],
            stage_timer_wait_metric_word_offset=32).items():
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
    for row in decoded['wait_metrics']:
        metrics[row['core']][row['metric']] = row
    rows = []
    for core, marks in points.items():
        spans = {name: (marks[end]-marks[start])/1000 for name, start, end in timer.PAIRS
                 if start in marks and end in marks}
        if spans['KERNEL'] <= 0:
            raise ValueError('Nonpositive MTE3 work span')
        rows.append(dict(rank=meta['rank'], iteration=meta['iteration'], core=core,
                         points=marks, spans=spans, work_us=spans['KERNEL'],
                         metrics=metrics[core], lanes=decoded['cores'][core]))
    return rows


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
        raise ValueError('Missing, extra or misplaced MTE3 rank/iteration samples')
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
        for key, value in dict(schema=16, active_aiv=56, send_cores=30,
                boundary_mode=MODE[condition], timer_profile='mte3', timer_profile_id=7,
                recording='mte3-backpressure').items():
            expect(decoded.get(key), value, 'decoded.'+key)
        rows.extend(core_rows(meta, decoded))
    return rows, result, active


def core_means(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['rank'], row['core']].append(row)
    means = {}
    for key, group in grouped.items():
        entry = dict(work_us=statistics.mean(row['work_us'] for row in group), metrics={},
                     lanes=group[0]['lanes'])
        for name in timer.METRICS:
            entry['metrics'][name] = {field: statistics.mean(row['metrics'][name][field] for row in group)
                for field in ('sum_us', 'max_us', 'calls', 'empty_control_us')}
        entry['spans'] = {name: statistics.mean(row['spans'][name] for row in group)
                          for name, _, _ in timer.PAIRS if all(name in row['spans'] for row in group)}
        means[key] = entry
    return means


def validate_packet_counts(conditions):
    # The fixed ST FP8 path is prequantized: one TX_INPUT_READY MTE2_S per packet.
    # RX_PREPARE_VS is a task-table preparation fence, not a packet count.
    reference = None
    for label in ('observe', 'tx', 'rx'):
        counts = defaultdict(set)
        for row in conditions[label][0]:
            rank, core = row['rank'], row['core']
            tx_packets = row['metrics']['TX_INPUT_READY']['calls']
            rx_packets = row['metrics']['RX_PAYLOAD_MTE2S']['calls']
            send_lanes, recv_lanes = row['lanes']['send_lanes'], row['lanes']['recv_lanes']
            if (tx_packets and not send_lanes) or (rx_packets and not recv_lanes):
                raise ValueError('Packet metrics require actual lane metadata')
            # Validate the raw actual layout bit before applying overlap reuse
            # formulas; a serial fallback has different first-lane semantics.
            if not row['lanes']['overlap_enabled'] or send_lanes > 6 or recv_lanes > 6 or (send_lanes and recv_lanes and send_lanes != recv_lanes):
                raise ValueError('Actual lanes disagree with the fixed-shape overlap layout')
            for metric in ('RX_PREPARE_VS', 'RX_FINAL_ALL'):
                expect(row['metrics'][metric]['calls'], int(rx_packets > 0), 'complete receive wait coverage '+metric)
            expected = dict(TX_REUSE_MTE3=max(tx_packets-send_lanes, 0),
                            TX_END_MTE3=int('SOURCE_SUBMIT_DONE' in row['points']),
                            RX_REUSE_MTE3=max(rx_packets-recv_lanes, 0), RX_END_MTE3=int(rx_packets > 0))
            for metric, calls in expected.items():
                selected = core < 30 and ((label=='tx' and metric.startswith('TX_')) or
                                          (label=='rx' and metric.startswith('RX_')))
                expect(row['metrics'][metric]['calls'], calls if selected else 0,
                       f'intervention coverage rank{rank}/core{core}/'+metric)
            for metric in ('TX_INPUT_READY', 'RX_PAYLOAD_MTE2S'):
                counts[rank, core, metric].add(row['metrics'][metric]['calls'])
            counts[rank, core, 'send_lanes'].add(send_lanes)
            counts[rank, core, 'recv_lanes'].add(recv_lanes)
            counts[rank, core, 'overlap_enabled'].add(row['lanes']['overlap_enabled'])
        if any(len(value) != 1 for value in counts.values()):
            raise ValueError('Packet counts or actual lanes changed across repeated fixed-route captures: '+label)
        counts = {key: next(iter(value)) for key, value in counts.items()}
        if reference is None:
            reference = counts
        elif reference != counts:
            raise ValueError('Per-rank/core packet counts or actual lanes differ across MTE3 modes')


def active_mean(condition):
    active = condition[2]
    return active['summary']['mean_kernel_us'] if active['status']=='available' else None


def percent(value, reference):
    return 100*(value/reference-1) if value is not None and reference else None


def describe_seed(run, conditions):
    seed = run['seed']; lines = ['', f'SEED requested={seed}, measured_route={seed+2}',
                                 'Execution order: '+','.join(run['condition_order'])]
    validate_packet_counts(conditions)
    baseline = conditions['off'][1]['dispatch']['mean_kernel_us']
    observe = active_mean(conditions['observe'])
    table(lines, 'ACTUAL FULL DEVICE TASKS: OFF rank-tail pooled mean; ON actual attached captures',
          ('condition', 'task_us', 'vs_OFF_pct', 'vs_OBSERVE_pct', 'status'),
          [('off', baseline, 0, None, 'available')] +
          [(label, active_mean(conditions[label]), percent(active_mean(conditions[label]), baseline),
            percent(active_mean(conditions[label]), observe), conditions[label][2]['status'])
           for label in ('observe', 'tx', 'rx')])
    for label in ('observe', 'tx', 'rx'):
        if conditions[label][2]['status'] != 'available':
            lines.append(label+' profiler unavailable: '+ '; '.join(
                str(row['rank'])+':'+str(row.get('reason', 'unavailable'))[:160]
                for row in conditions[label][2]['ranks'] if row['status']!='available'))
    means = {label: core_means(conditions[label][0]) for label in ('observe', 'tx', 'rx')}
    scalar = []; exposure = []; spans = []; packets = []
    for label in means:
        for role, predicate in (('S0..29', lambda core:core<30), ('R0..25', lambda core:core>=30)):
            group = [row for key, row in means[label].items() if predicate(key[1])]
            for name in timer.METRICS[:4]:
                present = [row for row in group if row['metrics'][name]['calls']]
                scalar.append((label, role, name, len(present), len(group),
                    median(row['metrics'][name]['sum_us'] for row in present),
                    median(row['metrics'][name]['calls'] for row in present),
                    median(row['metrics'][name]['max_us'] for row in present),
                    median(100*row['metrics'][name]['sum_us']/row['work_us'] for row in present)))
            for name, _, _ in timer.PAIRS:
                present = [row for row in group if name in row['spans']]
                spans.append((label, role, name, len(present), median(row['spans'][name] for row in present)))
            packets.append((label, role, sum(row['metrics']['TX_INPUT_READY']['calls'] for row in group),
                            sum(row['metrics']['RX_PAYLOAD_MTE2S']['calls'] for row in group),
                            min(row['lanes']['send_lanes'] for row in group),max(row['lanes']['send_lanes'] for row in group),
                            min(row['lanes']['recv_lanes'] for row in group),max(row['lanes']['recv_lanes'] for row in group)))
        for name in timer.METRICS[4:]:
            present = [row for key, row in means[label].items() if key[1]<30 and row['metrics'][name]['calls']]
            if present:
                exposure.append((label, name, len(present),
                    median(row['metrics'][name]['sum_us'] for row in present),
                    median(row['metrics'][name]['empty_control_us'] for row in present),
                    median(row['metrics'][name]['sum_us']-row['metrics'][name]['empty_control_us'] for row in present),
                    median(row['metrics'][name]['calls'] for row in present),
                    median(row['metrics'][name]['max_us'] for row in present)))
    table(lines, 'ORIGINAL SCALAR WAITS: per rank/core capture means, then median; same-core work=INIT_DONE->WORK_DONE',
          ('condition', 'initial_role', 'metric', 'executed_cores', 'selected_cores', 'sum_us_p50',
           'calls_p50', 'max_us_p50', 'same_core_work_pct_p50'), scalar)
    table(lines, 'S-CORE MTE3 EXPOSURE: raw and immediate empty control; signed difference is intervention-only',
          ('condition', 'metric', 'executed_cores', 'raw_sum_us_p50', 'control_sum_us_p50',
           'signed_excess_us_p50', 'calls_p50', 'max_us_p50'), exposure)
    transfer = []
    for label in ('tx', 'rx'):
        for name in timer.METRICS[:4]:
            keys = [key for key in means['observe'] if key[1]<30]
            deltas = [(means[label][key]['metrics'][name]['sum_us']-
                       means['observe'][key]['metrics'][name]['sum_us']) for key in keys]
            transfer.append((label, name, len(keys), statistics.mean(deltas), median(deltas),
                             sum(delta < 0 for delta in deltas), sum(delta > 0 for delta in deltas)))
    table(lines, 'PAIRED WAIT TRANSFER: condition minus OBSERVE at the SAME seed/rank/core, after capture averaging',
          ('condition', 'original_metric', 'S_cores', 'delta_us_mean', 'delta_us_p50', 'decreased', 'increased'), transfer)
    critical = []
    for rank in range(16):
        # Select ONCE from OBSERVE, then keep this same rank/core for both
        # interventions. There is no cross-run timestamp subtraction/alignment.
        key = max((key for key in means['observe'] if key[0]==rank),
                  key=lambda key:means['observe'][key]['work_us'])
        observed = means['observe'][key]
        original = [observed['metrics'][name]['sum_us'] for name in timer.METRICS[:4]]
        row = [rank, key[1], 'S/direct' if key[1]<30 else 'R/indirect-only', observed['work_us'], *original]
        for label in ('tx', 'rx'):
            current = means[label][key]
            row += [current['work_us'], sum(current['metrics'][name]['sum_us'] for name in timer.METRICS[:4])-sum(original),
                    sum(current['metrics'][name]['sum_us'] for name in timer.METRICS[4:]),
                    sum(current['metrics'][name]['empty_control_us'] for name in timer.METRICS[4:])]
        critical.append(row)
    table(lines, 'OBSERVE-LONGEST CORE HELD FIXED: select max capture-mean work per rank ONCE; match same core in TX/RX, no clock alignment',
          ('rank','core','intervention_scope','OBS_work_us',*('OBS_'+name+'_us' for name in timer.METRICS[:4]),
           'TX_work_us','TX_original_wait_delta_us','TX_forced_raw_us','TX_forced_control_us',
           'RX_work_us','RX_original_wait_delta_us','RX_forced_raw_us','RX_forced_control_us'), critical)
    table(lines, 'MILESTONE SPANS: same rank/core means then median; overlapping spans MUST NOT be added',
          ('condition', 'initial_role', 'span', 'present_cores', 'us_p50'), spans)
    table(lines, 'PACKET INVARIANCE: exact per rank/core equality validated across every capture and mode',
          ('condition', 'initial_role', 'TX_input_calls_total_per_capture', 'RX_payload_calls_total_per_capture',
           'send_lanes_min','send_lanes_max','recv_lanes_min','recv_lanes_max'), packets)
    if observe is None:
        lines.append('Interpretation: OBSERVE full-task timing unavailable; no perturbation or primary-bottleneck conclusion.')
    else:
        inflated = [label for label in ('tx', 'rx') if active_mean(conditions[label]) is not None and
                    percent(active_mean(conditions[label]), observe) > 10]
        if inflated:
            lines.append('Interpretation: '+','.join(inflated)+' task growth exceeds the descriptive 10% flag; '
                         'this is mechanism exposure under strong serialization, NOT a quantitative baseline attribution.')
        lines.append('Evidence rule: repeated same-core original-wait reduction plus MTE3 exposure can support dependency transfer. '
                     'Task growth alone, or a large forced MTE3 wait alone, cannot prove MTE3 is the baseline main bottleneck. '
                     'No automatic main-bottleneck verdict is produced.')
    return lines


def render(directory):
    directory = Path(directory).resolve(); plan = read(directory/'profile-plan.json'); validate_plan(plan)
    lines = ['MTE3 BACKPRESSURE DIAGNOSTIC REPORT',
        f'Diagnostic source={plan["revision"]}; business reference={plan["baseline_reference"]}',
        f'Runner={plan.get("runner_revision", "unknown")}; SDK={plan["dependency"]["shmem_revision"]}',
        'EP16 B=M256 H3584 K8 E384 prequantized FP8 dispatch only; G4 S30/R26; HT OFF; schema16/profile7; 1GHz.',
        f'OFF=10 warmup/100 tasks/rank tail50. Each ON mode=10 warmup/100 settle/{plan["timer_iters"]} actual attached captures.',
        'Exp + scalar all_reduce precedes every measured/settle/capture iteration; warmup has no prelude.',
        'All16 ranks/captures/correctness/loaded native+kernel+SDK hashes validated.',
        'OBSERVE instruments existing real Scalar waits without added barriers. TX and RX each add MTE3_S on S cores only.',
        'MTE3_S includes previously queued work and producer dependencies, not only transfer, remote arrival, or network congestion.',
        'Immediate repeated MTE3_S empty control has no newly issued transfer. Raw-minus-control stays signed and is NOT baseline latency.',
        'Forced waits change overlap and progress. Original wait costs may move to the inserted wait. These rows are NOT speedup estimates.',
        'Packet identity: fixed ST FP8 TX_INPUT_READY.calls and RX_PAYLOAD_MTE2S.calls only. No poll snapshot/empty-poll metrics are collected.',
        'Raw actual overlap=true and paired lanes<=6 verified on every core/capture; packet/lane counts must match across modes.',
        'Intervention coverage validated: reuse=max(packets-lanes,0); TX end follows SOURCE_SUBMIT_DONE, RX end requires receive packets.',
        'TX_INPUT_READY is original MTE2_S on this prequantized FP8 path; internal quantizing variants use their original V_S instead.',
        'RX_PREPARE_VS is the original V_S at the end of receive task-table Build, at most once per nonempty receive core.',
        'Metrics: mean across captures per seed/rank/core, then median across selected cores; max_us is mean of each capture maximum.',
        'The four original Scalar-wait scopes are disjoint and summed in the fixed-core table; milestone spans may not be summed.',
        'Missing/not-executed metrics=NA in wait summaries; observed calls0 is retained for paired matching. Ratios use the same core work.',
        'No cross-rank clocks are subtracted. No independently selected longest-core intervals are differenced. Overlapping spans are never summed.',
        'ENTRY->WORK_DONE excludes final export. Full task overhead comes only from actual profiler tasks uniquely matched to captures.',
        'SOURCE_SUBMIT_DONE can contain backpressure; PREFIX_WAIT is actual prefix entry->exit; RX_FINAL_ALL is existing final PIPE_ALL.',
        'Raw data remain on target. This single report omits raw ticks/per-packet timelines/all-core rows and keeps validated statistics.']
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
        lines.extend(describe_seed(run, conditions))
    data = ('\n'.join(lines)+'\n').encode('utf-8')
    if len(data) > MAX_BYTES:
        raise ValueError(f'Report is {len(data)} bytes, above {MAX_BYTES}; no output written or evidence truncated')
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path); parser.add_argument('--output', type=Path)
    args = parser.parse_args(); directory = (ROOT/args.directory).resolve()
    output = (ROOT/args.output).resolve() if args.output else directory/'mte3-report.txt'
    try:
        data = render(directory)
        with output.open('xb') as stream:
            stream.write(data)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f'MTE3 report rejected: {error}\n')
    print(f'Send only {output}; 1/10 files, {len(data)}/{MAX_BYTES} UTF-8 bytes')


if __name__ == '__main__':
    main()
