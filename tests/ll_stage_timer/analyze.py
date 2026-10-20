"""Aggregate validated LL stage traces per rank/core/stage; retain raw ticks."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('ll_stage_timer_parser',ROOT/'deep_ep/utils/ll_stage_timer.py')
timer=importlib.util.module_from_spec(spec);spec.loader.exec_module(timer)


def stats(values):
    values=sorted(values)
    def q(p):
        x=(len(values)-1)*p;lo=int(x);hi=min(lo+1,len(values)-1)
        return values[lo]+(values[hi]-values[lo])*(x-lo)
    return dict(samples=len(values),mean=statistics.mean(values),minimum=min(values),maximum=max(values),
                p50=q(.5),p90=q(.9),p99=q(.99))

def input_wait_report(samples):
    """Same-core/sample ratios only; no missing-as-zero or causal attribution."""
    buckets={}
    thresholds=dict(min_samples=3,send_share=0.20,kernel_share=0.10,
                    hot_sample_fraction=0.50,critical_core_fraction=0.90)
    for meta,decoded in samples:
        if decoded['operation']!='dispatch': continue
        cores={}
        for row in decoded['records']:
            cores.setdefault(row['core'],{})[row['stage']]=row
        longest=max((st.get('KERNEL',{}).get('total_ticks',0) for st in cores.values()),default=0)
        for core,st in cores.items():
            if 'SEND_TOKENS' not in st: continue
            quant=st.get('TOKEN_QUANT',{}).get('calls',0)>0
            name='TOKEN_QUANT_WAIT' if quant else 'TOKEN_INPUT_WAIT'
            bucket=buckets.setdefault((meta['rank'],core,name),dict(values=[],missing=0))
            packet=st.get('TOKEN_INPUT');wait=st.get(name)
            send=st['SEND_TOKENS']['total_ticks'];kernel=st.get('KERNEL',{}).get('total_ticks',0)
            if (decoded['schema']!=9 or not packet or not wait or
                    wait['calls']!=packet['calls'] or send<=0 or kernel<=0 or
                    not 0<=wait['total_ticks']<=min(send,kernel)):
                bucket['missing']+=1
                continue
            ticks=wait['total_ticks']
            bucket['values'].append(dict(iteration=meta['iteration'],ticks=ticks,calls=wait['calls'],
                send_share=ticks/send,kernel_share=ticks/kernel,
                critical=kernel>=thresholds['critical_core_fraction']*longest))
    rows=[]
    for (rank,core,stage),bucket in sorted(buckets.items()):
        values=bucket['values'];n=len(values)
        sufficient=n>=thresholds['min_samples'] and bucket['missing']==0
        hot=sum(v['send_share']>=thresholds['send_share'] and
                v['kernel_share']>=thresholds['kernel_share'] for v in values)
        candidate=hot/n>=thresholds['hot_sample_fraction'] if sufficient else None
        rows.append(dict(rank=rank,core=core,stage=stage,samples=n,missing_samples=bucket['missing'],
            status=('疑似瓶颈' if candidate else '未见明显证据') if sufficient else '数据不足',
            is_bottleneck_candidate=candidate,
            median_send_share=statistics.median(v['send_share'] for v in values) if n else None,
            median_kernel_share=statistics.median(v['kernel_share'] for v in values) if n else None,
            hot_sample_fraction=hot/n if n else None,
            critical_core_fraction=sum(v['critical'] for v in values)/n if n else None,
            weighted_ticks_per_call=sum(v['ticks'] for v in values)/sum(v['calls'] for v in values) if n else None))
    overall=('疑似瓶颈' if any(r['is_bottleneck_candidate'] is True for r in rows) else
             '数据不足' if not rows or any(r['is_bottleneck_candidate'] is None for r in rows) else '未见明显证据')
    return dict(status=overall,thresholds=thresholds,per_core=rows,
        interpretation='Heuristic candidates, not proven bottlenecks. MTE2_S measures residual input-path wait; '
        'V_S may include earlier vector work, not only quantization. Instrumented timing cannot prove prefetch benefit. '
        'Old/missing/inconsistent records are insufficient, never zero. Validate with timer-OFF A/B.')


def input_wait_markdown(report):
    lines=['# Dispatch 输入等待诊断','',f"总体筛查：{report['status']}（任一核命中即提示候选，不代表全局关键路径已证实）。",'',
           '启发式判断：至少3轮；至少半数有效轮次中，等待占同核 SEND_TOKENS ≥20% 且占同核 KERNEL ≥10%。',
           '疑似瓶颈不等于缺少预取已被证实；未见明显证据也不排除等待。单位为原始 ticks，未假定时钟频率。',
           'critical_core_fraction 表示该核时长达到同 rank/轮最长核的90%的轮次比例。','',
           '| rank | core | 等待阶段 | 结论 | 有效/缺失轮次 | 发送占比中位数 | kernel占比中位数 | 慢核轮次比例 | ticks/call |',
           '|---|---|---|---|---|---|---|---|---|']
    for row in report['per_core']:
        def fmt(key,percent=False):
            value=row[key]
            return '—' if value is None else (f'{value:.1%}' if percent else f'{value:.2f}')
        lines.append(f"| {row['rank']} | {row['core']} | {row['stage']} | {row['status']} | "
                     f"{row['samples']}/{row['missing_samples']} | {fmt('median_send_share',True)} | "
                     f"{fmt('median_kernel_share',True)} | {fmt('critical_core_fraction',True)} | {fmt('weighted_ticks_per_call')} |")
    if not report['per_core']: lines+=['','无 dispatch 发送样本，无法判断输入等待。']
    lines+=['','仅保留前两次实际区间；结论使用全量累计计数，不重建逐包流水。',
            'MTE2_S 是输入路径剩余等待；V_S 也可能等待此前向量工作。需用 timer OFF 的预取 A/B 确认收益。']
    return '\n'.join(lines)+'\n'


def analyze(paths, frequency_hz=None):
    configurations={};ranks={};groups={};identity=None;stage_max={};measured={};worker_roles={}
    wait_samples=[]
    for path in paths:
        saved=json.loads(path.read_text())
        meta=saved['metadata']
        if meta.get('correctness')!='passed':raise ValueError('Missing instrumented correctness evidence')
        report=timer.parse_records(saved['raw'],saved['operation'],frequency_hz if frequency_hz is not None else saved['frequency_hz'])
        wait_samples.append((meta,report))
        common=(meta['world_size'],meta['config'],(report['active_aiv'],report['send_cores']),report['boundary_mode'],report['frequency_hz'],
                meta.get('run_id'),meta.get('seed'),meta.get('case_id'),meta.get('sampling'),report['schema'],meta.get('comparison'))
        if identity is not None and common!=identity:raise ValueError('Mixed configurations or timer modes')
        identity=common
        op=report['operation'];rank=meta['rank'];iteration=meta['iteration']
        if not 0<=rank<meta['world_size'] or type(iteration) is not int or iteration<0:raise ValueError('Invalid rank/iteration')
        key=(op,rank)
        if iteration in ranks.setdefault(key,set()):raise ValueError('Duplicate rank/iteration')
        ranks[key].add(iteration)
        configurations[op]=meta['config']
        unit='us' if report['frequency_hz'] is not None else 'ticks'
        for row in report['records']:
            initial_role=row.get('initial_role',row['role'])
            key=(op,rank,row['core'],initial_role,row['stage'])
            if report['schema'] in (7,9):
                worker_roles.setdefault((op,rank,row['core']),set()).add(row['role'])
            duration=row['total_us'] if unit=='us' else row['total_ticks']
            groups.setdefault(key,[]).append((duration,row['calls']))
            measured[key]=row.get('time_measured',True)
            if not measured[key]: continue
            stage_key=(op,rank,row['stage'])
            per_iteration=stage_max.setdefault(stage_key,{})
            per_iteration[iteration]=max(per_iteration.get(iteration,0),duration)
    if identity is None:raise ValueError('No traces')
    for op in configurations:
        if {rank for kind,rank in ranks if kind==op}!=set(range(identity[0])):
            raise ValueError(f'Missing ranks for {op}')
        reference=ranks[(op,0)]
        if reference!=set(range(len(reference))) or any(ranks[(op,r)]!=reference for r in range(identity[0])):
            raise ValueError('Missing or misaligned diagnostic iterations')
        if identity[8] is not None:
            expected = identity[8].get('diagnostic_iters')
            if type(expected) is not int or expected < 1 or reference != set(range(expected)):
                raise ValueError('Incomplete diagnostic tail')
    rows=[]
    for (op,rank,core,role,stage), values in sorted(groups.items()):
        display_role='send_and_recv' if 'send_and_recv' in worker_roles.get((op,rank,core),set()) else role
        rows.append(dict(operation=op,rank=rank,core=core,role=display_role,initial_role=role,stage=stage,unit=unit,
                         **stats([v for v,n in values]),mean_calls=statistics.mean(n for v,n in values),
                         time_measured=measured[(op,rank,core,role,stage)]))
        if not rows[-1]['time_measured']:
            for name in ('mean','minimum','maximum','p50','p90','p99'): rows[-1][name]=None
    return dict(schema=identity[9],world_size=identity[0],boundary_mode=identity[3],frequency_hz=identity[4],comparison=identity[10],
                configurations=configurations,sampling=identity[8],per_rank_core_stage=rows,
                input_wait_bottleneck=input_wait_report(wait_samples),
                per_rank_stage_max=[dict(operation=op,rank=rank,stage=stage,unit=unit,**stats(list(values.values())))
                                    for (op,rank,stage),values in sorted(stage_max.items())],
                note='Per-call diagnostic stage totals; nested/parallel stages cannot be added. No cross-rank timestamp subtraction.')


def discover(source, latest=False):
    if latest:
        runs = [p for p in source.iterdir() if p.is_dir() and (p/'cases.json').is_file()]
        if not runs:
            raise ValueError('No run directories containing cases.json')
        # Manifest mtime is the run creation time, unaffected by later report directories.
        source = max(runs, key=lambda p: ((p/'cases.json').stat().st_mtime_ns, p.name))
    directories = sorted(p for p in source.rglob('ll_stage_timer')
                         if p.is_dir() and not (p/'summary.json').is_file())
    if source.name == 'll_stage_timer' or not directories:
        directories = [source]
    return source, directories


def expert_reports(paths, frequency_hz):
    spec=importlib.util.spec_from_file_location('expert_timer_parser',ROOT/'deep_ep/utils/ll_expert_timer.py')
    expert=importlib.util.module_from_spec(spec); spec.loader.exec_module(expert)
    rows, groups, presence = [], {}, set()
    for path in paths:
        saved=json.loads(path.read_text())
        if saved['operation'] != 'dispatch':
            if 'expert' in saved: raise ValueError('Expert records only supported for dispatch')
            continue
        payload=saved.get('expert'); meta=saved['metadata']; presence.add(payload is not None)
        if payload is None:
            if (meta.get('sampling') or {}).get('expert_timer'):
                raise ValueError('Missing requested expert records')
            continue
        if (payload['rank'],payload['world']) != (meta['rank'],meta['world_size']):
            raise ValueError('Expert sample identity mismatch')
        if payload['experts'] != meta['config'].get('num_experts',payload['experts']):
            raise ValueError('Expert count differs from case configuration')
        header=saved['raw'][0][0]
        if header[1] in (7,9): raise ValueError('Segment capture cannot contain legacy expert instrumentation')
        expected_send=header[5] >> 8 if header[1] in (5,6) else len(saved['raw'])//2
        if (header[1] in (5,6) and 'send_cores' not in payload) or payload.get('send_cores',expected_send)!=expected_send:
            raise ValueError('Expert/stage core split mismatch')
        if payload.get('shared_receive',False) != (header[1]==6):
            raise ValueError('Expert/stage receive participation mismatch')
        report=expert.parse_records(**payload, frequency_hz=frequency_hz)
        if report['active_aiv'] != len(saved['raw']) or report['boundary_mode'] != ('observe' if (saved['raw'][0][0][5] & 255)==1 else 'drain'):
            raise ValueError('Expert/stage core count or boundary mode mismatch')
        per_sample={}
        for row in report['records']:
            rows.append(dict(iteration=meta['iteration'], **row))
            key=(row['executing_rank'],row['expert_id'],row['expert_rank'],row['stage'])
            entry=per_sample.setdefault(key,dict(total=0,calls=0,first=row['first_tick'],last=row['last_tick']))
            entry['total']+=row['total_ticks']; entry['calls']+=row['calls']
            entry['first']=min(entry['first'],row['first_tick']);entry['last']=max(entry['last'],row['last_tick'])
        for key,value in per_sample.items(): groups.setdefault(key,[]).append(value)
    if len(presence)>1: raise ValueError('Mixed presence of expert diagnostics')
    summaries=[]
    factor=1e6/frequency_hz if frequency_hz is not None else 1
    for (rank,expert_id,owner,stage),values in sorted(groups.items()):
        summaries.append(dict(executing_rank=rank,expert_id=expert_id,expert_rank=owner,stage=stage,
            unit='us' if frequency_hz is not None else 'ticks',
            duration_kind='counter_only' if stage=='packets' else 'summed_core_work',
            **stats([v['total']*factor for v in values]),
            calls_total=sum(v['calls'] for v in values),mean_calls=statistics.mean(v['calls'] for v in values),
            mean_span=statistics.mean((v['last']-v['first'])*factor for v in values)))
        if stage=='packets':
            for key in ('mean','minimum','maximum','p50','p90','p99'): summaries[-1][key]=None
    return rows,summaries


def process_directory(source, output, frequency_hz, trace_iteration, lane_layout='core', show_repeated=False, operation=None):
    if operation not in (None,'dispatch','combine'): raise ValueError('Expected dispatch or combine')
    paths = sorted(source.rglob('sample_*.json'))
    if operation is not None:
        paths=[path for path in paths if json.loads(path.read_text())['operation']==operation]
        if not paths: raise ValueError(f'No {operation} diagnostic samples in {source}')
    report = analyze(paths, frequency_hz)
    selected = []
    for path in paths:
        item = json.loads(path.read_text())
        if item['metadata']['iteration'] == trace_iteration:
            selected.append(item)
    if not selected:
        raise ValueError(f'No diagnostic iteration {trace_iteration} in {source}')
    trace = timer.chrome_trace(selected, frequency_hz, lane_layout, show_repeated)
    expert_rows, expert_summary = expert_reports(paths, frequency_hz)
    report['per_expert']=expert_summary
    report['expert_interpretation']='Per executing rank and expert; core durations are summed work, not latency. Packet stage counts deliveries only. Shared FIRST_HOP preparation and batch waiting are not assigned. Only samples with executed records enter stage statistics.'
    output.mkdir(parents=True, exist_ok=False)
    (output/'summary.json').write_text(json.dumps(report, indent=2))
    wait_report=report['input_wait_bottleneck']
    (output/'input_wait_bottleneck.md').write_text(input_wait_markdown(wait_report),encoding='utf-8')
    if wait_report['per_core']:
        with (output/'input_wait_bottleneck.csv').open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(wait_report['per_core'][0]))
            writer.writeheader();writer.writerows(wait_report['per_core'])
    rows = report['per_rank_core_stage']
    with (output/'stages.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (output/'trace.json').write_text(json.dumps(trace))
    segment_metrics, segment_intervals, segment_cores = [], [], []
    for path in paths:
        saved=json.loads(path.read_text());meta=saved['metadata']
        decoded=timer.parse_records(saved['raw'],saved['operation'],frequency_hz)
        if decoded['schema'] not in (7,timer.SCHEMA): continue
        shared=dict(operation=saved['operation'],rank=meta['rank'],iteration=meta['iteration'])
        segment_metrics.extend(dict(**shared,**row) for row in decoded['records'])
        segment_intervals.extend(dict(**shared,**row) for row in decoded['intervals'])
        segment_cores.extend(dict(**shared,**row) for row in decoded['cores'])
    for filename,values in (('metrics.csv',segment_metrics),('intervals.csv',segment_intervals),('cores.csv',segment_cores)):
        if values:
            with (output/filename).open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    for filename,values in (('experts.csv',expert_summary),('expert_cores.csv',expert_rows)):
        if values:
            with (output/filename).open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(values[0]))
                writer.writeheader();writer.writerows(values)
    print(f'{source} -> {output} (all samples summarized; trace iteration {trace_iteration})')
    print(f"Dispatch input wait: {wait_report['status']} (heuristic; see input_wait_bottleneck.md)")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path,help='Timer directory, case, run, or run parent; every rank required per case')
    parser.add_argument('--op','--operation',dest='operation',choices=('dispatch','combine'),help='Filter all reports and trace to one operation')
    parser.add_argument('--latest',action='store_true',help='Select newest child run by cases.json creation mtime')
    parser.add_argument('--frequency-hz',type=float,default=timer.DEFAULT_FREQUENCY_HZ,
                        help='Default 1GHz: ticks/1000 us')
    parser.add_argument('--trace-iteration',type=int,default=0,help='Timeline iteration; statistics always include all iterations')
    parser.add_argument('--lane-layout',choices=('core','stage'),default='core',help='One track per AIV (default), or separate stage tracks')
    parser.add_argument('--show-repeated',action='store_true',help='Legacy envelope visibility only; segment samples are always shown')
    parser.add_argument('--output-dir',type=Path,help='Default: stage-analysis under selected source; existing reports rejected')
    args=parser.parse_args()
    source=args.directory if args.directory.is_absolute() else ROOT/args.directory
    source, directories = discover(source, args.latest)
    output = (args.output_dir if args.output_dir.is_absolute() else ROOT/args.output_dir) if args.output_dir else source/('stage-analysis-'+args.operation if args.operation else 'stage-analysis')
    for directory in directories:
        destination = output if len(directories)==1 else output/directory.relative_to(source)
        process_directory(directory, destination, args.frequency_hz, args.trace_iteration, args.lane_layout, args.show_repeated, args.operation)


if __name__=='__main__':main()
