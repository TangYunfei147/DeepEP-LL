"""One bounded sparse baseline report with real OFF/active-capture task times."""
import argparse
from collections import Counter
import csv
import importlib.util
import io
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[2]
MAX_BYTES = 65535
MEASUREMENT = 'instrumented_full_device_kernel_task_us'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT/path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


profile = load('sparse_baseline_profile', 'tests/ll_control_opt/profile_baseline.py')
timer = load('sparse_baseline_timer', 'deep_ep/utils/ll_stage_timer.py')
PAIRS = (*timer.sparse.PAIRS,
    ('ENTRY_TO_WORK_DONE','ENTRY','WORK_DONE'),
    ('SEND_PREPARED_TO_FIRST_SEND','SEND_PREPARED','FIRST_SEND'),
    ('RECV_DONE_TO_WORK_DONE','RECV_DONE','WORK_DONE'),
    ('SEND_DONE_TO_WORK_DONE','SEND_DONE','WORK_DONE'))


def expect(actual, expected, name):
    if actual != expected:
        raise ValueError(f'{name} mismatch: {actual!r}; expected {expected!r}')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_plan(plan):
    for key,value in dict(schema=2,revision=profile.SPARSE_REVISION,
            baseline_reference=profile.BASELINE_REFERENCE,send_cores=30,recv_cores=26,
            shape=profile.suite.SHAPE).items():
        expect(plan.get(key),value,key)
    if plan.get('timer_iters') not in (3,5):
        raise ValueError('Expected 3 or 5 complete diagnostic iterations')
    seeds=profile.suite.selected_seeds(argparse.Namespace(seeds=plan.get('seeds',[])))
    if len(seeds)>3 or [r.get('seed') for r in plan.get('runs',[])]!=seeds:
        raise ValueError('Expected one to three distinct matching run seeds')
    for variant in ('off','sparse'):
        build=plan['build_metadata'][variant]
        for key,value in dict(revision=plan['revision'],timer='OFF' if variant=='off' else 'ON',
                timer_schema=15,timer_profile='sparse',timer_profile_id=6,send_cores=30,recv_cores=26,
                logical_group=4,high_throughput=False).items():
            expect(build.get(key),value,variant+'.'+key)
        for key in ('native_sha256','kernel_sha256','sdk_library_sha256','sdk_manifest_sha256',
                    'wheel_sha256','source_sha256'):
            if not re.fullmatch('[0-9a-f]{64}',build.get(key,'')):
                raise ValueError('Missing build identity: '+key)
        expect(build.get('shmem_revision'),plan['dependency']['shmem_revision'],'SDK revision')
    for field in ('sdk_library_sha256','sdk_manifest_sha256','shmem_revision'):
        expect(plan['build_metadata']['off'][field],plan['build_metadata']['sparse'][field],
               'OFF/SPARSE matching '+field)


def validate_summary(directory, seed, plan, variant):
    saved=read(directory/'summary.json'); enabled=variant=='sparse'
    for key,value in dict(status='completed',world_size=16,case_count=1,result_count=1).items():
        expect(saved.get(key),value,'summary.'+key)
    config=saved['config']
    for key,value in dict(seed=seed,warmup=10,num_iters=100,perf_tail=50,route='normal',
            ll_stage_timer=enabled,ll_timer_mode='observe',skip_check=False,skip_perf_test=enabled,
            dispatch_only=False,correctness_repeats=3,use_1g_huge_pages=0,
            ll_profile_capture=enabled).items():
        expect(config.get(key),value,'config.'+key)
    if enabled:
        expect(config.get('ll_timer_iters'),plan['timer_iters'],'capture count')
        expect(config.get('ll_timer_frequency_hz'),1_000_000_000,'frequency')
    if len(saved['results'])!=1:
        raise ValueError('Expected one fixed-shape result')
    result=saved['results'][0]
    for key,value in dict(config=profile.suite.SHAPE,correctness='passed',case_id=0,route='normal').items():
        expect(result.get(key),value,'case.'+key)
    identity=result['comparison']
    for key,value in dict(source_revision=plan['revision'],control_revision=plan['revision'],
            source_sha256=plan['build_metadata'][variant]['source_sha256'],
            stage_timer=enabled,stage_timer_schema=15,timer_profile='sparse',timer_profile_id=6,
            send_cores=30,recv_cores=26,active_aiv=56,logical_group=4,transport='mte',
            runtime_session='ll-mte',high_throughput_enabled=False,seed=seed,world_size=16).items():
        expect(identity.get(key),value,'runtime.'+key)
    if not enabled:
        for operation in ('dispatch','combine'):
            perf=result[operation]; ranks=perf['ranks']
            if len(ranks)!=16 or {r['rank'] for r in ranks}!=set(range(16)):
                raise ValueError('Incomplete OFF profiler rank set')
            tail=[]
            for row in ranks:
                values=row['kernel_us']
                if row['samples']!=50 or len(values)!=100 or any(not math.isfinite(v) or v<=0 for v in values):
                    raise ValueError('Invalid OFF profiler samples')
                tail.extend(values[-50:])
                if not math.isclose(statistics.mean(values[-50:]),row['mean_us'],rel_tol=1e-9,abs_tol=1e-9):
                    raise ValueError('OFF rank mean does not match its raw tail')
            if not math.isclose(statistics.mean(tail),perf['mean_kernel_us'],rel_tol=1e-9,abs_tol=1e-9):
                raise ValueError('OFF pooled mean does not match raw tails')
    return result,config


def validate_active(directory, operation, result, iterations):
    active=read(directory/f'case_00000/normal/active_capture_{operation}.json')
    for key,value in dict(operation=operation,route='normal',config=profile.suite.SHAPE,
            comparison=result['comparison'],correctness='passed',diagnostic_iters=iterations,
            measurement=MEASUREMENT).items():
        expect(active.get(key),value,'active.'+key)
    ranks=active['ranks']
    if len(ranks)!=16 or [r.get('rank') for r in ranks]!=list(range(16)):
        raise ValueError('Incomplete/duplicate active profiler ranks')
    for row in ranks:
        for key,value in dict(operation=operation,expected_samples=iterations,measurement=MEASUREMENT).items():
            expect(row.get(key),value,'active rank.'+key)
        if row.get('status') not in ('available','unavailable'):
            raise ValueError('Unknown active profiler status')
        if row['status']=='available':
            values=row.get('kernel_us',[])
            if len(values)!=iterations or any(not math.isfinite(v) or v<=0 for v in values):
                raise ValueError('Invalid active capture durations')
        elif row.get('kernel_us'):
            raise ValueError('Unavailable profiler result cannot supply durations')
    available=all(r['status']=='available' for r in ranks)
    expect(active['status'],'available' if available else 'unavailable','active aggregate status')
    if available:
        pooled=statistics.mean(v for row in ranks for v in row['kernel_us'])
        if not math.isclose(active['summary']['mean_kernel_us'],pooled,rel_tol=1e-9,abs_tol=1e-9):
            raise ValueError('Active profiler summary disagrees with actual capture tasks')
    return active


def core_rows(meta, decoded):
    points={core:{} for core in range(56)}
    for mark in decoded['milestones']:
        points[mark['core']][mark['point']]=mark['tick']
    rows=[]
    for core,marks in sorted(points.items()):
        spans={}
        for name,start,end in PAIRS:
            if start in marks and end in marks:
                if marks[end]<marks[start]:
                    raise ValueError('Invalid same-core milestone span: '+name)
                spans[name]=(marks[end]-marks[start])/1000
        if spans['ENTRY_TO_WORK_DONE']<=0 or spans['KERNEL']<=0:
            raise ValueError('Nonpositive sparse work span')
        rows.append(dict(rank=meta['rank'],iteration=meta['iteration'],core=core,
            operation=decoded['operation'],points=marks,spans=spans,work_us=spans['ENTRY_TO_WORK_DONE']))
    return rows


def read_run(off_dir, sparse_dir, seed, plan):
    for variant,directory in (('off',off_dir),('sparse',sparse_dir)):
        profile.validate_loaded_ranks(directory,plan['build_metadata'][variant])
    off,_=validate_summary(off_dir,seed,plan,'off')
    result,config=validate_summary(sparse_dir,seed,plan,'sparse')
    active={op:validate_active(sparse_dir,op,result,plan['timer_iters']) for op in ('dispatch','combine')}
    sample_root=sparse_dir/'case_00000/normal/ll_stage_timer'
    expected={sample_root/op/f'rank_{rank:03d}'/f'sample_{iteration:04d}.json'
        for op in ('dispatch','combine') for rank in range(16) for iteration in range(plan['timer_iters'])}
    if set(sparse_dir.rglob('sample_*.json'))!=expected:
        raise ValueError('Missing, extra or misplaced sparse rank/iteration samples')
    rows=[]
    for path in sorted(expected):
        saved=read(path);meta=saved['metadata'];operation=path.parents[1].name
        rank=int(path.parent.name[5:]);iteration=int(path.stem[7:])
        for key,value in dict(rank=rank,world_size=16,iteration=iteration,run_id=sparse_dir.name,seed=seed,
            case_id=0,correctness='passed',config={**profile.suite.SHAPE,'logical_group_size':4},
            comparison=result['comparison']).items():
            expect(meta.get(key),value,'sample.'+key)
        for key,value in dict(after_perf=False,warmup=10,settle_iters=100,diagnostic_iters=plan['timer_iters'],
            delay_rows=config['delay_rows'],delay_cols=config['delay_cols'],group_backend='hccl',expert_timer=False).items():
            expect(meta['sampling'].get(key),value,'sampling.'+key)
        task=meta['active_capture_profile']; measured=active[operation]['ranks'][rank]
        for key,value in dict(status=measured['status'],all_ranks_status=active[operation]['status'],
            measurement=MEASUREMENT,kernel_us=measured['kernel_us'][iteration] if measured['status']=='available' else None,
            sample_index=iteration,expected_samples=plan['timer_iters'],
            sample_order='unique_finite_profiler_start_time').items():
            expect(task.get(key),value,'sample active task.'+key)
        expect(saved['operation'],operation,'operation');expect(saved['frequency_hz'],1_000_000_000,'frequency')
        decoded=timer.parse_records(saved['raw'],operation,saved['frequency_hz'])
        for key,value in dict(schema=15,active_aiv=56,send_cores=30,boundary_mode='observe',
            timer_profile='sparse',timer_profile_id=6,recording='sparse-milestones').items():
            expect(decoded.get(key),value,'decoded.'+key)
        rows.extend(core_rows(meta,decoded))
    return rows,off,active


def median(values):
    values=[v for v in values if v is not None]
    return statistics.median(values) if values else None


def fmt(value):
    if value is None:return 'NA'
    return f'{value:.5g}' if isinstance(value,float) else str(value)


def table(lines,title,fields,rows):
    lines.extend(['',title]);stream=io.StringIO(newline='');writer=csv.writer(stream,lineterminator='\n')
    writer.writerow(fields);writer.writerows([[fmt(v) for v in row] for row in rows])
    lines.append(stream.getvalue().rstrip('\n'))


def winners(rows):
    groups={}
    for row in rows:groups.setdefault((row['rank'],row['iteration']),[]).append(row)
    return [max(group,key=lambda row:row['work_us']) for group in groups.values()]


def describe_seed(seed, rows, off, active):
    lines=['',f'SEED requested={seed}, measured_route={seed+2}'];overhead=[]
    for op in ('dispatch','combine'):
        base=off[op]['mean_kernel_us'];on=active[op]
        full=on['summary']['mean_kernel_us'] if on['status']=='available' else None
        overhead.append((op,base,full,100*(full/base-1) if full else None,on['status']))
        if full is None:
            bad=[str(r['rank'])+':'+str(r.get('reason','unavailable'))[:160] for r in on['ranks'] if r['status']!='available']
            lines.append('Active profiler unavailable for '+op+'; '+'; '.join(bad))
    table(lines,'FULL DEVICE TASKS: OFF rank-tail pooled mean vs actual sparse capture pooled mean',
        ('operation','OFF_us','active_sparse_us','active_vs_OFF_pct','status'),overhead)
    dispatch=[r for r in rows if r['operation']=='dispatch'];combine=[r for r in rows if r['operation']=='combine']
    dw,cw=winners(dispatch),winners(combine);ranks=[]
    for rank in range(16):
        dr=[r for r in dw if r['rank']==rank];cr=[r for r in cw if r['rank']==rank]
        off_rank=next(r for r in off['dispatch']['ranks'] if r['rank']==rank)['mean_us']
        on_rank=active['dispatch']['ranks'][rank]
        on_mean=statistics.mean(on_rank['kernel_us']) if on_rank['status']=='available' else None
        spread=[r['work_us']/statistics.median(x['work_us'] for x in dispatch
                  if (x['rank'],x['iteration'])==(rank,r['iteration'])) for r in dr]
        ranks.append((rank,off_rank,on_mean,100*(on_mean/off_rank-1) if on_mean else None,
            median(r['work_us'] for r in dr),Counter(r['core'] for r in dr).most_common(1)[0][0],median(spread),
            median(r['work_us'] for r in cr)))
    table(lines,'RANKS: max same-core ENTRY->WORK_DONE is a proxy, not a synchronized whole-rank span',
        ('rank','D_OFF_us','D_active_us','D_inflation_pct','D_max_work_p50_us','D_core_mode','D_max/median_p50','C_max_work_p50_us'),ranks)
    groups=[('D-longest-per-rank/sample',dw),('D-S0..29',[r for r in dispatch if r['core']<30]),
            ('D-R0',[r for r in dispatch if r['core']==30]),('D-R1..25',[r for r in dispatch if r['core']>30])]
    for name,group in groups:
        values=[]
        for span,_,_ in PAIRS:
            present=[r for r in group if span in r['spans']]
            if present:
                values.append((span,f'{len(present)}/{len(group)}',median(r['spans'][span] for r in present),
                    median(100*r['spans'][span]/r['work_us'] for r in present)))
        table(lines,name+' same-core spans; rows overlap/nest and MUST NOT be summed',
            ('span','present/selected','us_p50','same_core_work_pct_p50'),values)
    table(lines,'Actual milestone occurrence counts (core-samples; NOT packets)',
        ('scope','selected',*timer.SPARSE_POINTS),
        [(name,len(group),*(sum(point in r['points'] for r in group) for point in timer.SPARSE_POINTS)) for name,group in groups[1:]])
    table(lines,'COMBINE overview only: ENTRY/INIT_DONE/WORK_DONE; no send/recv fine stages collected',
        ('span','selected','us_p50'),[(name,len(cw),median(r['spans'][name] for r in cw))
                                    for name in ('INIT','KERNEL','ENTRY_TO_WORK_DONE')])
    for rank in sorted(range(16),key=lambda n:median(r['work_us'] for r in dw if r['rank']==n),reverse=True)[:2]:
        selected=sorted((r for r in dw if r['rank']==rank),key=lambda r:r['iteration'])
        table(lines,f'RAW TICKS: rank{rank}, actual longest-work core selected independently each iteration',
            ('iteration','core',*timer.SPARSE_POINTS),
            [(r['iteration'],r['core'],*(r['points'].get(p) for p in timer.SPARSE_POINTS)) for r in selected])
    return lines


def render(directory):
    directory=Path(directory).resolve();plan=read(directory/'profile-plan.json');validate_plan(plan)
    lines=['BASELINE SPARSE BOTTLENECK REPORT',
        f'Actual diagnostic source: {plan["revision"]}; business reference: {plan["baseline_reference"]}',
        f'Runner: {plan.get("runner_revision","unknown")}; SDK: {plan["dependency"]["shmem_revision"]}',
        'EP16 B=M256 H3584 K8 E384 FP8 dispatch/BF16 combine; G4 S30/R26; HT OFF; schema15/profile6; 1GHz.',
        f'OFF=10 warmup/100 samples/rank tail50. SPARSE=10 warmup/100 settle/{plan["timer_iters"]} active captures.',
        'Both use Exp + scalar all_reduce. All16 ranks, captures, correctness and actual loaded hashes validated.',
        'Compare actual active-capture full kernel tasks with OFF before interpreting spans; perturbation is NOT assumed zero.',
        'Sampling windows/counts differ: active/OFF is a perturbation observation, NOT an algorithm speedup estimate.',
        'Internal ENTRY/INIT_DONE/WORK_DONE excludes final sparse export; never substitute internal spans for full task time.',
        'Only14 first-occurrence timestamps/core,512B allocation/core,192B export/core. Combine records only3 points.',
        'NO packet counts, poll counts, empty/ready ratios, cumulative leaf costs, or final-relay timestamps were collected.',
        'Missing=NA, not zero. Spans are same-core/sample; never add overlapping/nested spans or subtract cross-rank clocks.',
        'FIRST_RECV_READY is first observed ready task, not physical arrival. SEND_DONE includes original retirement and can follow receive.',
        'PREFIX_LOCAL_DONE is the existing V_MTE3 submission boundary, not Scalar-confirmed prefix VF completion.',
        'No per-port utilization, queue saturation or proven hardware cause is inferred. Raw data stay on the target.']
    for variant in ('off','sparse'):
        lines+=['',variant.upper()+' build identity:']
        lines.extend(f'{key}={plan["build_metadata"][variant][key]}' for key in
            ('source_sha256','native_sha256','kernel_sha256','sdk_library_sha256','sdk_manifest_sha256'))
    seen=set()
    for run in plan['runs']:
        expect(run.get('measured_route_seed'),run['seed']+2,'measured route seed');paths=[]
        for variant in ('off','sparse'):
            source=(directory/Path(run[variant+'_directory'])).resolve()
            if not source.is_relative_to(directory) or source==directory or source in seen:
                raise ValueError('Run paths must be distinct children of the requested profile directory')
            seen.add(source);paths.append(source)
        rows,off,active=read_run(*paths,run['seed'],plan)
        lines.extend(describe_seed(run['seed'],rows,off,active))
    data=('\n'.join(lines)+'\n').encode('utf-8')
    if len(data)>MAX_BYTES:
        raise ValueError(f'Report is {len(data)} bytes, above {MAX_BYTES}; no output written or evidence truncated')
    return data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args();directory=(ROOT/args.directory).resolve()
    output=(ROOT/args.output).resolve() if args.output else directory/'baseline-report.txt'
    try:
        data=render(directory)
        with output.open('xb') as stream:stream.write(data)
    except (OSError,ValueError,KeyError,TypeError) as error:
        parser.exit(1,f'Sparse report rejected: {error}\n')
    print(f'Send only {output};1/10 files,{len(data)}/{MAX_BYTES} bytes')


if __name__=='__main__':main()
