"""Validate SystemCycle captures and export one compact report plus an offline workbook."""
import argparse
from collections import defaultdict
import csv
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[2]
MAX_BYTES = 60000
FREQUENCY_HZ = 1_000_000_000
MEASUREMENT = 'instrumented_full_device_kernel_task_us'
OPERATIONS = ('dispatch','combine')


def load(name, relative):
    spec=importlib.util.spec_from_file_location(name,ROOT/relative)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


profile=load('systemcycle_report_profile','tests/ll_systemcycle/profile.py')
timer=load('systemcycle_report_timer','deep_ep/utils/ll_stage_timer_systemcycle.py')
xlsx=load('systemcycle_report_xlsx','tests/ll_systemcycle/xlsx_export.py')


def expect(actual, expected, name):
    if actual != expected or (isinstance(actual,bool) != isinstance(expected,bool)):
        raise ValueError(f'{name} mismatch: {actual!r}; expected {expected!r}')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def positive_number(value):
    return type(value) in (int,float) and math.isfinite(value) and value>0


def validate_plan(plan):
    for key,value in dict(schema=1,revision=profile.DIAGNOSTIC_REVISION,
            baseline_reference=profile.BASELINE_REFERENCE,starting_baseline=profile.STARTING_BASELINE,
            send_cores=30,recv_cores=26,shape=profile.suite.SHAPE).items():
        expect(plan.get(key),value,'plan.'+key)
    if not re.fullmatch('[0-9a-f]{40}',plan['revision']):
        raise ValueError('Diagnostic source must be an immutable commit')
    if type(plan.get('timer_iters')) is not int or plan['timer_iters'] not in (1,3,5):
        raise ValueError('Expected 1, 3 or 5 complete diagnostic captures')
    seeds=profile.suite.selected_seeds(argparse.Namespace(seeds=plan.get('seeds',[])))
    if len(seeds)>3 or [run.get('seed') for run in plan.get('runs',[])]!=seeds:
        raise ValueError('Expected one to three distinct matching run seeds')
    for key,value in dict(shape=profile.suite.SHAPE,world_size=16,send_cores=30,recv_cores=26,
            logical_group=4,timer_schema=18,timer_profile='systemcycle',timer_profile_id=9,
            timer_mode='observe',recording='systemcycle-stages',timer_iters=plan['timer_iters'],
            frequency_hz=FREQUENCY_HZ,dispatch_only=False,high_throughput=False,
            use_1g_huge_pages=False,warmup=10,num_iters=100,perf_tail=50,
            correctness_repeats=3,seeds=seeds).items():
        expect(plan['configuration'].get(key),value,'configuration.'+key)
    dependency=plan['dependency']
    for field,width in (('dependency_id',64),('shmem_revision',40)):
        if not re.fullmatch('[0-9a-f]{'+str(width)+'}',str(dependency.get(field,''))):
            raise ValueError('Missing planned dependency identity: '+field)
    for variant in ('off','on'):
        build=plan['build_metadata'][variant]
        for key,value in dict(revision=plan['revision'],timer='OFF' if variant=='off' else 'ON',
                timer_schema=18,timer_profile='systemcycle',timer_profile_id=9,
                timer_bytes_per_core=1024,send_cores=30,recv_cores=26,logical_group=4,
                high_throughput=False,build_type='Release',**dependency).items():
            expect(build.get(key),value,variant+'.'+key)
        for key in ('native_sha256','kernel_sha256','sdk_library_sha256','sdk_manifest_sha256',
                    'wheel_sha256','source_sha256'):
            if not re.fullmatch('[0-9a-f]{64}',str(build.get(key,''))):
                raise ValueError('Missing build identity: '+key)
    profile.validate_matched_builds(plan['build_metadata'])
    for run in plan['runs']:
        expect(run.get('condition_order'),['off','profile'],'condition order')
        expect(run.get('measured_route_seed'),run['seed']+2,'measured route seed')


def validate_runtime_layout(identity,enabled):
    expected=dict(implementation='g4-r0-prefix',core_protocol=19030,overlap_stage=3,
        source_dirty='False',stage_timer_slots=16,stage_timer_fields=8,
        stage_timer_recording='systemcycle-stages',stage_timer_points=list(timer.POINTS),
        stage_timer_cycle_metric_names=list(timer.METRIC_NAMES),
        stage_timer_cycle_metric_indices=list(timer.METRIC_INDICES),
        stage_timer_counter_indices=list(timer.COUNTER_INDICES),
        stage_timer_metric_word_offset=48,stage_timer_metric_record_words=2,
        profile_capture=enabled,correctness_repeats=3,timer_mode='observe' if enabled else None,
        dispatch_only=False)
    for key,value in expected.items():expect(identity.get(key),value,'runtime.'+key)


def validate_summary(directory, seed, plan, variant):
    saved=read(directory/'summary.json'); enabled=variant=='profile'
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
            source_sha256=plan['build_metadata']['on' if enabled else 'off']['source_sha256'],
            stage_timer=enabled,stage_timer_schema=18,timer_profile='systemcycle',timer_profile_id=9,
            send_cores=30,recv_cores=26,active_aiv=56,logical_group=4,transport='mte',
            runtime_session='ll-mte',high_throughput_enabled=False,seed=seed,world_size=16).items():
        expect(identity.get(key),value,'runtime.'+key)
    validate_runtime_layout(identity,enabled)
    if not enabled:
        for operation in ('dispatch','combine'):
            perf=result[operation]; ranks=perf['ranks']
            if len(ranks)!=16 or {r['rank'] for r in ranks}!=set(range(16)):
                raise ValueError('Incomplete OFF profiler rank set')
            tail=[]
            for row in ranks:
                values=row['kernel_us']
                if row['samples']!=50 or len(values)!=100 or any(not positive_number(v) for v in values):
                    raise ValueError('Invalid OFF profiler samples')
                tail.extend(values[-50:])
                if not math.isclose(statistics.mean(values[-50:]),row['mean_us'],rel_tol=1e-9,abs_tol=1e-9):
                    raise ValueError('OFF rank mean does not match its raw tail')
            if not math.isclose(statistics.mean(tail),perf['mean_kernel_us'],rel_tol=1e-9,abs_tol=1e-9):
                raise ValueError('OFF pooled mean does not match raw tails')
    return result,config


def validate_active(directory, operation, result, iterations):
    path=directory/f'case_00000/normal/active_capture_{operation}.json'
    active=read(path)
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
            if len(values)!=iterations or any(not positive_number(v) for v in values):
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


def core_rows(meta,decoded,raw):
    points={core:{} for core in range(56)}
    records={core:{} for core in range(56)}
    for point in decoded['milestones']:
        points[point['core']][point['point']]=point['tick']
    for record in decoded['records']:
        selected=records[record['core']]
        if record['stage'] in selected:raise ValueError('Duplicate decoded stage')
        selected[record['stage']]=record
    result=[]
    for core in range(56):
        selected=records[core]
        if selected['ENTRY_TO_WORK_DONE']['total_ticks']<=0 or selected['KERNEL']['total_ticks']<=0:
            raise ValueError('Nonpositive SystemCycle work interval')
        words=[int(word)&((1<<64)-1) for row in raw[core] for word in row]
        result.append(dict(seed=meta['seed'],measured_route_seed=meta['seed']+2,
            rank=meta['rank'],iteration=meta['iteration'],core=core,operation=decoded['operation'],
            points=points[core],records=selected,raw_words=words,core_metadata=decoded['cores'][core]))
    return result


def read_run(off_dir,profile_dir,seed,plan):
    if any(off_dir.rglob('sample_*.json')):
        raise ValueError('OFF directory contains unexpected diagnostic samples')
    for variant,directory in (('off',off_dir),('on',profile_dir)):
        profile.validate_loaded_ranks(directory,plan['build_metadata'][variant])
    off,off_config=validate_summary(off_dir,seed,plan,'off')
    result,config=validate_summary(profile_dir,seed,plan,'profile')
    for key,value in dict(delay_rows=118260,delay_cols=30720).items():
        expect(off_config.get(key),value,'OFF.'+key);expect(config.get(key),value,'ON.'+key)
    active={op:validate_active(profile_dir,op,result,plan['timer_iters']) for op in OPERATIONS}
    sample_root=profile_dir/'case_00000/normal/ll_stage_timer'
    expected={sample_root/op/f'rank_{rank:03d}'/f'sample_{iteration:04d}.json'
              for op in OPERATIONS for rank in range(16) for iteration in range(plan['timer_iters'])}
    if set(profile_dir.rglob('sample_*.json'))!=expected:
        raise ValueError('Missing, extra or misplaced SystemCycle rank/capture samples')
    rows=[]
    for path in sorted(expected):
        saved=read(path);meta=saved['metadata'];operation=path.parents[1].name
        rank=int(path.parent.name[5:]);iteration=int(path.stem[7:])
        for key,value in dict(rank=rank,world_size=16,iteration=iteration,run_id=profile_dir.name,seed=seed,
                case_id=0,correctness='passed',config={**profile.suite.SHAPE,'logical_group_size':4},
                comparison=result['comparison']).items():
            expect(meta.get(key),value,'sample.'+key)
        for key,value in dict(after_perf=False,warmup=10,settle_iters=100,diagnostic_iters=plan['timer_iters'],
                delay_rows=config['delay_rows'],delay_cols=config['delay_cols'],group_backend='hccl',expert_timer=False).items():
            expect(meta['sampling'].get(key),value,'sampling.'+key)
        task=meta['active_capture_profile'];measured=active[operation]['ranks'][rank]
        for key,value in dict(status=measured['status'],all_ranks_status=active[operation]['status'],
                measurement=MEASUREMENT,kernel_us=measured['kernel_us'][iteration] if measured['status']=='available' else None,
                sample_index=iteration,expected_samples=plan['timer_iters'],
                sample_order='unique_finite_profiler_start_time').items():
            expect(task.get(key),value,'sample active task.'+key)
        expect(saved['operation'],operation,'operation');expect(saved['frequency_hz'],FREQUENCY_HZ,'frequency')
        decoded=timer.parse_records(saved['raw'],operation,saved['frequency_hz'])
        for key,value in dict(schema=18,active_aiv=56,send_cores=30,recv_cores=26,boundary_mode='observe',
                timer_profile='systemcycle',timer_profile_id=9,recording='systemcycle-stages',points=list(timer.POINTS)).items():
            expect(decoded.get(key),value,'decoded.'+key)
        rows.extend(core_rows(meta,decoded,saved['raw']))
    return rows,off,active


def read_all(directory):
    directory=Path(directory).resolve();plan=read(directory/'profile-plan.json');validate_plan(plan)
    runs=[];seen=set()
    for run in plan['runs']:
        paths=[]
        for variant in ('off','profile'):
            path=(directory/Path(run[variant+'_directory'])).resolve()
            if not path.is_relative_to(directory) or path==directory or path in seen:
                raise ValueError('Run paths must be distinct children of the profile directory')
            seen.add(path);paths.append(path)
        rows,off,active=read_run(*paths,run['seed'],plan)
        runs.append(dict(seed=run['seed'],rows=rows,off=off,active=active))
    return dict(plan=plan,runs=runs,directory=directory)


def fmt(value):
    if value is None:return 'NA'
    return f'{value:.6g}' if isinstance(value,float) else str(value)


def table(lines,title,fields,rows):
    stream=io.StringIO(newline='');writer=csv.writer(stream,lineterminator='\n')
    writer.writerow(fields);writer.writerows([[fmt(value) for value in row] for row in rows])
    lines.extend(['',title,stream.getvalue().rstrip('\n')])


def full_task_rows(run):
    result=[]
    for operation in OPERATIONS:
        off=run['off'][operation]['mean_kernel_us'];active=run['active'][operation]
        on=active['summary']['mean_kernel_us'] if active['status']=='available' else None
        result.append((operation,off,on,100*(on/off-1) if on is not None else None,active['status']))
    return result


def stage_statistics(rows,operation,initial_role):
    selected=[row for row in rows if row['operation']==operation and
              ('S' if row['core']<30 else 'R')==initial_role]
    selected_cores={(row['rank'],row['core']) for row in selected}
    result=[]
    for definition in timer.definitions(operation):
        stage=definition['stage'];counter=definition['kind']=='counter';values=defaultdict(list)
        for row in selected:
            record=row['records'].get(stage)
            if record is not None:
                value=record['calls'] if counter else record['total_ticks']/1000
                values[(row['rank'],row['core'])].append(value)
        core_means={key:statistics.mean(value) for key,value in values.items()}
        slow=max(core_means,key=core_means.get) if core_means else None
        result.append((stage,'count' if counter else 'sum_us' if definition['kind']=='metric' else 'span_us',
            f'{len(core_means)}/{len(selected_cores)}',sum(map(len,values.values())),
            statistics.median(core_means.values()) if core_means else None,
            statistics.mean(core_means.values()) if core_means else None,
            f'{slow[0]}:{slow[1]}' if slow is not None else None,
            core_means[slow] if slow is not None else None))
    return result


def compact_report(evidence):
    plan=evidence['plan'];lines=['SYSTEMCYCLE STAGE TIMING REPORT',
        f'Business baseline: {plan["baseline_reference"]}; restored base with SDK PI support: {plan["starting_baseline"]}',
        f'Diagnostic source: {plan["revision"]}; runner: {plan.get("runner_revision","unknown")}',
        f'SDK revision: {plan["dependency"]["shmem_revision"]}; schema18/profile9/systemcycle; observe only; 1GHz (1000 ticks/us).',
        'EP16 B=M256 H3584 K8 E384 prequantized FP8 dispatch + BF16 combine; G4 S30/R26; HT OFF; no added synchronization.',
        f'OFF: warmup10,100 profiler samples/rank,last50. ON: warmup10,settle100,{plan["timer_iters"]} actual attached captures.',
        'Every measured/settle/capture iteration uses Exp + scalar all_reduce; warmups follow the ST prelude.',
        'Complete files, correctness and 16-rank loaded library hashes validated.',
        'Full-task perturbation uses actual attached-capture profiler tasks, never the internal ENTRY->WORK_DONE interval.',
        'Sampling windows differ; ON/OFF is observed instrumentation perturbation, NOT an algorithm speedup estimate.',
        'Internal cycles exclude final diagnostic GM export. Parent envelopes and cumulative scopes overlap; NEVER add the rows.',
        'Issue/submission scopes do not prove transfer completion. No cross-rank clock subtraction or network bandwidth is inferred.',
        'Statistics below: average available captures for each (rank,core), then p50/mean/max of those core means; seeds remain separate.',
        'S/R groups use the initial split. Dispatch S cores can later receive; workbook roles additionally use actual RECV_BEGIN.',
        'Missing=NA, not zero. Valid zero is included. present=measured cores/selected cores; n=measured core-captures.',
        'Raw128-word data and full XLSX remain on target. Workbook defaults to last capture per seed; --captures all exports all captures.',
        'Omitted here: full per-core matrices, absolute timestamps and profiler trace/log files. Those remain in raw JSON/XLSX.']
    if plan.get('synthetic_fixture') is True:
        lines.insert(1,'SYNTHETIC CPU FIXTURE — formatting/validation example, NOT measured hardware results.')
    for variant in ('off','on'):
        lines.extend(['',variant.upper()+' build identity:'])
        lines.extend(key+'='+plan['build_metadata'][variant][key] for key in
            ('source_sha256','native_sha256','kernel_sha256','sdk_library_sha256','sdk_manifest_sha256'))
    for run in evidence['runs']:
        lines.extend(['',f'SEED requested={run["seed"]}, measured_route={run["seed"]+2}'])
        table(lines,'FULL DEVICE TASKS: OFF rank-tail pooled mean vs actual attached-capture pooled mean',
              ('operation','OFF_us','active_ON_us','perturbation_pct','status'),full_task_rows(run))
        for operation in OPERATIONS:
            unavailable=[f'rank{row["rank"]}:{row.get("reason","unavailable")}'
                         for row in run['active'][operation]['ranks'] if row['status']!='available']
            if unavailable:lines.append(operation+' profiler unavailable: '+'; '.join(unavailable))
            for role in ('S','R'):
                table(lines,operation+' initial-'+role+' per-core capture means; nested/overlapping rows',
                    ('stage','unit','present','n','p50','mean','slow_rank:core','slow_mean'),
                    stage_statistics(run['rows'],operation,role))
    data=('\n'.join(lines)+'\n').encode('utf-8')
    if len(data)>MAX_BYTES:
        raise ValueError(f'Report is {len(data)} bytes, above {MAX_BYTES}; no output written or evidence truncated')
    return data


METADATA_HEADERS=('实验组','算子','请求seed','Rank','Core','实际角色','采集轮次','实际路由seed','服务器','本地卡')


def role_label(row):
    if row['operation']=='combine':return 'S发送' if row['core']<30 else 'R接收归约'
    prefix='S远端发送' if row['core']<30 else 'R本组发送'
    return prefix+'→共享接收' if 'RECV_BEGIN' in row['points'] else prefix+'（无接收标记）'


def row_metadata(row):
    return ['systemcycle',row['operation'],row['seed'],row['rank'],row['core'],role_label(row),
            row['iteration']+1,row['measured_route_seed'],row['rank']//16,row['rank']%16]


def excel_integer(value):
    if value is not None and (type(value) is not int or abs(value)>999_999_999_999_999):
        raise ValueError('Relative ticks/counters exceed exact 15-digit Excel numeric precision')
    return value


def workbook_payload(evidence,captures='last'):
    if captures not in ('last','all'):raise ValueError('captures must be last or all')
    plan=evidence['plan'];definitions={op:timer.definitions(op) for op in OPERATIONS}
    all_definitions={item['stage']:item for op in OPERATIONS for item in definitions[op]}
    stages=list(all_definitions)
    derived_columns={stage:len(METADATA_HEADERS)+128+2*index+1 for index,stage in enumerate(stages)}
    raw_headers=[*METADATA_HEADERS,*[f'word{index:03d}_uint64_text' for index in range(128)]]
    for stage in stages:
        raw_headers.extend([stage+('_count' if all_definitions[stage]['kind']=='counter' else '_ticks'),stage+'_calls'])
    raw_rows=[];operations={op:[] for op in OPERATIONS}
    rows=sorted((row for run in evidence['runs'] for row in run['rows']
                 if captures=='all' or row['iteration']==plan['timer_iters']-1),
                key=lambda row:(row['seed'],OPERATIONS.index(row['operation']),row['iteration'],row['rank'],row['core']))
    for row in rows:
        metadata=row_metadata(row);raw_row=[*metadata,*map(str,row['raw_words'])]
        for stage in stages:
            record=row['records'].get(stage)
            raw_row.extend([excel_integer(record['calls'] if record['kind']=='counter' else record['total_ticks'])
                            if record else None,excel_integer(record['calls']) if record else None])
        raw_rows.append(raw_row);excel_row=len(raw_rows)+1;values=list(metadata)
        for definition in definitions[row['operation']]:
            record=row['records'].get(definition['stage'])
            if record is None:values.append(None);continue
            source=f"'原始Cycle'!{xlsx.column_name(derived_columns[definition['stage']])}{excel_row}"
            if definition['kind']=='counter':
                values.append(dict(formula='='+source,value=record['calls'],number_format='0'))
            else:
                values.append(dict(formula='='+source+"/'阶段说明'!$B$2",value=record['total_ticks']/1000,number_format='0.000'))
        operations[row['operation']].append(values)
    notes=[f'EP16 B=M256 H3584 K8 E384，G4 S30/R26；预量化FP8 dispatch + BF16 combine。schema18/profile9，observe，无新增同步。',
           f'业务基线 {plan["baseline_reference"]}；恢复基线/PI兼容 {plan["starting_baseline"]}。',
           f'打点源码 {plan["revision"]}；OFF/ON同源码同SDK；GetSystemCycle频率 {FREQUENCY_HZ} Hz。',
           '实际路由seed=请求seed+2，沿用本基线三轮正确性后的路由。',
           '空单元格=未记录/不适用；已记录的0保留。累计项与父包络有包含关系，不可相加。',
           '完整task性能来自OFF最后50条和ON实际attach捕获profiler，不用核内总耗时替代。',
           f'逐核及原始Cycle页选择：{captures}；全部原始capture JSON与日志仍保留目标机。',
           '绝对uint64原始word保存为十进制文本；差值/累计先由Python整数算出，再通过B2换算。']
    if plan.get('synthetic_fixture') is True:
        notes.insert(0,'合成CPU样例：仅用于格式和校验测试，不是设备实测数据。')
    explanation=[['ticks / us',FREQUENCY_HZ/1_000_000,'frequency_hz / 1e6','逐核耗时唯一换算引用；数量列不除该系数。'],
                 ['frequency_hz',FREQUENCY_HZ,'AscendC::GetSystemCycle','用户确认1GHz，原始ticks保留。'],
                 ['业务基线',plan['baseline_reference'],'Git commit','不将诊断开销当作基线算法收益。'],
                 ['打点源码',plan['revision'],'Git commit','OFF/ON匹配源码和SDK；实际库hash记录在紧凑报告。']]
    explanation.extend(['说明',str(index+1),'',note] for index,note in enumerate(notes))
    for operation in OPERATIONS:
        for definition in definitions[operation]:
            if definition['kind']=='span':
                start=definition['start_point'];end=definition['end_point']
                origin=f'{start} word{8+timer.POINTS.index(start)} → {end} word{8+timer.POINTS.index(end)}'
            else:
                offset=48+2*definition['metric_index']
                origin=f'sum_ticks word{offset}; calls/count word{offset+1}; legacy stage {definition["legacy_stage_id"]}'
            explanation.append([operation+': '+definition['stage'],definition['label']+' / '+definition['kind'],origin,definition['note']])
    performance=[]
    for run in evidence['runs']:
        for operation,off,on,change,status in full_task_rows(run):
            performance.append(['systemcycle',run['seed'],run['seed']+2,operation,off,on,change,16*50,
                                16*plan['timer_iters'] if on is not None else None,status])
    return dict(metadata=dict(baseline_revision=plan['baseline_reference'],source_revision=plan['revision'],
                              frequency_hz=FREQUENCY_HZ,notes=notes),
        performance=dict(headers=['实验组','请求seed','实际路由seed','算子','OFF均值(us)','ON捕获均值(us)',
                                   '扰动(%)','OFF样本数','ON样本数','状态'],rows=performance,freeze_column=4,
                         widths=[18,13,15,16,20,20,18,16,16,18],number_formats=[None]*4+['0.000']*3+['0','0',None]),
        operations={op:dict(headers=[*METADATA_HEADERS,*[item['label']+(' 数量' if item['kind']=='counter'
                           else ' 累计(us)' if item['kind']=='metric' else ' (us)') for item in definitions[op]]],
                    rows=operations[op],freeze_column=7,
                    widths=[14,13,11,9,9,28,12,15,10,10]+[23]*len(definitions[op])) for op in OPERATIONS},
        definitions=dict(headers=['字段 / 阶段','值 / 类型','来源 / 边界','含义与包含关系'],rows=explanation),
        raw=dict(headers=raw_headers,rows=raw_rows,freeze_column=7,
                 widths=[14,13,11,9,9,28,12,15,10,10]+[22]*(len(raw_headers)-len(METADATA_HEADERS))))


def render(directory):
    """Return validated compact text without modifying any input or output."""
    return compact_report(read_all(directory))


def export(directory,output,workbook,captures='last'):
    for path in (output,workbook):
        if os.path.lexists(path):raise FileExistsError('Refusing to overwrite existing output: '+str(path))
    output=Path(output).resolve();workbook=Path(workbook).resolve()
    if output==workbook:raise ValueError('Text and workbook output paths must differ')
    for path in (output,workbook):
        if os.path.lexists(path):raise FileExistsError('Refusing to overwrite existing output: '+str(path))
        if not path.parent.is_dir():raise ValueError('Output parent must already exist: '+str(path.parent))
    evidence=read_all(directory);data=compact_report(evidence);payload=workbook_payload(evidence,captures)
    # Validate both artifacts fully before writing either one. Exporter validation
    # remains authoritative and publishes the complete workbook without replacement.
    tables=[payload['performance'],*payload['operations'].values(),payload['definitions'],payload['raw']]
    for index,table_data in enumerate(tables):xlsx._normalize_table('validate'+str(index),table_data)
    try:
        xlsx.export_workbook(payload,workbook)
        with output.open('xb') as stream:stream.write(data)
    except (OSError,ValueError,TypeError) as error:
        raise RuntimeError('Artifact export failed; existing/partially produced outputs were preserved. '
                           'Choose new output names before retrying: '+str(error)) from error
    return dict(text=output,workbook=workbook,text_bytes=len(data))


def main():
    cli=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    cli.add_argument('directory',type=Path);cli.add_argument('--output',type=Path);cli.add_argument('--xlsx',type=Path)
    cli.add_argument('--captures',choices=('last','all'),default='last')
    args=cli.parse_args();directory=(ROOT/args.directory).resolve()
    output=(ROOT/args.output).resolve() if args.output else directory/'stage-report.txt'
    workbook=(ROOT/args.xlsx).resolve() if args.xlsx else directory/'stage-times.xlsx'
    try:
        if any(not path.is_relative_to(ROOT.resolve()) for path in (output,workbook)):
            raise ValueError('Report outputs must stay inside the project')
        result=export(directory,output,workbook,args.captures)
    except (OSError,ValueError,KeyError,TypeError,RuntimeError) as error:
        cli.exit(1,f'SystemCycle report rejected: {error}\n')
    print(f'Send only {result["text"]}; 1 file, {result["text_bytes"]}/{MAX_BYTES} bytes. '
          f'Workbook remains on target: {result["workbook"]}')


if __name__=='__main__':main()
