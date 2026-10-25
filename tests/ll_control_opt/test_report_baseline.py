"""CPU disk fixtures for OFF tail50 and schema15 actual attached-capture reports."""
import copy
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import statistics

import pytest

ROOT=Path(__file__).resolve().parents[2]


def load(name,relative):
    spec=importlib.util.spec_from_file_location(name,ROOT/relative)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


report=load('baseline_report_tests','tests/ll_control_opt/report_baseline.py')
fixtures=load('baseline_sparse_fixtures','tests/ll_stage_timer/test_sparse_timer.py')
timer=report.timer
REVISION='c'*40
SHMEM='8e3fc610f91ae9e5e8cf13748cfb74826b272c2a'


@pytest.fixture(autouse=True)
def committed_sparse_identity(monkeypatch):
    # The runner's release SHA is filled in only after its implementation commit.
    monkeypatch.setattr(report.profile,'SPARSE_REVISION',REVISION)


def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,separators=(',',':')),encoding='utf-8')


def mutate(path,change):
    value=report.read(path);change(value);write(path,value)


def run_path(directory,seed=0,variant='sparse'):
    return Path(directory)/f'seed-{seed}-{variant}'


def sample_path(directory,seed=0,operation='dispatch',rank=1,iteration=0):
    return run_path(directory,seed)/f'case_00000/normal/ll_stage_timer/{operation}/rank_{rank:03d}/sample_{iteration:04d}.json'


def active_path(directory,seed=0,operation='dispatch'):
    return run_path(directory,seed)/f'case_00000/normal/active_capture_{operation}.json'


def raw(operation,rank=0,iteration=0):
    """Different core/rank epochs; exactly two measured send spans, never zeros for missing ones."""
    result=fixtures.raw(send=30,operation=operation,minimal=True)
    for core in range(56):
        epoch=1_000_000_000*(rank+1)+core*10_000_000+iteration*1_000_000
        work=1_000_000 if core==1 else 100_000
        points=dict(ENTRY=epoch,INIT_DONE=epoch+1000,WORK_DONE=epoch+work)
        if operation=='dispatch' and core in (0,1):
            points.update(SEND_PREPARED=epoch+11_000,
                          FIRST_SEND=epoch+1000+(50_000 if core==0 else 100_000),
                          SEND_DONE=epoch+work-5000)
        if operation=='dispatch' and core==30:
            points.update(COUNTS_RECEIVED=epoch+2000,PREFIX_LOCAL_DONE=epoch+3000,
                          PREFIX_READY=epoch+4000,RECV_PREPARED=epoch+5000,
                          FIRST_RECV_PROBE=epoch+6000,FIRST_RECV_READY=epoch+7000,
                          RECV_DONE=epoch+90_000)
        words=[word for row in result[core] for word in row]
        words[6]=0;words[8:]=[0]*56
        for name,tick in points.items():
            index=timer.SPARSE_POINTS.index(name);words[6]|=1<<index;words[8+index]=tick
        result[core]=[words[start:start+8] for start in range(0,64,8)]
    timer.parse_records(result,operation,1_000_000_000)
    return result


def make_profile(directory,seeds=(0,),iterations=3):
    directory=Path(directory)
    shared={key:hashlib.sha256(key.encode()).hexdigest() for key in
            ('sdk_library_sha256','sdk_manifest_sha256','source_sha256')}
    builds={}
    for variant in ('off','sparse'):
        hashes={key:hashlib.sha256((variant+key).encode()).hexdigest()
                for key in ('native_sha256','kernel_sha256','wheel_sha256')}
        builds[variant]=dict(revision=REVISION,timer='OFF' if variant=='off' else 'ON',
            timer_schema=15,timer_profile='sparse',timer_profile_id=6,send_cores=30,recv_cores=26,
            logical_group=4,high_throughput=False,shmem_revision=SHMEM,**shared,**hashes)
    plan=dict(schema=2,revision=REVISION,baseline_reference=report.profile.BASELINE_REFERENCE,
              runner_revision='fixture-runner',send_cores=30,recv_cores=26,
              shape=copy.deepcopy(report.profile.suite.SHAPE),timer_iters=iterations,seeds=list(seeds),
              build_metadata=builds,dependency=dict(shmem_revision=SHMEM),runs=[])
    for seed in seeds:
        plan['runs'].append(dict(seed=seed,measured_route_seed=seed+2,
            off_directory=run_path(directory,seed,'off').name,
            sparse_directory=run_path(directory,seed).name))
        for variant in ('off','sparse'):
            run=run_path(directory,seed,variant);enabled=variant=='sparse';build=builds[variant]
            identity=dict(source_revision=REVISION,control_revision=REVISION,source_sha256=build['source_sha256'],
                stage_timer=enabled,stage_timer_schema=15,timer_profile='sparse',timer_profile_id=6,
                stage_timer_slots=8,stage_timer_fields=8,stage_timer_recording='sparse-milestones',
                stage_timer_points=list(timer.SPARSE_POINTS),send_cores=30,recv_cores=26,active_aiv=56,
                logical_group=4,transport='mte',runtime_session='ll-mte',high_throughput_enabled=False,
                seed=seed,world_size=16,profile_capture=enabled)
            config=dict(seed=seed,warmup=10,num_iters=100,perf_tail=50,route='normal',
                ll_stage_timer=enabled,ll_timer_mode='observe',ll_timer_iters=iterations,
                ll_timer_frequency_hz=1_000_000_000,ll_profile_capture=enabled,
                skip_check=False,skip_perf_test=enabled,dispatch_only=False,correctness_repeats=3,
                use_1g_huge_pages=0,delay_rows=118260,delay_cols=30720)
            case=dict(config=plan['shape'],correctness='passed',case_id=0,route='normal',comparison=identity)
            active={}
            for operation,off_base,on_base in (('dispatch',100.,200.),('combine',300.,500.)):
                if not enabled:
                    ranks=[]
                    for rank in range(16):
                        values=[off_base+9000]*50+[off_base+rank]*50
                        ranks.append(dict(rank=rank,samples=50,kernel_us=values,mean_us=statistics.mean(values[-50:])))
                    case[operation]=dict(ranks=ranks,mean_kernel_us=statistics.mean(r['mean_us'] for r in ranks))
                else:
                    ranks=[dict(rank=rank,operation=operation,expected_samples=iterations,
                        measurement=report.MEASUREMENT,status='available',
                        kernel_us=[on_base+rank+i/10 for i in range(iterations)]) for rank in range(16)]
                    active[operation]=dict(operation=operation,route='normal',config=plan['shape'],comparison=identity,
                        correctness='passed',diagnostic_iters=iterations,measurement=report.MEASUREMENT,
                        status='available',ranks=ranks,summary=dict(mean_kernel_us=statistics.mean(
                            value for row in ranks for value in row['kernel_us'])))
                    write(active_path(directory,seed,operation),active[operation])
            write(run/'summary.json',dict(status='completed',world_size=16,case_count=1,result_count=1,
                                          config=config,results=[case]))
            for rank in range(16):
                write(run/f'loaded_rank_{rank:03d}.json',dict(rank=rank,libraries={
                    '/fixture/deep_ep/_C.test.so':build['native_sha256'],
                    '/fixture/deep_ep/libdeepep_kernels.so':build['kernel_sha256'],
                    '/fixture/shmem/lib/libshmem.so':build['sdk_library_sha256']}))
                if not enabled:continue
                for operation in ('dispatch','combine'):
                    for iteration in range(iterations):
                        task=active[operation]['ranks'][rank]
                        meta=dict(rank=rank,world_size=16,iteration=iteration,run_id=run.name,seed=seed,
                            case_id=0,correctness='passed',comparison=identity,
                            config={**plan['shape'],'logical_group_size':4},
                            sampling=dict(after_perf=False,warmup=10,settle_iters=100,diagnostic_iters=iterations,
                                delay_rows=config['delay_rows'],delay_cols=config['delay_cols'],
                                group_backend='hccl',expert_timer=False),
                            active_capture_profile=dict(status='available',all_ranks_status='available',
                                measurement=report.MEASUREMENT,kernel_us=task['kernel_us'][iteration],
                                sample_index=iteration,expected_samples=iterations,
                                sample_order='unique_finite_profiler_start_time',reason=None))
                        write(sample_path(directory,seed,operation,rank,iteration),dict(operation=operation,
                              frequency_hz=1_000_000_000,metadata=meta,raw=raw(operation,rank,iteration)))
    write(directory/'profile-plan.json',plan)
    return plan


def tree_hashes(directory):
    return {path.relative_to(directory).as_posix():hashlib.sha256(path.read_bytes()).hexdigest()
            for path in Path(directory).rglob('*') if path.is_file()}


def section(text,title):
    chunk=text.split(title+'\n',1)[1].split('\n\n',1)[0]
    return list(csv.DictReader(io.StringIO(chunk)))


@pytest.mark.parametrize('seeds,iterations',[((0,),3),((0,17,101),5)])
def test_complete_report_validates_all_six_runs_and_preserves_raw(tmp_path,monkeypatch,seeds,iterations):
    directory=tmp_path/'profile';make_profile(directory,seeds,iterations)
    before=tree_hashes(directory);checked=[];original=report.profile.validate_loaded_ranks
    def validate(directory,manifest):
        checked.append((directory.name,manifest['timer']));return original(directory,manifest)
    monkeypatch.setattr(report.profile,'validate_loaded_ranks',validate)
    data=report.render(directory);text=data.decode('utf-8')
    assert isinstance(data,bytes) and 0<len(data)<=65535
    assert len(checked)==len(seeds)*2 and {timer for _,timer in checked}=={'OFF','ON'}
    assert text.count('SEED requested=')==len(seeds)
    assert len(list(directory.rglob('sample_*.json')))==len(seeds)*2*16*iterations
    assert len(list(directory.rglob('loaded_rank_*.json')))==len(seeds)*2*16
    assert 'NO packet counts' in text and 'not zero' in text and 'NOT an algorithm speedup' in text
    assert '512B allocation/core,192B export/core' in text
    assert tree_hashes(directory)==before


def test_only_off_tail_and_actual_capture_profiler_determine_full_task_times(tmp_path):
    directory=tmp_path/'profile';make_profile(directory)
    # An accidental non-capture ON performance field must never become the numerator.
    mutate(run_path(directory)/'summary.json',lambda data:data['results'][0].update(
        dispatch={'mean_kernel_us':999999},combine={'mean_kernel_us':888888}))
    text=report.render(directory).decode('utf-8')
    rows=section(text,'FULL DEVICE TASKS: OFF rank-tail pooled mean vs actual sparse capture pooled mean')
    assert rows[0]['operation']=='dispatch'
    assert float(rows[0]['OFF_us'])==107.5
    assert float(rows[0]['active_sparse_us'])==207.6
    assert float(rows[0]['active_vs_OFF_pct'])==pytest.approx(100*(207.6/107.5-1),rel=1e-5)
    assert float(rows[1]['OFF_us'])==307.5 and float(rows[1]['active_sparse_us'])==507.6
    assert '999999' not in text and '888888' not in text


def test_same_core_ratios_and_missing_points_are_not_replaced_with_zero(tmp_path):
    directory=tmp_path/'profile';plan=make_profile(directory)
    rows,off,active=report.read_run(run_path(directory,variant='off'),run_path(directory),0,plan)
    present=[r for r in rows if r['operation']=='dispatch' and 'BEFORE_FIRST_SEND' in r['spans']]
    assert {r['core'] for r in present}=={0,1}
    assert statistics.median(r['spans']['BEFORE_FIRST_SEND'] for r in present)==75
    assert statistics.median(100*r['spans']['BEFORE_FIRST_SEND']/r['work_us'] for r in present)==30
    assert all('BEFORE_FIRST_SEND' not in r['spans'] for r in rows if r['core']==2)
    text='\n'.join(report.describe_seed(0,rows,off,active))
    sends=section(text,'D-S0..29 same-core spans; rows overlap/nest and MUST NOT be summed')
    first=next(row for row in sends if row['span']=='BEFORE_FIRST_SEND')
    assert first==dict(span='BEFORE_FIRST_SEND',**{'present/selected':'96/1440'},us_p50='75',same_core_work_pct_p50='30')
    ticks=section(text,'RAW TICKS: rank0, actual longest-work core selected independently each iteration')
    assert all(row['core']=='1' and row['FIRST_RECV_PROBE']=='NA' for row in ticks)
    assert all(set(r['points'])=={'ENTRY','INIT_DONE','WORK_DONE'} for r in rows if r['operation']=='combine')


def unavailable(directory,rank=15,operation='dispatch',iterations=3):
    def change(data):
        data['status']='unavailable';data.pop('summary',None)
        row=data['ranks'][rank];row.update(status='unavailable',reason='missing profiler task')
        row.pop('kernel_us',None)
    mutate(active_path(directory,operation=operation),change)
    for selected_rank in range(16):
        for iteration in range(iterations):
            def sample_change(data,selected_rank=selected_rank):
                task=data['metadata']['active_capture_profile'];task['all_ranks_status']='unavailable'
                if selected_rank==rank:task.update(status='unavailable',kernel_us=None,reason='missing profiler task')
            mutate(sample_path(directory,operation=operation,rank=selected_rank,iteration=iteration),sample_change)


def test_unavailable_capture_has_no_fabricated_overhead_or_rank_ratio(tmp_path):
    directory=tmp_path/'profile';make_profile(directory);unavailable(directory)
    text=report.render(directory).decode('utf-8')
    tasks=section(text,'FULL DEVICE TASKS: OFF rank-tail pooled mean vs actual sparse capture pooled mean')
    assert tasks[0]==dict(operation='dispatch',OFF_us='107.5',active_sparse_us='NA',active_vs_OFF_pct='NA',status='unavailable')
    assert tasks[1]['status']=='available'
    ranks=section(text,'RANKS: max same-core ENTRY->WORK_DONE is a proxy, not a synchronized whole-rank span')
    assert ranks[15]['D_active_us']==ranks[15]['D_inflation_pct']=='NA'
    assert '15:missing profiler task' in text


@pytest.mark.parametrize('variant',('off','sparse'))
@pytest.mark.parametrize('problem',('missing_loaded_rank','native_hash','kernel_hash','sdk_hash'))
def test_every_variant_checks_all_actual_loaded_libraries(tmp_path,variant,problem):
    directory=tmp_path/'profile';make_profile(directory)
    path=run_path(directory,variant=variant)/'loaded_rank_015.json'
    if problem=='missing_loaded_rank':path.unlink()
    else:
        name={'native_hash':'/fixture/deep_ep/_C.test.so','kernel_hash':'/fixture/deep_ep/libdeepep_kernels.so',
              'sdk_hash':'/fixture/shmem/lib/libshmem.so'}[problem]
        mutate(path,lambda data:data['libraries'].update({name:'f'*64}))
    with pytest.raises(ValueError,match='loaded|Loaded'):report.render(directory)


@pytest.mark.parametrize('problem',[
    'missing_rank_sample','extra_sample','wrong_iteration','shape','correctness','frequency','schema',
    'source_sha','wrong_seed','off_raw99','off_rank_mean','off_pooled_mean','active_rank_missing',
    'active_task_count','active_task_value','active_summary_mean','sample_task_duration',
    'sample_task_index','sample_task_count','sample_task_order','plan_source_sha','plan_sdk'])
def test_incomplete_or_mixed_evidence_is_rejected_without_changing_inputs(tmp_path,problem):
    directory=tmp_path/'profile';make_profile(directory);sample=sample_path(directory)
    summary=run_path(directory,variant='off')/'summary.json';active=active_path(directory)
    if problem=='missing_rank_sample':sample_path(directory,rank=15,iteration=2).unlink()
    elif problem=='extra_sample':write(sample.with_name('sample_extra.json'),report.read(sample))
    elif problem=='wrong_iteration':mutate(sample,lambda d:d['metadata'].update(iteration=1))
    elif problem=='shape':mutate(sample,lambda d:d['metadata']['config'].update(B=8))
    elif problem=='correctness':mutate(sample,lambda d:d['metadata'].update(correctness='skipped'))
    elif problem=='frequency':mutate(sample,lambda d:d.update(frequency_hz=2_000_000_000))
    elif problem=='schema':mutate(sample,lambda d:fixtures.setword(d['raw'],0,1,13))
    elif problem=='source_sha':mutate(run_path(directory)/'summary.json',lambda d:d['results'][0]['comparison'].update(source_sha256='f'*64))
    elif problem=='wrong_seed':mutate(sample,lambda d:d['metadata'].update(seed=17))
    elif problem=='off_raw99':mutate(summary,lambda d:d['results'][0]['combine']['ranks'][15]['kernel_us'].pop())
    elif problem=='off_rank_mean':mutate(summary,lambda d:d['results'][0]['dispatch']['ranks'][15].update(mean_us=1))
    elif problem=='off_pooled_mean':mutate(summary,lambda d:d['results'][0]['dispatch'].update(mean_kernel_us=1))
    elif problem=='active_rank_missing':mutate(active,lambda d:d['ranks'].pop())
    elif problem=='active_task_count':mutate(active,lambda d:d['ranks'][15]['kernel_us'].pop())
    elif problem=='active_task_value':mutate(active,lambda d:d['ranks'][15]['kernel_us'].__setitem__(0,float('nan')))
    elif problem=='active_summary_mean':mutate(active,lambda d:d['summary'].update(mean_kernel_us=1))
    elif problem.startswith('sample_task_'):
        key,value={'sample_task_duration':('kernel_us',1),'sample_task_index':('sample_index',1),
                   'sample_task_count':('expected_samples',5),'sample_task_order':('sample_order','file_order')}[problem]
        mutate(sample,lambda d:d['metadata']['active_capture_profile'].update({key:value}))
    elif problem=='plan_source_sha':mutate(directory/'profile-plan.json',lambda d:d['build_metadata']['sparse'].update(source_sha256='not-a-sha'))
    elif problem=='plan_sdk':mutate(directory/'profile-plan.json',lambda d:d['build_metadata']['sparse'].update(sdk_library_sha256='f'*64))
    before=tree_hashes(directory)
    with pytest.raises((ValueError,KeyError,OSError,TypeError)):report.render(directory)
    assert tree_hashes(directory)==before


def test_run_sections_cannot_reuse_another_variant_directory(tmp_path):
    directory=tmp_path/'profile';make_profile(directory)
    mutate(directory/'profile-plan.json',lambda d:d['runs'][0].update(sparse_directory=d['runs'][0]['off_directory']))
    with pytest.raises(ValueError,match='distinct'):report.render(directory)


def test_size_limit_rejects_without_truncating_or_writing(tmp_path,monkeypatch):
    directory=tmp_path/'profile';make_profile(directory);before=tree_hashes(directory)
    monkeypatch.setattr(report,'MAX_BYTES',128)
    with pytest.raises(ValueError,match='above.*no output written|truncated'):report.render(directory)
    assert tree_hashes(directory)==before
