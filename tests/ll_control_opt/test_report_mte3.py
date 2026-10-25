"""Disk-backed report evidence checks: matched captures, scope and no invented attribution."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics

import pytest

ROOT=Path(__file__).resolve().parents[2]


def load(name, relative):
    spec=importlib.util.spec_from_file_location(name,ROOT/relative)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


report=load('mte3_report_tested','tests/ll_control_opt/report_mte3.py')
fixtures=load('mte3_raw_fixture','tests/ll_stage_timer/test_mte3_timer.py')
common=load('mte3_fixture_common','tests/ll_control_opt/test_report_baseline.py')
write,mutate,tree_hashes,section=common.write,common.mutate,common.tree_hashes,common.section
REVISION='c'*40
SHMEM='8e3fc610f91ae9e5e8cf13748cfb74826b272c2a'


@pytest.fixture(autouse=True)
def fixed_revision(monkeypatch):
    monkeypatch.setattr(report.profile,'DIAGNOSTIC_REVISION',REVISION)


def run_path(directory,condition='observe',seed=0):
    return Path(directory)/f'seed-{seed}'/condition


def sample_path(directory,condition='observe',seed=0,rank=1,iteration=0):
    return run_path(directory,condition,seed)/f'case_00000/normal/ll_stage_timer/dispatch/rank_{rank:03d}/sample_{iteration:04d}.json'


def active_path(directory,condition='observe',seed=0):
    return run_path(directory,condition,seed)/'case_00000/normal/active_capture_dispatch.json'


def make_profile(directory,seeds=(0,),iterations=3):
    directory=Path(directory)
    shared={key:hashlib.sha256(key.encode()).hexdigest() for key in
            ('sdk_library_sha256','sdk_manifest_sha256','source_sha256')}
    builds={}
    for variant in ('off','on'):
        hashes={key:hashlib.sha256((variant+key).encode()).hexdigest()
                for key in ('native_sha256','kernel_sha256','wheel_sha256')}
        builds[variant]=dict(revision=REVISION,timer='OFF' if variant=='off' else 'ON',
            timer_schema=16,timer_profile='mte3',timer_profile_id=7,send_cores=30,recv_cores=26,
            logical_group=4,high_throughput=False,shmem_revision=SHMEM,dependency_id='fixture-dependencies',**shared,**hashes)
    plan=dict(schema=1,revision=REVISION,baseline_reference=report.profile.BASELINE_REFERENCE,
        runner_revision='fixture-runner',send_cores=30,recv_cores=26,shape=copy.deepcopy(report.profile.suite.SHAPE),
        timer_iters=iterations,seeds=list(seeds),build_metadata=builds,
        dependency=dict(shmem_revision=SHMEM,dependency_id='fixture-dependencies'),runs=[])
    for index,seed in enumerate(seeds):
        labels=['observe','tx','rx'];labels=labels[index%3:]+labels[:index%3]
        run=dict(seed=seed,measured_route_seed=seed+2,condition_order=['off']+labels)
        plan['runs'].append(run)
        for condition in report.CONDITIONS:
            directory_run=run_path(directory,condition,seed);enabled=condition!='off'
            run[condition+'_directory']=directory_run.relative_to(directory).as_posix()
            build=builds['on' if enabled else 'off']
            identity=dict(source_revision=REVISION,control_revision=REVISION,source_sha256=build['source_sha256'],
                stage_timer=enabled,stage_timer_schema=16,timer_profile='mte3',timer_profile_id=7,
                stage_timer_slots=8,stage_timer_fields=8,stage_timer_recording='mte3-backpressure',
                stage_timer_points=list(fixtures.timer.POINTS),send_cores=30,recv_cores=26,active_aiv=56,
                stage_timer_wait_metrics=list(fixtures.timer.METRICS),
                stage_timer_wait_metric_fields=['sum_ticks','max_ticks','calls','empty_control_ticks'],
                stage_timer_wait_metric_word_offset=32,implementation='g4-r0-prefix',overlap_stage=3,core_protocol=19030,
                timer_mode=report.MODE[condition] if enabled else None,dispatch_only=True,
                logical_group=4,transport='mte',runtime_session='ll-mte',high_throughput_enabled=False,
                seed=seed,world_size=16,profile_capture=enabled)
            config=dict(seed=seed,warmup=10,num_iters=100,perf_tail=50,route='normal',
                ll_stage_timer=enabled,ll_timer_mode=report.MODE[condition],ll_timer_iters=iterations,
                ll_timer_frequency_hz=1_000_000_000,ll_profile_capture=enabled,
                skip_check=False,skip_perf_test=enabled,dispatch_only=True,correctness_repeats=3,
                use_1g_huge_pages=0,delay_rows=118260,delay_cols=30720)
            case=dict(config=plan['shape'],correctness='passed',case_id=0,route='normal',comparison=identity)
            if not enabled:
                ranks=[]
                for rank in range(16):
                    values=[9100.]*50+[100.+rank]*50
                    ranks.append(dict(rank=rank,samples=50,kernel_us=values,mean_us=statistics.mean(values[-50:])))
                case['dispatch']=dict(ranks=ranks,mean_kernel_us=statistics.mean(row['mean_us'] for row in ranks))
            else:
                task_base={'observe':110.,'tx':150.,'rx':130.}[condition]
                ranks=[dict(rank=rank,operation='dispatch',expected_samples=iterations,
                    measurement=report.MEASUREMENT,status='available',
                    kernel_us=[task_base+rank+i/10 for i in range(iterations)]) for rank in range(16)]
                active=dict(operation='dispatch',route='normal',config=plan['shape'],comparison=identity,
                    correctness='passed',diagnostic_iters=iterations,measurement=report.MEASUREMENT,
                    status='available',ranks=ranks,summary=dict(mean_kernel_us=statistics.mean(
                        value for row in ranks for value in row['kernel_us'])))
                write(active_path(directory,condition,seed),active)
            write(directory_run/'summary.json',dict(status='completed',world_size=16,case_count=1,result_count=1,
                  config=config,results=[case]))
            for rank in range(16):
                write(directory_run/f'loaded_rank_{rank:03d}.json',dict(rank=rank,libraries={
                    '/fixture/deep_ep/_C.test.so':build['native_sha256'],
                    '/fixture/deep_ep/libdeepep_kernels.so':build['kernel_sha256'],
                    '/fixture/shmem/lib/libshmem.so':build['sdk_library_sha256']}))
                if not enabled:continue
                for iteration in range(iterations):
                    task=active['ranks'][rank]
                    meta=dict(rank=rank,world_size=16,iteration=iteration,run_id=directory_run.name,
                        seed=seed,case_id=0,correctness='passed',comparison=identity,
                        config={**plan['shape'],'logical_group_size':4},
                        sampling=dict(after_perf=False,warmup=10,settle_iters=100,diagnostic_iters=iterations,
                            delay_rows=config['delay_rows'],delay_cols=config['delay_cols'],
                            group_backend='hccl',expert_timer=False),
                        active_capture_profile=dict(status='available',all_ranks_status='available',
                            measurement=report.MEASUREMENT,kernel_us=task['kernel_us'][iteration],
                            sample_index=iteration,expected_samples=iterations,
                            sample_order='unique_finite_profiler_start_time',reason=None))
                    raw=fixtures.raw(mode={'observe':1,'tx':4,'rx':5}[condition],rank=rank,iteration=iteration)
                    write(sample_path(directory,condition,seed,rank,iteration),dict(operation='dispatch',
                          frequency_hz=1_000_000_000,metadata=meta,raw=raw))
    write(directory/'profile-plan.json',plan)
    return plan


@pytest.mark.parametrize('seeds,iterations',[((0,),3),((0,17,101),5)])
def test_report_validates_every_condition_and_respects_return_limit(tmp_path,monkeypatch,seeds,iterations):
    make_profile(tmp_path,seeds,iterations);before=tree_hashes(tmp_path);checked=[]
    validate=report.profile.validate_loaded_ranks
    def check(directory,manifest):
        checked.append(directory.name);return validate(directory,manifest)
    monkeypatch.setattr(report.profile,'validate_loaded_ranks',check)
    data=report.render(tmp_path);text=data.decode('utf-8')
    assert 0<len(data)<=65535
    assert len(checked)==4*len(seeds)
    assert text.count('SEED requested=')==len(seeds)
    assert 'prequantized FP8' in text and 'No poll snapshot/empty-poll metrics' in text
    assert 'NOT a quantitative baseline attribution' in text
    assert tree_hashes(tmp_path)==before


def test_full_tasks_use_actual_profile_and_paired_original_wait_changes(tmp_path):
    make_profile(tmp_path)
    mutate(run_path(tmp_path)/'summary.json',lambda data:data['results'][0].update(dispatch={'mean_kernel_us':99999}))
    text=report.render(tmp_path).decode('utf-8')
    rows=section(text,'ACTUAL FULL DEVICE TASKS: OFF rank-tail pooled mean; ON actual attached captures')
    assert [float(row['task_us']) for row in rows]==[107.5,117.6,157.6,137.6]
    paired=section(text,'PAIRED WAIT TRANSFER: condition minus OBSERVE at the SAME seed/rank/core, after capture averaging')
    prep=next(row for row in paired if row['condition']=='tx' and row['original_metric']=='RX_PREPARE_VS')
    assert float(prep['delta_us_mean'])==-.8 and int(prep['decreased'])==480
    drain=next(row for row in paired if row['condition']=='rx' and row['original_metric']=='RX_FINAL_ALL')
    assert float(drain['delta_us_p50'])==-3.8
    assert '99999' not in text


def test_control_excess_remains_signed(tmp_path):
    make_profile(tmp_path)
    for path in run_path(tmp_path,'tx').rglob('sample_*.json'):
        def change(data):
            for core in range(30):fixtures.setword(data['raw'],core,51,7000)
        mutate(path,change)
    text=report.render(tmp_path).decode('utf-8')
    rows=section(text,'S-CORE MTE3 EXPOSURE: raw and immediate empty control; signed difference is intervention-only')
    row=next(row for row in rows if row['metric']=='TX_REUSE_MTE3')
    assert float(row['signed_excess_us_p50'])==-2


def test_observe_longest_core_is_held_fixed_across_interventions(tmp_path):
    make_profile(tmp_path)
    # TX has a different longest core, but must still compare OBSERVE's core0.
    for iteration in range(3):
        def change(data):
            old=data['raw'][1][2][5]
            fixtures.setword(data['raw'],1,21,old+500_000)
        mutate(sample_path(tmp_path,'tx',rank=0,iteration=iteration),change)
    text=report.render(tmp_path).decode('utf-8')
    rows=section(text,'OBSERVE-LONGEST CORE HELD FIXED: select max capture-mean work per rank ONCE; match same core in TX/RX, no clock alignment')
    assert rows[0]['core']=='0'
    assert rows[0]['OBS_work_us']==rows[0]['TX_work_us']=='99'
    assert rows[0]['intervention_scope']=='S/direct'


@pytest.mark.parametrize('condition',report.CONDITIONS)
@pytest.mark.parametrize('problem',['missing_rank','wrong_hash'])
def test_actual_loaded_libraries_required_in_all_four_conditions(tmp_path,condition,problem):
    make_profile(tmp_path);path=run_path(tmp_path,condition)/'loaded_rank_015.json'
    if problem=='missing_rank':path.unlink()
    else:mutate(path,lambda data:data['libraries'].update({'/fixture/deep_ep/libdeepep_kernels.so':'f'*64}))
    with pytest.raises(ValueError,match='loaded|Loaded'):report.render(tmp_path)


@pytest.mark.parametrize('problem',['missing_sample','extra_sample','seed','iteration','shape','correctness',
    'frequency','mode','source','off_raw99','off_tail_mean','active_count','active_mean','active_value',
    'active_pair_value','active_pair_index','active_pair_order','reuse_directory','build_sdk','build_source',
    'packet_count_iteration','packet_count_mode','condition_order','build_source_mismatch','lane_mode',
    'actual_fallback','missing_prepare_wait','missing_final_wait','missing_forced_wait','disjoint_waits_exceed_work'])
def test_corrupt_or_mismatched_evidence_is_rejected_without_writes(tmp_path,problem):
    make_profile(tmp_path);sample=sample_path(tmp_path);active=active_path(tmp_path)
    if problem=='missing_sample':sample_path(tmp_path,rank=15,iteration=2).unlink()
    elif problem=='extra_sample':write(sample.with_name('sample_9999.json'),report.read(sample))
    elif problem in ('seed','iteration','shape','correctness'):
        def change(data):
            if problem=='shape':data['metadata']['config']['B']=8
            else:data['metadata'][problem]={'seed':17,'iteration':2,'correctness':'skipped'}[problem]
        mutate(sample,change)
    elif problem=='frequency':mutate(sample,lambda data:data.update(frequency_hz=2e9))
    elif problem=='mode':
        def change(data):
            for core in range(56):fixtures.setword(data['raw'],core,5,4|30<<8|(3 if core<30 else 2)<<16|7<<24)
        mutate(sample,change)
    elif problem=='source':mutate(run_path(tmp_path)/'summary.json',lambda data:data['results'][0]['comparison'].update(source_sha256='f'*64))
    elif problem=='off_raw99':mutate(run_path(tmp_path,'off')/'summary.json',lambda data:data['results'][0]['dispatch']['ranks'][15]['kernel_us'].pop())
    elif problem=='off_tail_mean':mutate(run_path(tmp_path,'off')/'summary.json',lambda data:data['results'][0]['dispatch']['ranks'][15].update(mean_us=1))
    elif problem=='active_count':mutate(active,lambda data:data['ranks'][15]['kernel_us'].pop())
    elif problem=='active_mean':mutate(active,lambda data:data['summary'].update(mean_kernel_us=1))
    elif problem=='active_value':mutate(active,lambda data:data['ranks'][15]['kernel_us'].__setitem__(0,float('nan')))
    elif problem.startswith('active_pair'):
        key,value={'active_pair_value':('kernel_us',1),'active_pair_index':('sample_index',1),
                   'active_pair_order':('sample_order','file_order')}[problem]
        mutate(sample,lambda data:data['metadata']['active_capture_profile'].update({key:value}))
    elif problem=='reuse_directory':mutate(tmp_path/'profile-plan.json',lambda data:data['runs'][0].update(tx_directory=data['runs'][0]['observe_directory']))
    elif problem=='build_sdk':mutate(tmp_path/'profile-plan.json',lambda data:data['build_metadata']['on'].update(sdk_library_sha256='f'*64))
    elif problem=='build_source':mutate(tmp_path/'profile-plan.json',lambda data:data['build_metadata']['on'].update(source_sha256='invalid'))
    elif problem=='build_source_mismatch':mutate(tmp_path/'profile-plan.json',lambda data:data['build_metadata']['on'].update(source_sha256='f'*64))
    elif problem=='lane_mode':
        for iteration in range(3):mutate(sample_path(tmp_path,'tx',iteration=iteration),lambda data:fixtures.setword(data['raw'],0,7,4|4<<8|1<<16))
    elif problem=='actual_fallback':mutate(sample,lambda data:fixtures.setword(data['raw'],0,7,3|3<<8))
    elif problem=='disjoint_waits_exceed_work':
        def change(data):
            for offset,value in enumerate([100_000,100_000,1,0]):fixtures.setword(data['raw'],0,44+offset,value)
        mutate(sample,change)
    elif problem in ('missing_prepare_wait','missing_final_wait','missing_forced_wait'):
        metric={'missing_prepare_wait':1,'missing_final_wait':3,'missing_forced_wait':4}[problem]
        path=sample_path(tmp_path,'tx') if problem=='missing_forced_wait' else sample
        def change(data):
            for offset in range(4):fixtures.setword(data['raw'],0,32+4*metric+offset,0)
        mutate(path,change)
    elif problem.startswith('packet_count'):
        paths=[sample] if problem=='packet_count_iteration' else [sample_path(tmp_path,'tx',rank=1,iteration=i) for i in range(3)]
        for path in paths:mutate(path,lambda data:fixtures.setword(data['raw'],0,34,9))
    elif problem=='condition_order':mutate(tmp_path/'profile-plan.json',lambda data:data['runs'][0].update(condition_order=['off','tx','tx','rx']))
    before=tree_hashes(tmp_path)
    with pytest.raises((ValueError,KeyError,OSError,TypeError)):report.render(tmp_path)
    assert tree_hashes(tmp_path)==before


def test_missing_active_task_never_fabricates_overhead_or_main_cause(tmp_path):
    make_profile(tmp_path)
    def change(data):
        data['status']='unavailable';data.pop('summary')
        data['ranks'][15].update(status='unavailable',kernel_us=[],reason='missing task')
    mutate(active_path(tmp_path),change)
    for rank in range(16):
        for iteration in range(3):
            def change_sample(data,rank=rank):
                data['metadata']['active_capture_profile']['all_ranks_status']='unavailable'
                if rank==15:data['metadata']['active_capture_profile'].update(status='unavailable',kernel_us=None)
            mutate(sample_path(tmp_path,rank=rank,iteration=iteration),change_sample)
    text=report.render(tmp_path).decode('utf-8')
    tasks=section(text,'ACTUAL FULL DEVICE TASKS: OFF rank-tail pooled mean; ON actual attached captures')
    assert tasks[1]['task_us']==tasks[1]['vs_OFF_pct']=='NA'
    assert tasks[2]['vs_OBSERVE_pct']=='NA'
    assert 'no perturbation or primary-bottleneck conclusion' in text


def test_report_bytes_limit_is_checked_without_truncation(tmp_path,monkeypatch):
    make_profile(tmp_path);before=tree_hashes(tmp_path)
    monkeypatch.setattr(report,'MAX_BYTES',128)
    with pytest.raises(ValueError,match='above.*no output written'):report.render(tmp_path)
    assert tree_hashes(tmp_path)==before
