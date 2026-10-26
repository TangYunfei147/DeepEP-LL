"""Synthetic CPU fixtures for SystemCycle report validation and Excel export."""
import copy
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import statistics
import sys
import xml.etree.ElementTree as ET
import zipfile

import pytest

ROOT=Path(__file__).resolve().parents[2]


def load(name,relative):
    spec=importlib.util.spec_from_file_location(name,ROOT/relative)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


report=load('systemcycle_report_tests','tests/ll_systemcycle/report.py')
fixtures=load('systemcycle_stage_fixtures','tests/ll_stage_timer/test_systemcycle_timer.py')
timer=report.timer
REVISION='c'*40
SHMEM='8e3fc610f91ae9e5e8cf13748cfb74826b272c2a'


@pytest.fixture(autouse=True)
def committed_identity(monkeypatch):
    # The runner's release SHA is filled in only after its implementation commit.
    monkeypatch.setattr(report.profile,'DIAGNOSTIC_REVISION',REVISION)


def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,separators=(',',':')),encoding='utf-8')


def mutate(path,change):
    value=report.read(path);change(value);write(path,value)


def run_path(directory,seed=0,variant='profile'):
    return Path(directory)/f'seed-{seed}-{variant}'


def sample_path(directory,seed=0,operation='dispatch',rank=1,iteration=0):
    return run_path(directory,seed)/f'case_00000/normal/ll_stage_timer/{operation}/rank_{rank:03d}/sample_{iteration:04d}.json'


def active_path(directory,seed=0,operation='dispatch'):
    return run_path(directory,seed)/f'case_00000/normal/active_capture_{operation}.json'


def raw(operation,rank=0,iteration=0):
    return fixtures.raw(operation,rank,iteration)


def make_profile(directory,seeds=(0,),iterations=3):
    directory=Path(directory)
    shared={key:hashlib.sha256(key.encode()).hexdigest() for key in
            ('sdk_library_sha256','sdk_manifest_sha256','source_sha256')}
    builds={}
    for variant in ('off','on'):
        hashes={key:hashlib.sha256((variant+key).encode()).hexdigest()
                for key in ('native_sha256','kernel_sha256','wheel_sha256')}
        builds[variant]=dict(revision=REVISION,timer='OFF' if variant=='off' else 'ON',
            timer_schema=18,timer_profile='systemcycle',timer_profile_id=9,timer_bytes_per_core=1024,send_cores=30,recv_cores=26,
            logical_group=4,high_throughput=False,build_type='Release',shmem_revision=SHMEM,dependency_id='d'*64,**shared,**hashes)
    plan=dict(schema=1,revision=REVISION,baseline_reference=report.profile.BASELINE_REFERENCE,
              starting_baseline=report.profile.STARTING_BASELINE,synthetic_fixture=True,
              runner_revision='fixture-runner',send_cores=30,recv_cores=26,
              shape=copy.deepcopy(report.profile.suite.SHAPE),timer_iters=iterations,seeds=list(seeds),
              build_metadata=builds,dependency=dict(shmem_revision=SHMEM,dependency_id='d'*64),runs=[])
    plan['configuration']=dict(shape=plan['shape'],world_size=16,send_cores=30,recv_cores=26,logical_group=4,
        timer_schema=18,timer_profile='systemcycle',timer_profile_id=9,timer_mode='observe',
        recording='systemcycle-stages',timer_iters=iterations,frequency_hz=1_000_000_000,
        dispatch_only=False,high_throughput=False,use_1g_huge_pages=False,warmup=10,num_iters=100,
        perf_tail=50,correctness_repeats=3,seeds=list(seeds))
    for seed in seeds:
        plan['runs'].append(dict(seed=seed,measured_route_seed=seed+2,
            off_directory=run_path(directory,seed,'off').name,
            profile_directory=run_path(directory,seed).name,condition_order=['off','profile']))
        for variant in ('off','profile'):
            run=run_path(directory,seed,variant);enabled=variant=='profile';build=builds['on' if enabled else 'off']
            identity=dict(source_revision=REVISION,control_revision=REVISION,source_sha256=build['source_sha256'],
                stage_timer=enabled,stage_timer_schema=18,timer_profile='systemcycle',timer_profile_id=9,timer_bytes_per_core=1024,
                stage_timer_slots=16,stage_timer_fields=8,stage_timer_recording='systemcycle-stages',
                stage_timer_points=list(timer.POINTS),stage_timer_cycle_metric_names=list(timer.METRIC_NAMES),
                stage_timer_cycle_metric_indices=list(timer.METRIC_INDICES),
                stage_timer_counter_indices=list(timer.COUNTER_INDICES),stage_timer_metric_word_offset=48,
                stage_timer_metric_record_words=2,implementation='g4-r0-prefix',core_protocol=19030,
                overlap_stage=3,source_dirty='False',correctness_repeats=3,send_cores=30,recv_cores=26,active_aiv=56,
                logical_group=4,transport='mte',runtime_session='ll-mte',high_throughput_enabled=False,
                seed=seed,world_size=16,profile_capture=enabled,timer_mode='observe' if enabled else None,
                dispatch_only=False)
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
    return list(csv.DictReader(io.StringIO(text.split(title+'\n',1)[1].split('\n\n',1)[0])))


@pytest.mark.parametrize('seeds,iterations',[((0,),3),((0,17,101),5)])
def test_complete_report(tmp_path,monkeypatch,seeds,iterations):
    directory=tmp_path/'profile with spaces';make_profile(directory,seeds,iterations)
    before=tree_hashes(directory);checked=[];original=report.profile.validate_loaded_ranks
    def validate(path,manifest):
        checked.append((path.name,manifest['timer']));return original(path,manifest)
    monkeypatch.setattr(report.profile,'validate_loaded_ranks',validate)
    evidence=report.read_all(directory);text=report.compact_report(evidence)
    assert len(text)<=60000 and len(checked)==len(seeds)*2
    assert text.count(b'SEED requested=')==len(seeds)
    assert b'NOT an algorithm speedup' in text and b'Missing=NA, not zero' in text
    assert b'schema18/profile9' in text
    payload=report.workbook_payload(evidence)
    assert len(payload['performance']['rows'])==2*len(seeds)
    assert set(payload['operations'])=={'dispatch','combine'}
    for operation in ('dispatch','combine'):
        assert len(payload['operations'][operation]['rows'])==len(seeds)*16*56
        capture_column=payload['operations'][operation]['headers'].index('采集轮次')
        assert {row[capture_column] for row in payload['operations'][operation]['rows']}=={iterations}
    assert len(payload['raw']['rows'])==len(seeds)*2*16*56
    assert tree_hashes(directory)==before


def test_actual_full_tasks_only(tmp_path):
    directory=tmp_path/'p';make_profile(directory)
    mutate(run_path(directory)/'summary.json',lambda d:d['results'][0].update(
        dispatch={'mean_kernel_us':999999},combine={'mean_kernel_us':888888}))
    text=report.render(directory).decode()
    rows=section(text,'FULL DEVICE TASKS: OFF rank-tail pooled mean vs actual attached-capture pooled mean')
    assert float(rows[0]['OFF_us'])==107.5 and float(rows[0]['active_ON_us'])==207.6
    assert float(rows[0]['perturbation_pct'])==pytest.approx(100*(207.6/107.5-1),rel=1e-5)
    assert float(rows[1]['OFF_us'])==307.5 and float(rows[1]['active_ON_us'])==507.6
    assert '999999' not in text and '888888' not in text


def test_per_core_means_before_rank_stats(tmp_path):
    directory=tmp_path/'p';make_profile(directory)
    evidence=report.read_all(directory);rows=evidence['runs'][0]['rows']
    # Three captures on one core must contribute one core mean, not three extra
    # independent cores. Missing fields stay absent; valid zero stays measured.
    selected=[row for row in rows if row['operation']=='dispatch' and row['core']<30]
    for row in selected:
        row['records'].pop('TOKEN_METADATA',None)
    special=[row for row in selected if (row['rank'],row['core'])==(0,0)]
    special[0]['records']['TOKEN_METADATA']=dict(kind='metric',total_ticks=0,calls=1)
    special[1]['records']['TOKEN_METADATA']=dict(kind='metric',total_ticks=1000,calls=1)
    special[2]['records']['TOKEN_METADATA']=dict(kind='metric',total_ticks=5000,calls=1)
    data={row[0]:row for row in report.stage_statistics(rows,'dispatch','S')}
    assert data['TOKEN_METADATA'][2:] == ('1/480',3,2,2,'0:0',2)
    assert data['COUNT_SEND'][2:]==('0/480',0,None,None,None,None)


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


def test_explicit_unavailable_profiler(tmp_path):
    directory=tmp_path/'p';make_profile(directory);unavailable(directory)
    evidence=report.read_all(directory);text=report.compact_report(evidence).decode()
    tasks=section(text,'FULL DEVICE TASKS: OFF rank-tail pooled mean vs actual attached-capture pooled mean')
    assert tasks[0]==dict(operation='dispatch',OFF_us='107.5',active_ON_us='NA',perturbation_pct='NA',status='unavailable')
    assert tasks[1]['status']=='available' and 'rank15:missing profiler task' in text
    payload=report.workbook_payload(evidence)
    assert payload['performance']['rows'][0][5:7]==[None,None]
    assert payload['operations']['dispatch']['rows'][0][10]['value']==2.8


@pytest.mark.parametrize('variant',('off','profile'))
@pytest.mark.parametrize('problem',('missing','native','kernel','sdk'))
def test_loaded_libraries(tmp_path,variant,problem):
    directory=tmp_path/'p';make_profile(directory)
    path=run_path(directory,variant=variant)/'loaded_rank_015.json'
    if problem=='missing':path.unlink()
    else:
        name={'native':'/fixture/deep_ep/_C.test.so','kernel':'/fixture/deep_ep/libdeepep_kernels.so',
              'sdk':'/fixture/shmem/lib/libshmem.so'}[problem]
        mutate(path,lambda d:d['libraries'].update({name:'f'*64}))
    with pytest.raises(ValueError,match='loaded|Loaded'):report.render(directory)


@pytest.mark.parametrize('fault',[
    'missing_sample','extra_sample','wrong_iteration','shape','correctness','frequency','schema',
    'source_sha','wrong_seed','off_99','off_zero','off_mean','off_pooled','active_file_missing',
    'active_rank_missing','active_count','active_nan','active_mean','task_duration','task_index',
    'task_count','task_order','plan_source','plan_sdk','plan_configuration','layout','off_sample',
    'runtime_mode','runtime_dispatch_only'])
def test_reject_mixed_evidence(tmp_path,fault):
    directory=tmp_path/'p';make_profile(directory);sample=sample_path(directory)
    off=run_path(directory,variant='off')/'summary.json';active=active_path(directory)
    if fault=='missing_sample':sample_path(directory,rank=15,iteration=2).unlink()
    elif fault=='extra_sample':write(sample.with_name('sample_extra.json'),report.read(sample))
    elif fault=='wrong_iteration':mutate(sample,lambda d:d['metadata'].update(iteration=1))
    elif fault=='shape':mutate(sample,lambda d:d['metadata']['config'].update(B=8))
    elif fault=='correctness':mutate(sample,lambda d:d['metadata'].update(correctness='skipped'))
    elif fault=='frequency':mutate(sample,lambda d:d.update(frequency_hz=2_000_000_000))
    elif fault=='schema':mutate(sample,lambda d:fixtures.setword(d['raw'],0,1,17))
    elif fault=='source_sha':mutate(run_path(directory)/'summary.json',lambda d:d['results'][0]['comparison'].update(source_sha256='f'*64))
    elif fault=='wrong_seed':mutate(sample,lambda d:d['metadata'].update(seed=17))
    elif fault=='off_99':mutate(off,lambda d:d['results'][0]['combine']['ranks'][15]['kernel_us'].pop())
    elif fault=='off_zero':mutate(off,lambda d:d['results'][0]['combine']['ranks'][15]['kernel_us'].__setitem__(0,0))
    elif fault=='off_mean':mutate(off,lambda d:d['results'][0]['dispatch']['ranks'][15].update(mean_us=1))
    elif fault=='off_pooled':mutate(off,lambda d:d['results'][0]['dispatch'].update(mean_kernel_us=1))
    elif fault=='active_file_missing':active.unlink()
    elif fault=='active_rank_missing':mutate(active,lambda d:d['ranks'].pop())
    elif fault=='active_count':mutate(active,lambda d:d['ranks'][15]['kernel_us'].pop())
    elif fault=='active_nan':mutate(active,lambda d:d['ranks'][15]['kernel_us'].__setitem__(0,float('nan')))
    elif fault=='active_mean':mutate(active,lambda d:d['summary'].update(mean_kernel_us=1))
    elif fault.startswith('task_'):
        key,value={'task_duration':('kernel_us',1),'task_index':('sample_index',1),
                   'task_count':('expected_samples',5),'task_order':('sample_order','file_order')}[fault]
        mutate(sample,lambda d:d['metadata']['active_capture_profile'].update({key:value}))
    elif fault=='plan_source':mutate(directory/'profile-plan.json',lambda d:d['build_metadata']['on'].update(source_sha256='not-a-sha'))
    elif fault=='plan_sdk':mutate(directory/'profile-plan.json',lambda d:d['build_metadata']['on'].update(sdk_library_sha256='f'*64))
    elif fault=='plan_configuration':mutate(directory/'profile-plan.json',lambda d:d['configuration'].update(recording='stage-bandwidth'))
    elif fault=='layout':mutate(run_path(directory)/'summary.json',lambda d:d['results'][0]['comparison'].update(stage_timer_metric_word_offset=32))
    elif fault=='off_sample':write(run_path(directory,variant='off')/'sample_0000.json',{})
    elif fault=='runtime_mode':mutate(run_path(directory)/'summary.json',lambda d:d['results'][0]['comparison'].update(timer_mode='drain'))
    elif fault=='runtime_dispatch_only':mutate(run_path(directory)/'summary.json',lambda d:d['results'][0]['comparison'].update(dispatch_only=True))
    before=tree_hashes(directory)
    with pytest.raises((ValueError,KeyError,OSError,TypeError)):report.render(directory)
    assert tree_hashes(directory)==before


def test_distinct_run_paths(tmp_path):
    directory=tmp_path/'p';make_profile(directory)
    mutate(directory/'profile-plan.json',lambda d:d['runs'][0].update(profile_directory=d['runs'][0]['off_directory']))
    with pytest.raises(ValueError,match='distinct'):report.render(directory)


def test_limit_before_output(tmp_path,monkeypatch):
    directory=tmp_path/'p';make_profile(directory);before=tree_hashes(directory)
    monkeypatch.setattr(report,'MAX_BYTES',128)
    with pytest.raises(ValueError,match='above.*no output written'):
        report.export(directory,directory/'out.txt',directory/'out.xlsx')
    assert tree_hashes(directory)==before


def test_xlsx_precision_and_missing(tmp_path):
    directory=tmp_path/'p';make_profile(directory);evidence=report.read_all(directory)
    payload=report.workbook_payload(evidence)
    assert payload['definitions']['rows'][0][1]==1000
    raw_table=payload['raw'];first=raw_table['rows'][0]
    entry_column=raw_table['headers'].index('word008_uint64_text')
    assert isinstance(first[entry_column],str) and int(first[entry_column])>2**63
    raw_ticks=raw_table['headers'].index('ENTRY_TO_WORK_DONE_ticks')
    assert first[raw_ticks]==2800
    dispatch=payload['operations']['dispatch'];row=dispatch['rows'][0]
    assert row[10]['value']==2.8 and row[10]['formula'].endswith("/'阶段说明'!$B$2")
    missing=next(i for i,h in enumerate(dispatch['headers']) if h.startswith('计数准备与提交'))
    assert row[missing] is None
    counter=next(i for i,h in enumerate(dispatch['headers']) if h.startswith('发送packet数'))
    assert row[counter]['value']==0 and '/' not in row[counter]['formula']
    all_payload=report.workbook_payload(evidence,'all')
    assert len(all_payload['raw']['rows'])==3*len(raw_table['rows'])
    capture_column=all_payload['operations']['combine']['headers'].index('采集轮次')
    assert {r[capture_column] for r in all_payload['operations']['combine']['rows']}=={1,2,3}
    path=directory/'stages.xlsx';report.xlsx.export_workbook(payload,path.resolve())
    with zipfile.ZipFile(path) as archive:
        ns={'m':report.xlsx.MAIN}
        book=ET.fromstring(archive.read('xl/workbook.xml'))
        assert [s.attrib['name'] for s in book.findall('m:sheets/m:sheet',ns)]==[
            '性能对照','Dispatch逐核耗时','Combine逐核耗时','阶段说明','原始Cycle']
        sheet=ET.fromstring(archive.read('xl/worksheets/sheet2.xml'))
        total=sheet.find(".//m:c[@r='K2']",ns)
        assert total.find('m:f',ns).text==row[10]['formula'][1:]
        assert float(total.find('m:v',ns).text)==2.8
        assert sheet.find(f".//m:c[@r='{report.xlsx.column_name(missing+1)}2']",ns) is None
        raw_xml=ET.fromstring(archive.read('xl/worksheets/sheet5.xml'))
        value=raw_xml.find(f".//m:c[@r='{report.xlsx.column_name(entry_column+1)}2']",ns)
        assert value.attrib['t']=='inlineStr' and value.find('m:is/m:t',ns).text==first[entry_column]


def test_export_from_another_cwd(tmp_path,monkeypatch,capsys):
    directory=tmp_path/'project with spaces'/'results';make_profile(directory,iterations=1)
    elsewhere=tmp_path/'another cwd';elsewhere.mkdir();monkeypatch.chdir(elsewhere)
    monkeypatch.setattr(report,'ROOT',directory.parent)
    monkeypatch.setattr(sys,'argv',['report.py','results','--output','results/summary.txt','--xlsx','results/timings.xlsx'])
    report.main()
    assert (directory/'summary.txt').stat().st_size<=60000 and (directory/'timings.xlsx').is_file()
    assert 'Send only' in capsys.readouterr().out
    before=tree_hashes(directory)
    with pytest.raises(SystemExit):report.main()
    assert tree_hashes(directory)==before


def test_partial_export_preserved(tmp_path,monkeypatch):
    directory=tmp_path/'p';make_profile(directory,iterations=1)
    text=directory/'out.txt';workbook=directory/'out.xlsx'
    original=report.xlsx.export_workbook
    def publish_then_conflict(payload,path):
        original(payload,path);text.write_text('concurrent writer',encoding='utf-8')
    monkeypatch.setattr(report.xlsx,'export_workbook',publish_then_conflict)
    with pytest.raises(RuntimeError,match='preserved.*new output names'):
        report.export(directory,text,workbook)
    assert workbook.is_file() and text.read_text()=='concurrent writer'
