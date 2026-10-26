"""Complete disk evidence and corrupt-record rejection, without device imports."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[2]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


report=load('bw_report_tested','tests/ll_control_opt/report_bandwidth.py')
old=load('bw_old_fixture','tests/ll_control_opt/test_report_mte3.py')
fixture=load('bw_raw_fixture','tests/ll_stage_timer/test_bandwidth_timer.py')
write,mutate,tree_hashes,section=old.write,old.mutate,old.tree_hashes,old.section
REVISION=old.REVISION


@pytest.fixture(autouse=True)
def revision(monkeypatch):monkeypatch.setattr(report.profile,'DIAGNOSTIC_REVISION',REVISION)


def make_profile(root,seeds=(0,),iterations=3):
    source=root/'historical-fixture';directory=root/'actual'
    plan=old.make_profile(source,seeds,iterations)
    def convert_identity(identity):
        identity.update(stage_timer_schema=17,timer_profile='bandwidth',timer_profile_id=8,
            stage_timer_slots=16,stage_timer_fields=8,stage_timer_recording='stage-bandwidth',
            stage_timer_points=list(fixture.timer.POINTS),
            stage_timer_bandwidth_stages=list(fixture.timer.STAGES),stage_timer_bandwidth_fields=list(fixture.timer.FIELDS),
            stage_timer_bandwidth_word_offset=32,stage_timer_bandwidth_record_words=16)
        for key in list(identity):
            if key.startswith('stage_timer_wait_'):identity.pop(key)
        if identity.get('timer_mode')=='mte3-tx':identity['timer_mode']='bandwidth-complete'
    plan['baseline_reference']=report.profile.BASELINE_REFERENCE
    for build in plan['build_metadata'].values():
        build.update(timer_schema=17,timer_profile='bandwidth',timer_profile_id=8,timer_bytes_per_core=1024)
    for index,run in enumerate(plan['runs']):
        run.pop('rx_directory');run['complete_directory']=run.pop('tx_directory').replace('/tx','/complete')
        run['condition_order']=['off','observe','complete'] if index%2==0 else ['off','complete','observe']
    for path in source.rglob('*.json'):
        relative=path.relative_to(source)
        if path.name=='profile-plan.json' or 'rx' in relative.parts:continue
        value=json.loads(path.read_text())
        if path.name=='summary.json':
            config=value['config']
            if config['ll_timer_mode']=='mte3-tx':config['ll_timer_mode']='bandwidth-complete'
            convert_identity(value['results'][0]['comparison'])
        elif path.name.startswith('active_capture_'):convert_identity(value['comparison'])
        elif path.name.startswith('sample_'):
            meta=value['metadata'];convert_identity(meta['comparison'])
            if meta['run_id']=='tx':meta['run_id']='complete'
            value['raw']=fixture.raw(mode=6 if 'tx' in relative.parts else 1,rank=meta['rank'],iteration=meta['iteration'])
        target=directory.joinpath(*('complete' if piece=='tx' else piece for piece in relative.parts))
        write(target,value)
    write(directory/'profile-plan.json',plan)
    return directory


def sample(directory,condition='observe',rank=0,iteration=0):
    return directory/f'seed-0/{condition}/case_00000/normal/ll_stage_timer/dispatch/rank_{rank:03d}/sample_{iteration:04d}.json'


@pytest.mark.parametrize('seeds,iterations',[((0,),3),((0,17,101),5)])
def test_full_report_checks_all_records_and_is_bounded(tmp_path,seeds,iterations):
    directory=make_profile(tmp_path,seeds,iterations);before=tree_hashes(directory)
    data=report.render(directory);text=data.decode()
    assert 0<len(data)<=65535 and text.count('SEED requested=')==len(seeds)
    assert 'neither baseline' in fixture.timer.__doc__ and 'not only DMA' in text
    assert tree_hashes(directory)==before
    (tmp_path/'example-report.txt').write_bytes(data)


def test_calculated_phase_and_full_task_rates_use_their_own_denominators(tmp_path):
    directory=make_profile(tmp_path)
    text=report.render(directory).decode()
    full=section(text,'ACTUAL FULL DEVICE TASKS: OFF rank-tail pooled mean; ON actual attached captures')
    assert [float(row['task_us']) for row in full]==[107.5,117.6,157.6]
    phase=section(text,'COMPLETION EFFECTIVE MOVEMENT GB/s: intervention only; per-core elapsed includes self and peer work')
    relay=next(row for row in phase if row['stage']=='RELAY' and row['initial_role']=='S0..29')
    assert float(relay['peer_GBps_time_weighted'])==49.152
    assert float(relay['self_GBps_time_weighted'])==16.384
    normalized=section(text,'FULL-TASK-NORMALIZED GB/s: per-rank total phase bytes / SAME rank complete task, then arithmetic mean')
    source=next(row for row in normalized if row['condition']=='off' and row['stage']=='REMOTE_FIRST_HOP')
    expected=sum(30*24*4096/(100+rank)/1000 for rank in range(16))/16
    assert float(source['peer_GBps_mean'])==pytest.approx(expected,abs=5e-4)


@pytest.mark.parametrize('fault',['missing_sample','extra_sample','mode','bytes','global_packets','sample_pair',
    'source','sdk','loaded_hash','control','prefix','lane','same_rank_firsthop','complete_coverage'])
def test_invalid_evidence_rejected_without_writes(tmp_path,fault):
    directory=make_profile(tmp_path);path=sample(directory)
    if fault=='missing_sample':path.unlink()
    elif fault=='extra_sample':write(path.with_name('sample_9999.json'),report.read(path))
    elif fault=='mode':mutate(path,lambda data:fixture.setword(data['raw'],0,5,6|(30<<8)|(3<<16)|(8<<24)))
    elif fault=='bytes':mutate(path,lambda data:fixture.setmetric(data['raw'],0,'SOURCE_DIRECT','peer_bytes',1234))
    elif fault=='sample_pair':mutate(path,lambda data:data['metadata']['active_capture_profile'].update(kernel_us=1))
    elif fault=='source':mutate(directory/'profile-plan.json',lambda data:data['build_metadata']['on'].update(source_sha256='f'*64))
    elif fault=='sdk':mutate(directory/'profile-plan.json',lambda data:data['build_metadata']['on'].update(sdk_library_sha256='f'*64))
    elif fault=='loaded_hash':mutate(directory/'seed-0/complete/loaded_rank_015.json',lambda data:data['libraries'].update({'/fixture/deep_ep/libdeepep_kernels.so':'f'*64}))
    elif fault=='control':mutate(path,lambda data:fixture.setmetric(data['raw'],30,'COUNT_CONTROL','peer_bytes',32))
    elif fault=='prefix':mutate(path,lambda data:fixture.setmetric(data['raw'],30,'PREFIX_LOCAL','self_bytes',32))
    elif fault=='lane':mutate(path,lambda data:fixture.setword(data['raw'],0,7,3|(3<<8)))
    elif fault=='same_rank_firsthop':
        def change(data):
            for field,value in [('self_operations',1),('self_bytes',4096),('self_payload_bytes',3696)]:
                fixture.setmetric(data['raw'],0,'REMOTE_FIRST_HOP',field,value)
        mutate(path,change)
    elif fault=='complete_coverage':mutate(sample(directory,'complete'),lambda data:fixture.setmetric(data['raw'],0,'RELAY','completed_operations',7))
    elif fault=='global_packets':
        for condition in ('observe','complete'):
            for iteration in range(3):
                def change(data):
                    for field,value in [('peer_operations',11),('peer_bytes',11*4096),('peer_payload_bytes',11*3696)]:
                        fixture.setmetric(data['raw'],0,'SOURCE_DIRECT',field,value)
                    if condition=='complete':fixture.setmetric(data['raw'],0,'SOURCE_DIRECT','completed_operations',11)
                mutate(sample(directory,condition,iteration=iteration),change)
    before=tree_hashes(directory)
    with pytest.raises((ValueError,OSError,KeyError)):report.render(directory)
    assert tree_hashes(directory)==before


def test_limit_rejects_instead_of_truncating(tmp_path,monkeypatch):
    directory=make_profile(tmp_path);monkeypatch.setattr(report,'MAX_BYTES',100)
    with pytest.raises(ValueError,match='no output written'):report.render(directory)


def test_missing_profiler_keeps_stage_rates_but_never_invents_full_task_rates(tmp_path):
    directory=make_profile(tmp_path);run=directory/'seed-0/observe'
    path=run/'case_00000/normal/active_capture_dispatch.json'
    def unavailable(data):
        data['status']='unavailable';data['summary']={}
        data['ranks'][15].update(status='unavailable',kernel_us=[],reason='synthetic missing task')
    mutate(path,unavailable)
    for path in run.rglob('sample_*.json'):
        def change(data):
            meta=data['metadata'];meta['active_capture_profile']['all_ranks_status']='unavailable'
            if meta['rank']==15:meta['active_capture_profile'].update(status='unavailable',kernel_us=None)
        mutate(path,change)
    text=report.render(directory).decode()
    full=section(text,'ACTUAL FULL DEVICE TASKS: OFF rank-tail pooled mean; ON actual attached captures')
    assert full[1]['task_us']=='NA' and full[1]['status']=='unavailable'
    normalized=section(text,'FULL-TASK-NORMALIZED GB/s: per-rank total phase bytes / SAME rank complete task, then arithmetic mean')
    assert all(row['peer_GBps_mean']=='NA' for row in normalized if row['condition']=='observe')
    assert '49.152' in text
