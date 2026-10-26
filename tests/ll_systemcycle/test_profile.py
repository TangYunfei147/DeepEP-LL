"""CPU checks for matched OFF/ON profiling; no Torch/NPU execution."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest


_spec=importlib.util.spec_from_file_location('profile_systemcycle_tests',Path(__file__).with_name('profile.py'))
profile=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(profile)
build=profile.build


def git(root,*args):
    return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()


@pytest.fixture
def project(tmp_path,monkeypatch):
    root=tmp_path/'project with spaces';root.mkdir()
    subprocess.run(['git','init',str(root)],check=True,capture_output=True)
    git(root,'config','user.name','profile fixture');git(root,'config','user.email','fixture@example.invalid')
    (root/'.gitignore').write_text('.tmp/\nresults/\nthird-party/\n')
    (root/'dependencies.lock.json').write_text('{"shmem":{"revision":"sdk-source"}}\n')
    (root/'kernel.h').write_text('fixed baseline\n')
    git(root,'add','.');git(root,'commit','-qm','baseline')
    reference=git(root,'rev-parse','HEAD')
    (root/'timer.h').write_text('systemcycle instrumentation\n')
    git(root,'add','.');git(root,'commit','-qm','systemcycle instrumentation')
    revision=git(root,'rev-parse','HEAD')
    monkeypatch.setattr(profile,'BASELINE_REFERENCE',reference)
    monkeypatch.setattr(profile,'STARTING_BASELINE',reference)
    monkeypatch.setattr(profile,'DIAGNOSTIC_REVISION',revision)
    args=profile.parser().parse_args(['--devices',','.join(map(str,range(16))),
        '--master-addr','127.0.0.1','--master-port','29600','--shmem-port','29601','--run-prefix','profile-001'])
    return root,args,revision


def test_plan_uses_same_source_off_on_wheels_and_true_capture_tasks(project):
    root,args,revision=project
    plan=profile.make_plan(args,root)
    assert plan['revision']==revision and plan['seeds']==[0,17,101] and plan['timer_iters']==5
    assert plan['runner_revision']==git(root,'rev-parse','HEAD')
    assert plan['baseline_reference'] != plan['revision']
    assert plan['schema']==1 and set(plan['builds'])=={'off','on'}
    assert plan['builds']['off']['source']==plan['builds']['on']['source']
    assert plan['builds']['off']['venv']!=plan['builds']['on']['venv']
    assert plan['send_cores']==30 and plan['recv_cores']==26
    assert len(plan['runs'])==3
    for item in plan['runs']:
        assert item['measured_route_seed']==item['seed']+2
        for variant in ('off','profile'):
            command=item[variant+'_command']
            for flag,value in {'--num-processes':'16','--num-nodes':'1','--node-rank':'0','--num-tokens':'256',
                '--num-max-tokens-per-rank':'256','--hidden':'3584','--num-topk':'8','--num-experts':'384',
                '--dtype':'fp8','--do-cpu-sync':'0','--warmup':'10','--num-iters':'100',
                '--perf-tail':'50','--use-1g-huge-pages':'0'}.items():
                assert command[command.index(flag)+1]==value
            assert '--dispatch-only' not in command
            assert 'relay-complete' not in command and 'drain' not in command
        off=item['off_command']
        assert all(flag not in off for flag in ('--ll-stage-timer','--skip-perf-test','--ll-profile-capture'))
        for label,mode in (('profile','observe'),):
            command=item[label+'_command']
            assert all(flag in command for flag in ('--ll-stage-timer','--skip-perf-test','--ll-profile-capture'))
            assert command[command.index('--ll-timer-mode')+1]==mode
            assert command[command.index('--ll-timer-iters')+1]=='5'
            assert command[0]==plan['builds']['on']['python']
        assert off[0]==plan['builds']['off']['python']
    assert [item['condition_order'] for item in plan['runs']]==[
        ['off','profile'],['off','profile'],['off','profile']]
    assert plan['configuration']['dispatch_only'] is False
    assert plan['configuration']['timer_mode']=='observe'
    assert plan['configuration']['timer_schema']==18
    assert plan['configuration']['timer_profile']=='systemcycle'
    assert plan['configuration']['timer_profile_id']==9
    assert plan['configuration']['recording']=='systemcycle-stages'
    assert plan['builds']['off']['timer']=='OFF' and plan['builds']['on']['timer']=='ON'
    assert 'll_systemcycle' in plan['export_command'][1] and plan['export_command'][1].endswith('report.py')
    assert plan['export_command'][plan['export_command'].index('--output')+1].endswith('stage-report.txt')
    assert plan['export_command'][-1].endswith('stage-times.xlsx')
    assert not (root/profile.STATE).exists()


@pytest.mark.parametrize('iterations',[1,3,5])
def test_build_only_needs_no_devices_or_endpoints(project,iterations):
    root,args,_=project
    args.build_only=True;args.devices=None;args.master_addr=None;args.master_port=None;args.shmem_port=None
    args.timer_iters=iterations
    plan=profile.make_plan(args,root)
    assert plan['runs']==[] and plan['timer_iters']==iterations


@pytest.mark.parametrize('field,value',[('devices','0,1'),('run_prefix','../bad'),('master_addr',None),
    ('master_port',None),('shmem_port',29600),('seeds',[0,0]),('seeds',[0,1,2,3]),('output_dir',Path('../outside'))])
def test_invalid_plan_is_rejected_before_build(project,field,value):
    root,args,_=project;setattr(args,field,value)
    with pytest.raises(ValueError):profile.make_plan(args,root)


@pytest.mark.parametrize('timer',['OFF','ON'])
def test_environment_forces_systemcycle_profile_without_mutating_parent(project,monkeypatch,timer):
    root,_,_=project
    monkeypatch.setenv('DEEPEP_LL_STAGE_TIMER','OFF')
    monkeypatch.setenv('DEEPEP_BUILD_HT','ON')
    env=profile.profile_environment(root,timer)
    assert env['DEEPEP_LL_STAGE_TIMER']==timer and env['DEEPEP_BUILD_HT']=='OFF'
    assert env['DEEPEP_LL_TIMER_PROFILE']==env['DEEPEP_LL_EXPECT_TIMER_PROFILE']=='systemcycle'
    assert env['DEEPEP_LL_SEND_CORES']=='30' and env['DEEPEP_LL_RECV_CORES']=='26'
    import os
    assert os.environ['DEEPEP_LL_STAGE_TIMER']=='OFF' and os.environ['DEEPEP_BUILD_HT']=='ON'


def test_builds_share_source_keep_artifacts(project,monkeypatch):
    root,args,revision=project;plan=profile.make_plan(args,root)
    off=root/build.STATE/'wheels'/'old-off';off.mkdir(parents=True)
    (off/'sentinel').write_bytes(b'OFF cache stays unchanged')
    monkeypatch.setattr(build,'copy_local_dependencies',lambda *args:None)
    monkeypatch.setattr(build,'check_dependencies',lambda *args:None)
    real_run=subprocess.run;calls=[]
    def run(command,*pos,**kw):
        if command[0]=='git':return real_run(command,*pos,**kw)
        calls.append(command)
        source=Path(kw['cwd'])
        assert kw['env']['DEEPEP_LL_STAGE_TIMER'] in ('ON','OFF') and kw['env']['DEEPEP_BUILD_HT']=='OFF'
        assert kw['env']['DEEPEP_LL_TIMER_PROFILE']=='systemcycle'
        if command[1]=='scripts/build_shmem.py':
            sdk=source/build.SDK;(sdk/'lib').mkdir(parents=True,exist_ok=True)
            build.write_json(sdk/'deepep-sdk.json',dict(revision='sdk-source'))
            (sdk/'lib/libshmem.so').write_bytes(b'sdk')
        else:
            assert '--no-build-isolation' in command and '--no-deps' in command
            output=Path(command[command.index('--wheel-dir')+1])
            with zipfile.ZipFile(output/'fixture.whl','w') as archive:
                archive.writestr('deep_ep/_C.fixture.so',b'host')
                archive.writestr('deep_ep/libdeepep_kernels.so',kw['env']['DEEPEP_LL_STAGE_TIMER'].encode())
                archive.writestr('deep_ep/_build_info.json',json.dumps(dict(source_sha256='a'*64)))
        return subprocess.CompletedProcess(command,0)
    monkeypatch.setattr(profile.subprocess,'run',run)
    first=profile.build_wheel(plan,'off',root)
    on=profile.build_wheel(plan,'on',root)
    assert len(calls)==4 and first['revision']==on['revision']==revision
    assert first['timer']=='OFF' and on['timer']=='ON'
    assert first['timer_schema']==on['timer_schema']==18
    assert first['timer_bytes_per_core']==on['timer_bytes_per_core']==1024
    assert first['timer_profile']==on['timer_profile']=='systemcycle'
    assert first['timer_profile_id']==on['timer_profile_id']==9
    assert first['kernel_sha256']!=on['kernel_sha256']
    assert first['sdk_library_sha256']==on['sdk_library_sha256']
    assert profile.build_wheel(plan,'off',root)==first
    assert profile.build_wheel(plan,'on',root)==on and len(calls)==4
    assert (off/'sentinel').read_bytes()==b'OFF cache stays unchanged'
    assert Path(plan['builds']['off']['source']).is_relative_to(root/profile.STATE)
    assert git(Path(plan['builds']['off']['source']),'rev-parse','HEAD')==revision


@pytest.mark.parametrize('timer',['OFF','ON'])
def test_preflight_checks_schema_profile_ht_and_actual_libraries(timer):
    code=profile.preflight_code(dict(timer=timer,native_sha256='host',kernel_sha256='kernel',sdk_library_sha256='sdk',source_sha256='a'*64))
    expected='assert bool' if timer=='ON' else 'assert not bool'
    assert expected+'(n.low_latency_stage_timer_enabled)' in code
    compile(code,'systemcycle-preflight','exec')
    assert 'layout["schema"]==18' in code and 'n.high_throughput_enabled is False' in code
    assert 'low_latency_stage_timer_profile=="systemcycle"' in code and 'low_latency_stage_timer_profile_id==9' in code
    assert 'create_low_latency_runtime' in code and '/proc/self/maps' in code
    assert 'kernel' in code and 'sdk' in code
    assert 'import torch, torch_npu' in code
    assert '_build_info.json' in code and 'source_sha256' in code
    assert 'layout["recording"]=="systemcycle-stages"' in code
    assert 'layout["slots"]==16' in code and 'layout["fields"]==8' in code
    assert 'layout["slots"]*layout["fields"]*8==1024' in code
    assert 'n.low_latency_overlap_stage==3' in code
    for field in ('points','cycle_metric_names','cycle_metric_indices','counter_indices','metric_word_offset','metric_record_words'):
        assert 'layout["'+field+'"]' in code


@pytest.mark.parametrize('field,value',[
    ('timer_schema',16),('timer_profile','mte3'),('timer_profile_id',7),
    ('timer_bytes_per_core',512),('send_cores',28),('high_throughput',True)])
def test_old_profile_or_different_build_manifest_is_rejected(project,field,value):
    root,args,_=project;plan=profile.make_plan(args,root)
    manifest=Path(plan['builds']['on']['manifest'])
    manifest.parent.mkdir(parents=True)
    data=dict(revision=plan['revision'],timer='ON',timer_schema=18,
              timer_profile='systemcycle',timer_profile_id=9,timer_bytes_per_core=1024,
              send_cores=30,recv_cores=26,logical_group=4,
              build_type='Release',high_throughput=False,**plan['dependency'])
    data[field]=value
    manifest.write_text(json.dumps(data),encoding='utf-8')
    with pytest.raises(ValueError,match=field):
        profile.verify_manifest(manifest,plan,'on',root)


@pytest.mark.parametrize('problem',[None,'missing','duplicate','wrong-kernel','wrong-sdk','wrong-native','missing-native','duplicate-native'])
def test_all16_actual_loaded_libraries_must_match(tmp_path,problem):
    for rank in range(16):
        if problem=='missing' and rank==15:continue
        record=dict(rank=0 if problem=='duplicate' and rank==15 else rank,libraries={
            '/venv/deep_ep/_C.fixture.so':'other' if problem=='wrong-native' and rank==4 else 'host',
            '/venv/deep_ep/libdeepep_kernels.so':'other' if problem=='wrong-kernel' and rank==9 else 'kernel',
            '/sdk/lib/libshmem.so':'other' if problem=='wrong-sdk' and rank==6 else 'sdk'})
        if problem=='missing-native' and rank==4:record['libraries'].pop('/venv/deep_ep/_C.fixture.so')
        if problem=='duplicate-native' and rank==4:record['libraries']['/old/deep_ep/_C.other.so']='host'
        build.write_json(tmp_path/f'loaded_rank_{rank:03}.json',record)
    if problem is None:
        profile.validate_loaded_ranks(tmp_path,dict(native_sha256='host',kernel_sha256='kernel',sdk_library_sha256='sdk'))
    else:
        with pytest.raises(ValueError):profile.validate_loaded_ranks(tmp_path,dict(native_sha256='host',kernel_sha256='kernel',sdk_library_sha256='sdk'))


@pytest.mark.parametrize('outcome',['success','busy','preflight-failed','large-report','missing-workbook','build-only','run-only'])
def test_main_stops_before_devices_or_keeps_bounded_report(project,monkeypatch,outcome):
    root,args,_=project;calls=[];real_run=subprocess.run
    args.build_only=outcome=='build-only';args.run_only=outcome=='run-only'
    monkeypatch.setattr(profile,'ROOT',root)
    monkeypatch.setattr(profile,'parser',lambda:SimpleNamespace(parse_args=lambda:args))
    data={variant:dict(timer='OFF' if variant=='off' else 'ON',native_sha256='host',kernel_sha256='kernel-'+variant,
                      sdk_library_sha256='sdk',wheel='fixture.whl',sdk='sdk',source_sha256='a'*64,revision='revision',
                      dependency_id='dependency',shmem_revision='sdk-source',sdk_manifest_sha256='manifest') for variant in ('off','on')}
    monkeypatch.setattr(profile,'build_wheel',lambda plan,variant,root: calls.append(('build',variant)) or data[variant])
    monkeypatch.setattr(profile,'verify_manifest',lambda path,plan,variant,root: calls.append(('verify',variant)) or data[variant])
    monkeypatch.setattr(build,'acquire_lock',lambda path:lambda:calls.append('release'))
    # Atomic writer correctness has independent tests; this fixture checks the
    # launch/build/report sequence without Windows hard-link sandbox access.
    monkeypatch.setattr(profile.suite,'write_new_json',build.write_json)
    monkeypatch.setattr(profile.suite,'_write_new_file',lambda path,text:Path(path).write_text(text))
    def idle(devices):
        calls.append('idle')
        if outcome=='busy':raise RuntimeError('selected devices busy')
        return 'all selected devices idle'
    monkeypatch.setattr(profile.suite,'require_idle',idle)
    def run(command,*a,**kw):
        if command[0]=='git':return real_run(command,*a,**kw)
        if 'venv' in command:
            destination=Path(command[-1]);destination.mkdir(parents=True)
            (destination/'pyvenv.cfg').write_text('fixture')
            python=destination/('Scripts/python.exe' if sys.platform=='win32' else 'bin/python')
            python.parent.mkdir();python.write_bytes(b'not executable')
            calls.append('venv')
        elif 'pip' in command:calls.append('install')
        elif '-c' in command:
            calls.append('preflight')
            if outcome=='preflight-failed':raise subprocess.CalledProcessError(1,command)
        elif command[1].endswith('ll_systemcycle/report.py') or command[1].endswith('ll_systemcycle\\report.py'):
            calls.append('export')
            Path(command[command.index('--output')+1]).write_bytes(b'x'*60001 if outcome=='large-report' else b'one readable report\n')
            if outcome!='missing-workbook':Path(command[command.index('--xlsx')+1]).write_bytes(b'workbook fixture')
        else:pytest.fail(f'Unexpected external command: {command}')
        return subprocess.CompletedProcess(command,0)
    monkeypatch.setattr(profile.subprocess,'run',run)
    def run_st(command,log,env):
        variant='on' if '--ll-stage-timer' in command else 'off'
        label='profile' if variant=='on' else 'off'
        assert env['DEEPEP_LL_STAGE_TIMER']==data[variant]['timer'] and env['DEEPEP_BUILD_HT']=='OFF'
        seed=int(command[command.index('--seed')+1]);calls.append(('st',label,seed))
        directory=Path(command[command.index('--output-dir')+1])/command[command.index('--run-name')+1]
        directory.mkdir()
        for rank in range(16):
            build.write_json(directory/f'loaded_rank_{rank:03}.json',dict(rank=rank,libraries={
                '/venv/deep_ep/_C.fixture.so':'host','/venv/libdeepep_kernels.so':'kernel-'+variant,'/sdk/libshmem.so':'sdk'}))
    monkeypatch.setattr(profile.suite,'run_logged',run_st)
    if outcome in ('busy','preflight-failed'):
        with pytest.raises(RuntimeError,match='busy' if outcome=='busy' else 'preflight failed'):profile.main()
        assert not any(isinstance(call,tuple) and call[0]=='st' for call in calls)
        assert 'export' not in calls
    elif outcome=='build-only':
        profile.main()
        assert 'idle' not in calls and 'export' not in calls
        assert calls.count('preflight')==2
    elif outcome in ('large-report','missing-workbook'):
        with pytest.raises(ValueError,match='60000' if outcome=='large-report' else 'workbook'):profile.main()
        assert 'export' in calls
    else:
        profile.main()
        build_action='verify' if args.run_only else 'build'
        assert calls==[(build_action,'off'),(build_action,'on'),'venv','install','preflight','venv','install','preflight',
                      'idle',('st','off',0),'idle',('st','profile',0),
                      'idle',('st','off',17),'idle',('st','profile',17),
                      'idle',('st','off',101),'idle',('st','profile',101),'export','release']
        report=root/args.output_dir/args.run_prefix/'stage-report.txt'
        assert report.stat().st_size<=60000
        plan=json.loads((report.parent/'profile-plan.json').read_text())
        assert plan['build_metadata']==data and plan['timer_iters']==5
        with pytest.raises(ValueError,match='fresh'):profile.main()
    assert calls[-1]=='release'


@pytest.mark.parametrize('field',['revision','source_sha256','dependency_id','shmem_revision',
                                'sdk_manifest_sha256','sdk_library_sha256'])
def test_off_on_build_inputs_must_match(field):
    data=dict(revision='rev',source_sha256='a'*64,dependency_id='deps',shmem_revision='sdk-src',
              sdk_manifest_sha256='sdk-manifest',sdk_library_sha256='sdk-library')
    profile.validate_matched_builds(dict(off=data,on=dict(data)))
    with pytest.raises(ValueError,match=field):
        profile.validate_matched_builds(dict(off=data,on=dict(data,**{field:'different'})))
    missing=dict(data);missing.pop(field)
    with pytest.raises(ValueError,match=field):
        profile.validate_matched_builds(dict(off=missing,on=dict(data)))
