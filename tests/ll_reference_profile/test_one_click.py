"""Real ST argument validation and isolated launcher orchestration; CPU only."""
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('reference_one_click',Path(__file__).with_name('one_click.py'))
launch=importlib.util.module_from_spec(spec);spec.loader.exec_module(launch)
DEVICES=','.join(map(str,range(16)))


def argv(*extra):return ['0','127.0.0.1','case-001','--devices',DEVICES,*extra]


def test_real_st_configuration_and_reference_defaults():
    args=launch.parse_args(argv());launch.validate_test_arguments(args)
    values=launch.test_arguments(args)
    assert '--dispatch-only' in values and '--ll-reference-profile' in values and '--skip-perf-test' in values
    assert values[values.index('--ll-timer-iters')+1]=='1'
    assert values[values.index('--backend')+1]=='mte'
    assert all(value not in values for value in ('--ll-stage-timer','--ll-profile-capture','--ll-expert-timer'))
    perf=launch.parse_args(argv('--mode','perf'));launch.validate_test_arguments(perf)
    assert '--ll-reference-profile' not in launch.test_arguments(perf)
    assert '--dispatch-only' in launch.test_arguments(perf)


@pytest.mark.parametrize('extra',[
    ['--mode','timer'],['--devices','0,1'],['--devices',','.join(['0']*16)],
    ['--num-nodes','3'],['--send-cores','0'],['--send-cores','56'],
    ['--master-port','29501'],['--reference-iters','0'],['--reference-iters','1001'],
    ['--skip-build','--rebuild-shmem']])
def test_bad_configuration_rejected_before_build(extra):
    with pytest.raises(SystemExit):launch.parse_args(argv(*extra))


def test_only_shmem_preparation_and_explicit_rebuild():
    args=launch.parse_args(argv());commands=launch.command_plan(args,Path('/python'))
    assert commands[0]==['git','submodule','update','--init','--recursive','--depth','1','--','third-party/shmem']
    assert not any('catlass' in str(part).lower() for command in commands for part in command)
    assert not any('--rebuild' in command for command in commands)
    args.rebuild_shmem=True;assert '--rebuild' in launch.command_plan(args,Path('/python'))[1]
    args.rebuild_shmem=False;args.skip_build=True
    commands=launch.command_plan(args,Path('/python'))
    assert len(commands)==2 and all('pip' not in command for command in commands)


def test_cpu_plan_does_not_query_device_or_create_paths(tmp_path,monkeypatch,capsys):
    args=launch.parse_args(argv('--plan'))
    monkeypatch.setattr(launch,'ROOT',tmp_path)
    monkeypatch.setattr(launch,'parse_args',lambda _:args)
    monkeypatch.setattr(launch,'validate_test_arguments',lambda _:None)
    monkeypatch.setattr(launch.subprocess,'check_output',lambda *a,**k:pytest.fail('plan must be read-only'))
    assert launch.main([])==0
    result=json.loads(capsys.readouterr().out)
    assert result['mode']=='reference' and result['reference_source']==launch.REFERENCE_SOURCE
    assert result['launch'][1]=='tests/elastic/test_ep_low_latency.py'
    assert list(tmp_path.iterdir())==[]


def test_environment_and_paths_are_project_local(tmp_path,monkeypatch):
    monkeypatch.setattr(launch,'ROOT',tmp_path/'project with spaces');launch.ROOT.mkdir()
    monkeypatch.setenv('SHMEM_ROOT','/old');monkeypatch.setenv('CATLASS_ROOT','/old');monkeypatch.setenv('RANK','99')
    args=launch.parse_args(argv())
    env=launch.child_environment(args)
    assert env['DEEPEP_LL_REFERENCE_PROFILE']=='ON' and env['DEEPEP_BUILD_HT']=='OFF'
    assert env['DEEPEP_LL_SEND_CORES']=='30' and env['DEEPEP_LL_RECV_CORES']=='26'
    assert all(key not in env for key in ('SHMEM_ROOT','CATLASS_ROOT','RANK'))
    assert all(env[key]==DEVICES for key in launch.VISIBILITY)
    for key in ('TMPDIR','TMP','TEMP','SHMEM_LOG_PATH','ASCEND_WORK_PATH','PIP_CACHE_DIR'):
        assert Path(env[key]).is_relative_to(launch.ROOT)
    with pytest.raises(ValueError):launch.project_path('../escape')


@pytest.mark.parametrize('failure',('none','build','identity','busy','prepare','skip-build'))
def test_prepare_all_checks_before_launch_and_preserve_logs(tmp_path,monkeypatch,failure):
    root=tmp_path/'repo';root.mkdir();monkeypatch.setattr(launch,'ROOT',root)
    args=launch.parse_args(argv(*(['--prepare-only'] if failure=='prepare' else []),
                               *(['--skip-build'] if failure=='skip-build' else [])))
    monkeypatch.setattr(launch,'parse_args',lambda _:args)
    monkeypatch.setattr(launch,'validate_test_arguments',lambda _:None)
    monkeypatch.setattr(launch.sys,'platform','linux')
    monkeypatch.setenv('ASCEND_HOME_PATH','/existing/cann')
    if failure=='skip-build':
        path=root/'.venv-ll-reference/bin/python';path.parent.mkdir(parents=True);path.write_text('fixture')
    def git(command,**kwargs):
        if command[1]=='branch':return launch.BRANCH+'\n'
        if command[1]=='status':return ''
        if command[1]=='rev-parse':return 'c'*40
        pytest.fail(str(command))
    monkeypatch.setattr(launch.subprocess,'check_output',git)
    calls=[]
    def logged(command,env,path):
        command=list(map(str,command));calls.append(command)
        Path(path).write_text('fixture log',encoding='utf-8')
        assert env['DEEPEP_LL_BACKEND']=='mte'
        if 'scripts/build_shmem.py' in command and failure=='build':raise RuntimeError('build failed')
        if '--identity-output' in command:
            if failure=='identity':raise RuntimeError('identity failed')
            Path(command[-1]).write_text(json.dumps(dict(sdk_build=dict(target='Ascend950',rdma_backend='HNS_1825',hns_1825_pi_on_chip=0),
                sdk_manifest_sha256='a'*64,mode='reference')))
    monkeypatch.setattr(launch,'run_logged',logged)
    def idle(_):
        calls.append(['idle'])
        if failure=='busy':raise RuntimeError('busy')
        return 'all16 idle'
    monkeypatch.setattr(launch,'require_idle',idle)
    def ready(*args):calls.append(['ready']);return {'all_prepared':True}
    monkeypatch.setitem(sys.modules,'target_ready',SimpleNamespace(synchronize=ready))
    if failure in ('build','identity','busy'):
        with pytest.raises(RuntimeError):launch.main([])
    else:assert launch.main([])==0
    tested=[command for command in calls if '-I' in command]
    if failure in ('build','identity','busy','prepare'):assert not tested
    else:
        assert len(tested)==1
        assert calls.index(['idle'])<calls.index(['ready'])<calls.index(tested[0])
        assert any(str(command[1]).endswith('analyze.py') for command in calls if len(command)>1)
    assert not (root/'.tmp/ll-reference-one-click.lock').exists()
    assert list((root/'.logs').rglob('*.log'))


def test_identity_retains_local_record_and_normalizes_only_mte_firmware_variant(tmp_path):
    path=tmp_path/'identity.json';value=dict(sdk_manifest_sha256='a',sdk_build=dict(
        cann_version='9.2.0',cann_timestamp='stamp',target='Ascend950',source_patches=[],rdma_backend='HNS_1825',hns_1825_pi_on_chip=1))
    path.write_text(json.dumps(value));before=path.read_bytes();actual=launch.read_identity(path)
    assert actual['sdk_build']=={key:item for key,item in value['sdk_build'].items() if key!='hns_1825_pi_on_chip'}
    assert 'sdk_manifest_sha256' not in actual and path.read_bytes()==before


def test_busy_device_check_never_kills(monkeypatch):
    monkeypatch.setattr(launch.subprocess,'check_output',lambda *a,**k:'No running processes found in NPU 0')
    with pytest.raises(RuntimeError,match='no launch'):launch.require_idle(DEVICES)


def test_project_lock_refuses_existing_owner(tmp_path,monkeypatch):
    monkeypatch.setattr(launch,'ROOT',tmp_path)
    with launch.project_lock():
        with pytest.raises(FileExistsError):
            with launch.project_lock():pytest.fail('must not acquire a live project lock')
    assert not (tmp_path/'.tmp/ll-reference-one-click.lock').exists()


@pytest.mark.parametrize('fault',(None,'source','sdk_revision','sdk_build','sdk_hash','cann_version','cann_timestamp','missing_build'))
def test_real_wheel_input_identity_includes_current_cann(fault):
    sdk=dict(revision='locked',build=dict(cann_version='9.2.0',cann_timestamp='stamp',target='Ascend950',source_patches=[],
                                       rdma_backend='HNS_1825',hns_1825_pi_on_chip=0))
    hashes={'lib/libshmem.so':'h','src/device/header.hpp':'d'}
    info=dict(source_sha256='source',shmem=dict(revision='locked',build=dict(sdk['build']),sha256=dict(hashes)),
              cann=dict(version='9.2.0',timestamp='stamp'))
    if fault=='source':info['source_sha256']='different'
    elif fault=='sdk_revision':info['shmem']['revision']='different'
    elif fault=='sdk_build':info['shmem']['build']['hns_1825_pi_on_chip']=1
    elif fault=='sdk_hash':info['shmem']['sha256']['src/device/header.hpp']='changed'
    elif fault=='cann_version':info['cann']['version']='9.3.0'
    elif fault=='cann_timestamp':info['cann']['timestamp']='new'
    elif fault=='missing_build':info['shmem'].pop('build')
    if fault is None:launch.validate_wheel_inputs(info,sdk,'source',hashes,dict(Version='9.2.0',timestamp='stamp'))
    else:
        with pytest.raises(RuntimeError):launch.validate_wheel_inputs(info,sdk,'source',hashes,dict(Version='9.2.0',timestamp='stamp'))
