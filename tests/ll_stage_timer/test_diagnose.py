"""CPU-only checks of staged launch ordering, isolation and transfer indexes."""
import importlib.util
import json
from pathlib import Path
import subprocess
import runpy
import sys
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location('ll_staged_diagnose', Path(__file__).with_name('diagnose.py'))
diagnose = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnose)
DEVICES = ','.join(map(str, range(16)))


def plan(tmp_path, *arguments):
    args, forwarded = diagnose.parser().parse_known_args(
        ['--run-name', 'batch1', '--devices', DEVICES, *arguments])
    return diagnose.make_plan(args, forwarded, root=tmp_path, python='/existing/env/python')


def test_default_matrix_off_first_and_disjoint_profiles(tmp_path):
    result = plan(tmp_path)
    assert result['matrix'] == dict(B=[1, 2, 4, 8, 16, 24, 32], M=32, H=3584, K=8, E=384,
                                    dtype='fp8', cpu_sync=0)
    assert [stage['name'] for stage in result['stages']] == ['off', *diagnose.PROFILES]
    off, *profiles = result['stages']
    assert '--ll-stage-timer' not in off['run'] and '--skip-perf-test' not in off['run']
    assert off['run'][-2:] == ['--perf-tail', '50']
    assert off['timer'] is False and off['profile'] == 'full'
    for stage in profiles:
        assert '--ll-stage-timer' in stage['run'] and '--skip-perf-test' in stage['run']
        assert '--ll-profile-capture' in stage['run']
        assert stage['run'][-1] == '--skip-perf-test'
        assert '--ll-timer-mode' in stage['run']
        assert stage['analyze'][2] == stage['results']
    assert len({s['results'] for s in result['stages']}) == len(result['stages'])
    assert all('--no-isolation' in s['build'] for s in result['stages'])
    assert all('--no-index' in s['install'] and '--no-deps' in s['install'] for s in result['stages'])
    assert '--system-site-packages' in result['create_venv']
    assert list(tmp_path.iterdir()) == []


def test_plan_cli_is_nonmutating(tmp_path, monkeypatch, capsys):
    def forbid(*args, **kwargs):
        raise AssertionError('Plan attempted execution or filesystem creation')
    monkeypatch.setattr(diagnose, 'execute', forbid)
    monkeypatch.setattr(subprocess, 'run', forbid)
    monkeypatch.setattr(Path, 'mkdir', forbid)
    assert diagnose.main(['--plan', '--run-name', 'pure-plan', '--devices', DEVICES,
                          '--profiles', 'prefix', '--bs', '1', '8', '32']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['matrix']['B'] == [1, 8, 32]
    assert [s['name'] for s in result['stages']] == ['off', 'prefix']


@pytest.mark.parametrize('arguments, message', [
    (['--run-name', '../escape'], 'directory name'),
    (['--output-root', '../outside'], 'inside this checkout'),
    (['--output-root', 'tests/ll_stage_timer/results'], 'source-fingerprint trees'),
    (['--output-root', 'experiments/megamoe/results'], 'source-fingerprint trees'),
    (['--bs', '1', '64'], '1..32'),
    (['--bs', '1', '1'], 'distinct'),
    (['--m', '16'], 'max'),
    (['--profiles', 'prefix', 'prefix'], 'Duplicate'),
    (['--skip-build'], 'together'),
    (['--skip-check'], 'controlled'),
    (['--ll-stage-timer'], 'controlled'),
    (['--config', 'hidden-flags.json'], 'controlled'),
    (['--perf-tail', '101'], 'perf-tail'),
])
def test_invalid_or_conflicting_inputs_rejected(tmp_path, arguments, message):
    with pytest.raises(ValueError, match=message):
        plan(tmp_path, *arguments)


def test_launcher_topology_forwarded_and_node_storage_isolated(tmp_path):
    result = plan(tmp_path, '--num-nodes', '2', '--node-rank', '1',
                  '--master-addr', '10.0.0.1', '--master-port', '23456',
                  '--shmem-ip-port', 'tcp://10.0.0.1:23457', '--defer-analysis')
    assert result['nodes'] == 2 and result['node'] == 1 and result['defer_analysis'] is True
    assert result['node_dir'].endswith('node_001')
    assert all('--master-addr' in stage['run'] for stage in result['stages'])
    assert all('10.0.0.1' in stage['run'] for stage in result['stages'])


def test_environment_is_offline_and_profile_specific(tmp_path, monkeypatch):
    monkeypatch.setenv('SHMEM_ROOT', '/unrelated/sdk')
    monkeypatch.setenv('CATLASS_ROOT', '/unrelated/catlass')
    result = plan(tmp_path, '--profiles', 'prefix')
    for stage in result['stages']:
        env = diagnose._environment(result, stage)
        assert env['PIP_NO_INDEX'] == '1'
        assert env['DEEPEP_LL_TIMER_PROFILE'] == stage['profile']
        assert env['DEEPEP_LL_STAGE_TIMER'] == ('ON' if stage['timer'] else 'OFF')
        assert env['DEEPEP_LL_EXPECT_TIMER_PROFILE'] == stage['profile']
        assert 'SHMEM_ROOT' not in env and 'CATLASS_ROOT' not in env
        for name in ('TMPDIR', 'PIP_CACHE_DIR', 'SHMEM_LOG_PATH'):
            assert Path(result['node_dir']) in Path(env[name]).parents
        assert env['PROBE_VISIBLE_DEVICES'] == DEVICES
    assert diagnose.os.environ['SHMEM_ROOT'] == '/unrelated/sdk'


def test_skip_build_wheel_map_requires_all_profiles(tmp_path):
    wheel = tmp_path/'candidate.whl'
    wheel.write_bytes(b'wheel')
    mapping = tmp_path/'wheels.json'
    mapping.write_text(json.dumps(dict(off='candidate.whl')))
    result = plan(tmp_path, '--profiles', 'prefix', '--skip-build', '--wheel-map', str(mapping))
    with pytest.raises(ValueError, match='missing prefix'):
        diagnose._wheel_map(result)
    mapping.write_text(json.dumps(dict(off='candidate.whl', prefix='candidate.whl')))
    assert set(diagnose._wheel_map(result)) == {'off', 'prefix'}
    assert 'wrong timer profile' in diagnose._verify_code(result['stages'][1], 'branch')


def test_execute_build_install_verify_run_analyze_order_without_devices(tmp_path, monkeypatch):
    result = plan(tmp_path, '--profiles', 'prefix', '--bs', '1', '32')
    root = tmp_path
    sdk = root/'third-party/shmem/install/shmem'
    (sdk/'include/host').mkdir(parents=True)
    (sdk/'include/host/shmem_host_def.h').write_text('header')
    (sdk/'deepep-sdk.json').write_text('{}')
    (root/'tests/ll_stage_timer').mkdir(parents=True)
    (root/'tests/ll_stage_timer/branch.json').write_text('{"implementation":"candidate"}')
    monkeypatch.setattr(diagnose.importlib.util, 'find_spec', lambda name: object())
    monkeypatch.setattr(diagnose, '_validate_launcher_before_build', lambda plan: None)
    seen = []
    def command(argv, actual_plan, env, log):
        seen.append(Path(log).name)
        if '-m' in argv and 'build' in argv:
            out = Path(argv[argv.index('--outdir')+1])
            (out/'candidate.whl').write_bytes(b'wheel')
        Path(log).write_text('mocked execution')
    monkeypatch.setattr(diagnose, '_command', command)
    diagnose.execute(result)
    assert seen == ['venv.log', 'off-build.log', 'off-install.log', 'off-identity.log',
                    'off-run.log', 'prefix-build.log', 'prefix-install.log',
                    'prefix-identity.log', 'prefix-run.log', 'prefix-analyze.log']
    assert (Path(result['suite'])/'SEND_FIRST.txt').is_file()
    with pytest.raises(ValueError, match='already exists'):
        diagnose.execute(result)


def test_missing_sdk_fails_before_creating_run(tmp_path):
    result = plan(tmp_path)
    with pytest.raises(ValueError, match='SDK'):
        diagnose.execute(result)
    assert not Path(result['suite']).exists()


def test_send_first_index_is_bounded_and_profile_paths_distinct(tmp_path):
    suite = tmp_path/'suite'
    for profile in ('overview', 'prefix', 'recv'):
        for case in range(140):
            directory = suite/'analysis'/profile/f'case_{case:05d}'/'transfer'
            directory.mkdir(parents=True)
            names = ['00_overview.md'] + [f'priority_{i:03d}_of_003.txt' for i in range(1, 4)]
            (directory/'manifest.json').write_text(json.dumps(dict(bundle_id=f'{profile}-{case}', send_first=names)))
    result = diagnose.write_send_first(suite)
    indexes = sorted(suite.glob('SEND_FIRST*.txt'))
    assert result.name == 'SEND_FIRST.txt' and len(indexes) > 1
    text = ''.join(p.read_text() for p in indexes)
    assert 'analysis/prefix/' in text and 'analysis/recv/' in text
    assert 'Do not add costs' in text
    assert all(len(p.read_bytes()) <= 60000 for p in indexes)


def test_analyze_existing_does_not_silently_skip_partial_profile(tmp_path):
    suite = tmp_path/'suite'
    (suite/'node_000').mkdir(parents=True)
    (suite/'node_000/plan.json').write_text(json.dumps(dict(
        matrix=dict(B=[1, 32]), stages=[dict(timer=True, name='prefix')])))
    partial = suite/'analysis/prefix/case_00000/transfer'
    partial.mkdir(parents=True)
    (partial/'manifest.json').write_text('{}')
    with pytest.raises(ValueError, match='incomplete analysis'):
        diagnose.analyze_existing(suite, root=tmp_path)


def test_performance_table_uses_full_tasks_and_matching_identity(tmp_path):
    suite=tmp_path/'suite'
    (suite/'runs/off').mkdir(parents=True)
    config=dict(B=1,M=32,H=3584,num_topk=8,num_experts=384,dtype='fp8')
    identity=dict(implementation='candidate',transport='mte',logical_group=4,active_aiv=56,
                  send_cores=28,recv_cores=28,core_protocol=19028,source_revision='commit',source_sha256='a'*64)
    (suite/'runs/off/summary.json').write_text(json.dumps(dict(results=[dict(
        config=config,route='normal',comparison=identity,dispatch=dict(max_rank_mean_kernel_us=10))])))
    directory=suite/'runs/prefix/case_00000/normal'
    directory.mkdir(parents=True)
    on=dict(status='available',config=config,route='normal',comparison=identity,diagnostic_iters=5,
            summary=dict(max_rank_mean_kernel_us=12))
    path=directory/'active_capture_dispatch.json'
    path.write_text(json.dumps(on))
    rows=diagnose._performance_rows(suite)
    assert rows[0]['ratio']==1.2 and rows[0]['delta_us']==2
    diagnose.write_send_first(suite)
    assert 'PERFORMANCE_COMPARE.txt' in (suite/'SEND_FIRST.txt').read_text()
    assert len((suite/'PERFORMANCE_COMPARE.txt').read_bytes())<=60000
    on['comparison']=dict(identity,send_cores=30,recv_cores=26)
    path.write_text(json.dumps(on))
    row=diagnose._performance_rows(suite)[0]
    assert row['status'].startswith('unavailable') and 'ratio' not in row
    on['comparison']=dict(identity,source_sha256='b'*64)
    path.write_text(json.dumps(on))
    row=diagnose._performance_rows(suite)[0]
    assert row['status'].startswith('unavailable') and 'ratio' not in row


def test_installed_wheel_source_and_native_layout_are_checked(tmp_path,monkeypatch):
    result=plan(tmp_path,'--profiles','prefix')
    package=tmp_path/'installed/deep_ep'
    package.mkdir(parents=True)
    (package/'_build_info.json').write_text(json.dumps(dict(source_sha256='matching-source')))
    native=SimpleNamespace(low_latency_implementation='candidate',low_latency_stage_timer_enabled=True,
                           low_latency_stage_timer_profile='prefix',low_latency_stage_timer_profile_id=3,
                           low_latency_stage_timer_layout=lambda:dict(timer_profile='prefix',timer_profile_id=3))
    monkeypatch.setitem(sys.modules,'deep_ep._native',SimpleNamespace(require_native=lambda _:native))
    def fake_run_path(path):
        if path=='tests/elastic/test_dispatch.py':
            return dict(_import_installed_deep_ep=lambda:SimpleNamespace(__file__=str(package/'__init__.py')))
        if path=='build_support.py':
            return dict(source_fingerprint=lambda:'matching-source')
        raise AssertionError(path)
    monkeypatch.setattr(runpy,'run_path',fake_run_path)
    code=diagnose._verify_code(result['stages'][1],'candidate')
    exec(code,{})
    native.low_latency_stage_timer_layout=lambda:dict(timer_profile='recv',timer_profile_id=5)
    with pytest.raises(RuntimeError,match='layout mismatch'):
        exec(code,{})
    native.low_latency_stage_timer_layout=lambda:dict(timer_profile='prefix',timer_profile_id=3)
    (package/'_build_info.json').write_text(json.dumps(dict(source_sha256='old-source')))
    with pytest.raises(RuntimeError,match='source fingerprint differs'):
        exec(code,{})


def test_default_output_migrates_outside_sources_in_directory_with_spaces(tmp_path):
    root=tmp_path/'Project With Spaces'
    result=plan(root,'--profiles','overview')
    assert Path(result['suite'])==root/'diagnostic-runs/batch1'
    assert Path(result['venv']).is_relative_to(root/'diagnostic-runs')
    assert list(tmp_path.iterdir())==[]


def test_endpoint_preflight_is_pure_and_requires_explicit_values(tmp_path,monkeypatch):
    from tests.elastic import test_ep_low_latency as st
    monkeypatch.setattr(diagnose.runpy,'run_path',lambda path:vars(st))
    for name in ('MASTER_ADDR','MASTER_PORT','DEEPEP_SHMEM_ENDPOINT'):
        monkeypatch.delenv(name,raising=False)
    with pytest.raises(ValueError,match='Explicit MASTER_ADDR'):
        diagnose._validate_launcher_before_build(plan(tmp_path))
    explicit=plan(tmp_path,'--master-addr','127.0.0.1','--master-port','23456','--shmem-port','23457')
    diagnose._validate_launcher_before_build(explicit)
    assert 'MASTER_ADDR' not in diagnose.os.environ
    assert 'DEEPEP_SHMEM_ENDPOINT' not in diagnose.os.environ
    assert list(tmp_path.iterdir())==[]
    monkeypatch.setenv('MASTER_ADDR','127.0.0.1')
    monkeypatch.setenv('MASTER_PORT','23456')
    monkeypatch.setenv('DEEPEP_SHMEM_ENDPOINT','tcp://127.0.0.1:23457')
    diagnose._validate_launcher_before_build(plan(tmp_path))


def test_missing_endpoint_fails_before_venv_and_build(tmp_path,monkeypatch):
    from tests.elastic import test_ep_low_latency as st
    monkeypatch.setattr(diagnose.runpy,'run_path',lambda path:vars(st))
    for name in ('MASTER_ADDR','MASTER_PORT','DEEPEP_SHMEM_ENDPOINT'):
        monkeypatch.delenv(name,raising=False)
    result=plan(tmp_path)
    sdk=tmp_path/'third-party/shmem/install/shmem'
    (sdk/'include/host').mkdir(parents=True)
    (sdk/'include/host/shmem_host_def.h').write_text('header')
    (sdk/'deepep-sdk.json').write_text('{}')
    def fail(*args,**kwargs):raise AssertionError('build/venv was invoked')
    monkeypatch.setattr(diagnose,'_command',fail)
    with pytest.raises(ValueError,match='Explicit MASTER_ADDR'):
        diagnose.execute(result)
    assert not Path(result['suite']).exists()
