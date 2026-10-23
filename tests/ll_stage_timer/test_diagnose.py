"""CPU-only checks of staged launch ordering, isolation and transfer indexes."""
import builtins
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
    assert result['transfer_mode'] == 'summary'
    off, *profiles = result['stages']
    assert '--ll-stage-timer' not in off['run'] and '--skip-perf-test' not in off['run']
    assert off['run'][-2:] == ['--perf-tail', '50']
    assert off['timer'] is False and off['profile'] == 'sparse'
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
                          '--profiles', 'sparse', '--bs', '1', '8', '32']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['matrix']['B'] == [1, 8, 32]
    assert [s['name'] for s in result['stages']] == ['off', 'sparse']
    assert result['transfer_mode'] == 'summary'


def test_archive_mode_is_rejected_before_planning(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        plan(tmp_path, '--transfer-mode', 'archive')
    assert error.value.code == 2
    assert "invalid choice: 'archive'" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('profile', ['full', 'overview', 'prepare', 'prefix', 'send', 'recv'])
def test_legacy_profile_capture_is_rejected_without_creating_files(tmp_path, profile, capsys):
    with pytest.raises(SystemExit) as error:
        plan(tmp_path, '--profiles', profile)
    assert error.value.code == 2
    assert 'invalid choice' in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_legacy_execution_plan_rejected_before_sdk_or_device_work(tmp_path, monkeypatch):
    result = plan(tmp_path)
    result['stages'][1]['profile'] = result['stages'][1]['name'] = 'prefix'
    def forbid(*args, **kwargs):
        raise AssertionError('Legacy plan must fail before work starts')
    monkeypatch.setattr(diagnose, '_validate_launcher_before_build', forbid)
    monkeypatch.setattr(diagnose, '_command', forbid)
    with pytest.raises(ValueError, match='offline-analysis only'):
        diagnose.execute(result)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('arguments, message', [
    (['--run-name', '../escape'], 'directory name'),
    (['--output-root', '../outside'], 'inside this checkout'),
    (['--output-root', 'tests/ll_stage_timer/results'], 'source-fingerprint trees'),
    (['--output-root', 'experiments/megamoe/results'], 'source-fingerprint trees'),
    (['--bs', '1', '64'], '1..32'),
    (['--bs', '1', '1'], 'distinct'),
    (['--m', '16'], 'max'),
    (['--profiles', 'sparse', 'sparse'], 'Duplicate'),
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
    result = plan(tmp_path, '--profiles', 'sparse')
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
    result = plan(tmp_path, '--profiles', 'sparse', '--skip-build', '--wheel-map', str(mapping))
    with pytest.raises(ValueError, match='missing sparse'):
        diagnose._wheel_map(result)
    mapping.write_text(json.dumps(dict(off='candidate.whl', sparse='candidate.whl')))
    assert set(diagnose._wheel_map(result)) == {'off', 'sparse'}
    assert 'wrong timer profile' in diagnose._verify_code(result['stages'][1], 'branch')


@pytest.mark.parametrize('transfer_mode', [None, 'summary'])
@pytest.mark.parametrize('node', [0, 1])
def test_execute_build_install_verify_run_analyze_order_without_devices(tmp_path, monkeypatch, transfer_mode, node):
    result = plan(tmp_path, '--profiles', 'sparse', '--bs', '1', '32',
                  '--num-nodes', '2', '--node-rank', str(node))
    if transfer_mode is None:
        result.pop('transfer_mode')
    else:
        result['transfer_mode'] = transfer_mode
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
    exports = []
    def export_once(path, transfer_mode='summary', *, receipt=None):
        exports.append((path, transfer_mode))
        if receipt is not None:
            receipt.update(new_count=1, issued=1, remaining=9)
        target = path / 'SEND_FIRST.txt'
        target.write_text('One bounded initial-analysis report')
        return target
    monkeypatch.setattr(diagnose, 'write_send_first', export_once)
    receipt = {}
    output = diagnose.execute(result, receipt=receipt)
    assert output == Path(result['suite']) / 'SEND_FIRST.txt'
    assert seen == ['venv.log', 'off-build.log', 'sparse-build.log',
                    'off-install.log', 'off-identity.log', 'sparse-install.log', 'sparse-identity.log',
                    'off-run-install.log', 'off-run-identity.log', 'off-run.log',
                    'sparse-run-install.log', 'sparse-run-identity.log', 'sparse-run.log'] + (['sparse-analyze.log'] if node == 0 else [])
    mapping = json.loads((Path(result['node_dir']) / 'wheel-map.json').read_text())
    assert mapping == dict(off='wheels/off/candidate.whl', sparse='wheels/sparse/candidate.whl')
    assert output.is_file() is (node == 0)
    assert exports == ([(Path(result['suite']), transfer_mode or 'summary')] if node == 0 else [])
    assert receipt == (dict(new_count=1, issued=1, remaining=9) if node == 0 else {})
    with pytest.raises(ValueError, match='already exists'):
        diagnose.execute(result)


def test_missing_sdk_fails_before_creating_run(tmp_path):
    result = plan(tmp_path)
    with pytest.raises(ValueError, match='SDK'):
        diagnose.execute(result)
    assert not Path(result['suite']).exists()


@pytest.mark.parametrize('operation', ['execute', 'write', 'analyze', 'plan'])
def test_internal_archive_request_fails_before_mutations_or_commands(tmp_path, monkeypatch, operation):
    result = plan(tmp_path)
    result['transfer_mode'] = 'archive'
    def forbid(*args, **kwargs):
        raise AssertionError('Archive rejection must precede imports, writes, builds and device work')
    monkeypatch.setattr(diagnose, '_transfer_module', forbid)
    monkeypatch.setattr(diagnose, '_write_performance_compare', forbid)
    monkeypatch.setattr(diagnose, '_command', forbid)
    monkeypatch.setattr(diagnose.subprocess, 'run', forbid)
    monkeypatch.setattr(Path, 'mkdir', forbid)
    with pytest.raises(ValueError, match='10-transfer budget'):
        if operation == 'execute':
            diagnose.execute(result)
        elif operation == 'write':
            diagnose.write_send_first(Path(result['suite']), transfer_mode='archive')
        elif operation == 'analyze':
            diagnose.analyze_existing(Path(result['suite']), root=tmp_path, transfer_mode='archive')
        else:
            args, forwarded = diagnose.parser().parse_known_args(
                ['--run-name', 'batch1', '--devices', DEVICES])
            args.transfer_mode = 'archive'
            diagnose.make_plan(args, forwarded, root=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_default_send_first_never_loads_full_archive(tmp_path, monkeypatch):
    suite = tmp_path / 'suite'
    suite.mkdir()
    loaded = []
    def load(name):
        loaded.append(name)
        assert name == 'triage_report', 'Default transfer must not load/compress the full archive'
        def export_report(actual_suite, *, receipt=None):
            assert actual_suite == suite
            if receipt is not None:
                receipt.update(new_count=1, issued=1, remaining=9)
            destination = actual_suite / 'SEND_FIRST.txt'
            destination.write_text('Bounded initial analysis\n', encoding='utf-8')
            return destination
        return SimpleNamespace(export_report=export_report)
    monkeypatch.setattr(diagnose, '_transfer_module', load)
    receipt = {}
    output = diagnose.write_send_first(suite, receipt=receipt)
    assert output == suite / 'SEND_FIRST.txt'
    assert loaded == ['triage_report']
    assert len(output.read_bytes()) <= diagnose.LIMIT
    assert not (suite / 'transfer').exists()
    assert receipt == dict(new_count=1, issued=1, remaining=9)


@pytest.mark.parametrize('saved_mode', [None, 'archive'])
def test_analyze_existing_old_plans_default_to_summary_without_device_or_archive_work(
        tmp_path, monkeypatch, saved_mode):
    suite = tmp_path / 'suite'
    (suite / 'node_000').mkdir(parents=True)
    saved = dict(stages=[dict(name='prefix', timer=True)])
    if saved_mode is not None:
        saved['transfer_mode'] = saved_mode
    (suite / 'node_000/plan.json').write_text(json.dumps(saved))
    (suite / 'analysis/prefix').mkdir(parents=True)
    (suite / 'runs/prefix').mkdir(parents=True)
    archive = suite / 'transfer' / 'old_bundle' / 'suite_001_part_001_of_887.txt'
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b'Existing archive stays untouched and readable')
    seen = []
    def write(path, transfer_mode='summary', *, receipt=None):
        seen.append((path, transfer_mode))
        if receipt is not None:
            receipt.update(new_count=0, issued=2, remaining=8)
        destination = path / 'SEND_FIRST.txt'
        destination.write_text('initial analysis')
        return destination
    def forbid(*args, **kwargs):
        raise AssertionError('Existing analysis must not invoke build, sampling, or reparsing')
    monkeypatch.setattr(diagnose, 'write_send_first', write)
    monkeypatch.setattr(diagnose.subprocess, 'run', forbid)
    monkeypatch.setattr(diagnose, 'execute', forbid)
    receipt = {}
    output = diagnose.analyze_existing(suite, root=tmp_path, receipt=receipt)
    assert output == suite / 'SEND_FIRST.txt'
    assert seen == [(suite, 'summary')]
    assert archive.read_bytes() == b'Existing archive stays untouched and readable'
    assert receipt == dict(new_count=0, issued=2, remaining=8)


def test_analyze_existing_cli_only_allows_summary(monkeypatch, capsys):
    seen = []
    def analyze(suite, **kwargs):
        seen.append((suite, kwargs['transfer_mode']))
        return suite / 'SEND_FIRST.txt'
    monkeypatch.setattr(diagnose, 'analyze_existing', analyze)
    assert diagnose.main(['--analyze-existing', 'diagnostic-runs/existing']) == 0
    assert diagnose.main(['--analyze-existing', 'diagnostic-runs/existing',
                          '--transfer-mode', 'summary']) == 0
    assert seen == [(Path('diagnostic-runs/existing'), 'summary'),
                    (Path('diagnostic-runs/existing'), 'summary')]
    with pytest.raises(SystemExit) as error:
        diagnose.main(['--analyze-existing', 'diagnostic-runs/existing',
                       '--transfer-mode', 'archive'])
    assert error.value.code == 2
    seen.clear()
    assert diagnose.main(['--analyze-existing', 'diagnostic-runs/existing', '--plan']) == 0
    assert seen == []
    result = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert result['transfer_mode'] == 'summary' and result['mutation'] is False


@pytest.mark.parametrize('mode', ['execute', 'analyze'])
@pytest.mark.parametrize('new_count', [0, 1])
def test_cli_budget_receipt_only_prints_new_summary_path(monkeypatch, capsys, mode, new_count):
    target = Path('diagnostic-runs/known/SEND_FIRST.txt')
    def run(*args, receipt=None, **kwargs):
        assert receipt == {}
        receipt.update(new_count=new_count, issued=4, remaining=6)
        return target
    if mode == 'execute':
        monkeypatch.setattr(diagnose, 'make_plan', lambda *args: {})
        monkeypatch.setattr(diagnose, 'execute', run)
        arguments = ['--run-name', 'known']
    else:
        monkeypatch.setattr(diagnose, 'analyze_existing', run)
        arguments = ['--analyze-existing', 'diagnostic-runs/known']
    assert diagnose.main(arguments) == 0
    output = capsys.readouterr().out
    assert '4/10 issued; 6 remaining' in output
    assert (str(target) in output) is bool(new_count)
    assert ('do not resend' in output) is (not new_count)


def test_nonzero_node_cli_does_not_request_summary_transfer(monkeypatch, capsys):
    monkeypatch.setattr(diagnose, 'make_plan', lambda *args: {'node': 1})
    monkeypatch.setattr(diagnose, 'execute', lambda plan, receipt=None:
                        Path('diagnostic-runs/known/SEND_FIRST.txt'))
    assert diagnose.main(['--run-name', 'known']) == 0
    output = capsys.readouterr().out
    assert 'No summary issued on this node' in output
    assert 'SEND_FIRST.txt' not in output and '/10 issued' not in output


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
    diagnose._write_performance_compare(suite)
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
    result=plan(tmp_path,'--profiles','sparse')
    package=tmp_path/'installed/deep_ep'
    package.mkdir(parents=True)
    info = dict(source_sha256='matching-source', ll_stage_timer=True, ll_timer_profile='sparse',
                ll_send_cores=30, ll_recv_cores=26)
    (package/'_build_info.json').write_text(json.dumps(info))
    from tests.elastic import test_ep_low_latency as st
    points=runpy.run_path(str(diagnose.ROOT/'deep_ep/utils/ll_stage_timer.py'))['SPARSE_POINTS']
    layout=dict(schema=15,slots=8,fields=8,recording='sparse-milestones',
                points=list(points),timer_profile='sparse',timer_profile_id=6)
    native=SimpleNamespace(low_latency_implementation='candidate',low_latency_stage_timer_enabled=True,
                           low_latency_core_protocol=19030,
                           low_latency_stage_timer_profile='sparse',low_latency_stage_timer_profile_id=6,
                           low_latency_stage_timer_layout=lambda:dict(layout))
    def no_device_query(*args, **kwargs):
        raise AssertionError('Host identity verification must not query or initialize a device')
    native.low_latency_core_config = no_device_query
    real_import = builtins.__import__
    def host_only_import(name, *args, **kwargs):
        if name.split('.')[0] in ('torch', 'torch_npu'):
            raise AssertionError('Host identity verification must not import an NPU framework')
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', host_only_import)
    monkeypatch.setitem(sys.modules,'deep_ep._native',SimpleNamespace(require_native=lambda _:native))
    def fake_run_path(path):
        if path=='tests/elastic/test_dispatch.py':
            return dict(_import_installed_deep_ep=lambda:SimpleNamespace(__file__=str(package/'__init__.py')))
        if path=='tests/elastic/test_ep_low_latency.py':
            return dict(validate_sparse_timer_layout=st.validate_sparse_timer_layout)
        if Path(path)==st.PROJECT_ROOT/'deep_ep/utils/ll_stage_timer.py':
            return dict(SPARSE_POINTS=points)
        if path=='build_support.py':
            return dict(source_fingerprint=lambda:'matching-source')
        raise AssertionError(path)
    monkeypatch.setattr(runpy,'run_path',fake_run_path)
    code=diagnose._verify_code(result['stages'][1],'candidate', dict(send=30, recv=26))
    exec(code,{})
    native.low_latency_stage_timer_layout=lambda:dict(timer_profile='recv',timer_profile_id=5)
    with pytest.raises(RuntimeError,match='layout mismatch'):
        exec(code,{})
    native.low_latency_stage_timer_layout=lambda:dict(layout)
    (package/'_build_info.json').write_text(json.dumps(dict(info, source_sha256='old-source')))
    with pytest.raises(RuntimeError,match='source fingerprint differs'):
        exec(code,{})
    (package/'_build_info.json').write_text(json.dumps(dict(info, ll_send_cores=28, ll_recv_cores=28)))
    with pytest.raises(RuntimeError,match='S/R differs'):
        exec(code,{})
    (package/'_build_info.json').write_text(json.dumps(dict(info, ll_stage_timer=False)))
    with pytest.raises(RuntimeError,match='build metadata'):
        exec(code,{})


def test_default_output_migrates_outside_sources_in_directory_with_spaces(tmp_path):
    root=tmp_path/'Project With Spaces'
    result=plan(root,'--profiles','sparse')
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


def _mock_build_environment(root, monkeypatch, *, fail_log=None):
    sdk = root / 'third-party/shmem/install/shmem'
    (sdk / 'include/host').mkdir(parents=True)
    (sdk / 'include/host/shmem_host_def.h').write_text('header')
    (sdk / 'deepep-sdk.json').write_text('{}')
    (root / 'tests/ll_stage_timer').mkdir(parents=True)
    (root / 'tests/ll_stage_timer/branch.json').write_text('{"implementation":"candidate"}')
    monkeypatch.setattr(diagnose.importlib.util, 'find_spec', lambda name: object())
    seen = []
    def command(argv, actual_plan, env, log):
        seen.append(Path(log).name)
        if Path(log).name == fail_log:
            raise RuntimeError('intentional build or identity failure')
        if '-m' in argv and 'build' in argv:
            out = Path(argv[argv.index('--outdir') + 1])
            (out / 'candidate.whl').write_bytes(b'wheel')
        Path(log).write_text('mocked execution')
    monkeypatch.setattr(diagnose, '_command', command)
    return seen


@pytest.mark.parametrize('skip_build', [False, True])
@pytest.mark.parametrize('fail_log', ['off-identity.log', 'sparse-identity.log'])
def test_every_wheel_identity_must_pass_before_any_device_launch(tmp_path, monkeypatch, skip_build, fail_log):
    arguments = []
    if skip_build:
        (tmp_path / 'off.whl').write_bytes(b'off')
        (tmp_path / 'sparse.whl').write_bytes(b'sparse')
        (tmp_path / 'input-map.json').write_text(json.dumps(dict(off='off.whl', sparse='sparse.whl')))
        arguments = ['--skip-build', '--wheel-map', 'input-map.json']
    result = plan(tmp_path, *arguments)
    seen = _mock_build_environment(tmp_path, monkeypatch, fail_log=fail_log)
    monkeypatch.setattr(diagnose, '_validate_launcher_before_build', lambda plan: None)
    with pytest.raises(RuntimeError, match='intentional'):
        diagnose.execute(result)
    assert not any('-run' in name or '-analyze' in name for name in seen)
    assert not (Path(result['node_dir']) / 'wheel-map.json').exists()
    assert not (Path(result['suite']) / 'SEND_FIRST.txt').exists()
    assert not (Path(result['suite']) / 'triage').exists()
    assert ('off-build.log' in seen) is (not skip_build)


def test_second_variant_build_failure_cannot_run_first_variant(tmp_path, monkeypatch):
    result = plan(tmp_path)
    seen = _mock_build_environment(tmp_path, monkeypatch, fail_log='sparse-build.log')
    monkeypatch.setattr(diagnose, '_validate_launcher_before_build', lambda plan: None)
    with pytest.raises(RuntimeError, match='intentional'):
        diagnose.execute(result)
    assert seen == ['venv.log', 'off-build.log', 'sparse-build.log']
    assert not (Path(result['node_dir']) / 'wheel-map.json').exists()


def test_build_only_needs_no_devices_or_endpoints_and_preserves_visibility(tmp_path, monkeypatch):
    monkeypatch.setenv('DEEPEP_LL_SEND_CORES', '30')
    monkeypatch.setenv('DEEPEP_LL_RECV_CORES', '26')
    monkeypatch.setenv('ASCEND_RT_VISIBLE_DEVICES', '0,1')
    monkeypatch.delenv('PROBE_VISIBLE_DEVICES', raising=False)
    args, forwarded = diagnose.parser().parse_known_args(['--run-name', 'compile', '--build-only'])
    result = diagnose.make_plan(args, forwarded, root=tmp_path)
    assert result['build_only'] is True and result['device_visibility'] is None
    assert result['core_split'] == dict(send=30, recv=26)
    assert all(stage['run'] is None and stage['analyze'] is None for stage in result['stages'])
    seen = _mock_build_environment(tmp_path, monkeypatch)
    def forbid(*args, **kwargs):
        raise AssertionError('Build-only must not validate launcher topology, sample, analyze, or issue a send budget')
    monkeypatch.setattr(diagnose, '_validate_launcher_before_build', forbid)
    monkeypatch.setattr(diagnose, 'write_send_first', forbid)
    receipt = {}
    result_path = diagnose.execute(result, receipt=receipt)
    assert result_path == Path(result['node_dir']) / 'wheel-map.json'
    assert receipt == {}
    assert seen == ['venv.log', 'off-build.log', 'sparse-build.log',
                    'off-install.log', 'off-identity.log', 'sparse-install.log', 'sparse-identity.log']
    assert json.loads(result_path.read_text()) == dict(off='wheels/off/candidate.whl', sparse='wheels/sparse/candidate.whl')
    assert not (Path(result['suite']) / 'runs').exists()
    assert not (Path(result['suite']) / 'triage').exists()
    # Compiling and constant-layout imports inherit any explicit device limit.
    # They neither query devices nor synthesize a default visible device set.
    env = diagnose._environment(result, result['stages'][0])
    assert env['ASCEND_RT_VISIBLE_DEVICES'] == '0,1'
    assert 'PROBE_VISIBLE_DEVICES' not in env
    assert env['DEEPEP_LL_SEND_CORES'] == '30' and env['DEEPEP_LL_RECV_CORES'] == '26'


def test_build_only_plan_cli_has_no_mutation_or_device_commands(monkeypatch, capsys):
    def forbid(*args, **kwargs):
        raise AssertionError('Pure build planning must not execute')
    monkeypatch.setattr(diagnose, 'execute', forbid)
    monkeypatch.setattr(subprocess, 'run', forbid)
    monkeypatch.setattr(Path, 'mkdir', forbid)
    assert diagnose.main(['--build-only', '--plan', '--run-name', 'compile']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['build_only'] is True and result['device_visibility'] is None
    assert all(stage['run'] is None and stage['analyze'] is None for stage in result['stages'])


def test_normal_run_still_requires_explicit_devices(tmp_path):
    args, forwarded = diagnose.parser().parse_known_args(['--run-name', 'run'])
    with pytest.raises(ValueError, match='explicit launcher --devices'):
        diagnose.make_plan(args, forwarded, root=tmp_path)


def test_relative_wheel_map_survives_checkout_move_and_new_run(tmp_path, monkeypatch):
    root = tmp_path / 'Original Project'
    root.mkdir()
    args, forwarded = diagnose.parser().parse_known_args(['--run-name', 'compile', '--build-only'])
    result = diagnose.make_plan(args, forwarded, root=root)
    _mock_build_environment(root, monkeypatch)
    mapping = diagnose.execute(result)
    map_relative = mapping.relative_to(root)
    destination = tmp_path / 'Moved Project'
    root.rename(destination)
    reused = plan(destination, '--skip-build', '--wheel-map', str(map_relative))
    wheels = diagnose._wheel_map(reused)
    assert set(wheels) == {'off', 'sparse'}
    assert all(Path(wheel).is_relative_to(destination) and Path(wheel).is_file() for wheel in wheels.values())
    assert all(not Path(value).is_absolute() for value in json.loads((destination / map_relative).read_text()).values())
    assert Path(reused['suite']).name == 'batch1'


def test_build_only_cli_prints_map_without_summary_or_send_receipt(monkeypatch, capsys):
    target = Path('diagnostic-runs/compile/node_000/wheel-map.json')
    monkeypatch.setattr(diagnose, 'make_plan', lambda *args: dict(build_only=True))
    monkeypatch.setattr(diagnose, 'execute', lambda plan, receipt=None: target)
    assert diagnose.main(['--build-only', '--run-name', 'compile']) == 0
    output = capsys.readouterr().out
    assert str(target) in output and '--skip-build --wheel-map' in output
    assert 'no device sampling was run' in output
    assert 'SEND_FIRST.txt' not in output and '/10 issued' not in output


def test_build_only_cannot_analyze_existing_or_skip_its_build(tmp_path, capsys):
    with pytest.raises(ValueError, match='cannot be combined'):
        plan(tmp_path, '--build-only', '--skip-build', '--wheel-map', 'old-map.json')
    with pytest.raises(SystemExit) as error:
        diagnose.main(['--build-only', '--analyze-existing', 'diagnostic-runs/existing'])
    assert error.value.code == 1
    assert '--build-only cannot be combined' in capsys.readouterr().err
