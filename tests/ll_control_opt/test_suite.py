"""CPU tests for commit selection, cache safety and the user's ST metric."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import socket
import threading
import sys
import types
import zipfile

import pytest

_spec = importlib.util.spec_from_file_location('control_opt_suite_tests', Path(__file__).with_name('suite.py'))
suite = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(suite)
build = suite.build


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def repository(tmp_path):
    root = tmp_path / 'relocated project with spaces'
    root.mkdir()
    subprocess.run(['git', 'init', str(root)], check=True, capture_output=True)
    git(root, 'config', 'user.name', 'fixture')
    git(root, 'config', 'user.email', 'fixture@example.invalid')
    (root / '.gitignore').write_text('.tmp/\nresults/\nthird-party/\n')
    (root / 'dependencies.lock.json').write_text('{}')
    (root / 'kernel.h').write_text('baseline\n')
    git(root, 'add', '.')
    git(root, 'commit', '-qm', 'baseline')
    baseline = git(root, 'rev-parse', 'HEAD')
    (root / 'kernel.h').write_text('candidate\n')
    git(root, 'commit', '-qam', 'candidate')
    candidate = git(root, 'rev-parse', 'HEAD')
    manifest = root / '.tmp/commits.json'
    manifest.parent.mkdir()
    build.write_json(manifest, {'commits': [dict(label='baseline', revision=baseline), dict(label='candidate', revision=candidate)]})
    return root, baseline, candidate, manifest


def args(commits):
    return argparse.Namespace(commits=commits, send_cores=30, rounds=1, order='alternating', build_only=False, run_only=False,
                              devices=','.join(map(str, range(16))), run_prefix='b256-001',
                              output_dir=Path('results/test'), num_nodes=2, node_rank=1,
                              master_addr='192.0.2.1', master_port=6000, shmem_port=6001,
                              resume=True, build_workspace=None, offline_deps=None, control_port=None, ready_timeout=10)


def test_plan_preserves_user_st_and_builds_all_first(tmp_path):
    root, baseline, candidate, manifest = repository(tmp_path)
    plan = suite.make_plan(args(manifest), root)
    assert [row['revision'] for row in plan['commits']] == [baseline, candidate]
    assert len(plan['phases']['build']) == 2
    assert len(plan['phases']['install']) == 2
    commands = [row['command'] for row in plan['phases']['run']]
    for command in commands:
        assert command[command.index('--num-processes')+1] == '16'
        assert command[command.index('--num-nodes')+1] == '2'
        assert command[command.index('--node-rank')+1] == '1'
        assert command[command.index('--num-tokens')+1] == '256'
        assert command[command.index('--num-max-tokens-per-rank')+1] == '256'
        assert '--dispatch-only' not in command
        assert '--ll-stage-timer' not in command
        assert '--skip-check' not in command
        assert '--warmup' not in command and '--num-iters' not in command and '--perf-tail' not in command
        assert str(root / 'tests/elastic/test_ep_low_latency.py') in command
    assert not (root / '.tmp/ll-control-opt').exists()


def test_build_only_needs_no_network_or_device_configuration(tmp_path):
    root, _, _, manifest = repository(tmp_path)
    options = args(manifest)
    options.build_only = True
    options.devices = options.num_nodes = options.node_rank = options.master_addr = None
    plan = suite.make_plan(options, root)
    assert len(plan['phases']['build']) == 2
    assert plan['phases']['install'] == plan['phases']['run'] == []


@pytest.mark.parametrize('field,value', [('num_nodes', None), ('node_rank', None), ('devices', '0,1'), ('run_prefix', '../bad')])
def test_invalid_launch_rejected(tmp_path, field, value):
    root, _, _, manifest = repository(tmp_path)
    options = args(manifest)
    setattr(options, field, value)
    with pytest.raises(ValueError):
        suite.make_plan(options, root)


def test_occupancy_checks_only_explicit_devices(monkeypatch):
    monkeypatch.setattr(suite.subprocess, 'check_output', lambda *a, **kw: '\n'.join(f'No running processes found in NPU {i}' for i in range(16)))
    assert suite.require_idle(','.join(map(str, range(16))))
    monkeypatch.setattr(suite.subprocess, 'check_output', lambda *a, **kw: 'unrecognized output')
    with pytest.raises(RuntimeError, match='no launch'):
        suite.require_idle(','.join(map(str, range(16))))


def report_fixture(directory, revision='abc', world=16):
    directory.mkdir()
    config = dict(warmup=10, num_iters=100, perf_tail=50, route='normal', seed=0,
                  skip_check=False, skip_perf_test=False, dispatch_only=False, ll_stage_timer=False)
    identity = dict(source_revision=revision, control_revision=revision, send_cores=30, recv_cores=26, active_aiv=56,
                    logical_group=4, transport='mte', stage_timer=False,
                    sampling_protocol='exp-reduce-stable-tail-v2', world_size=world)
    result = dict(case_id=0, route='normal', config=suite.SHAPE.copy(), comparison=identity, correctness='passed')
    rows = []
    for operation, duration in [('dispatch', 88.0), ('combine', 66.0)]:
        result[operation] = dict(mean_kernel_us=duration, ranks=[dict(rank=i, samples=50, kernel_us=[duration+10]*50+[duration]*50) for i in range(world)])
        rows.append(dict(operation=operation, source_revision=revision, case_id=0, route='normal',
                         summary_statistics_scope='pooled_rank_tail_kernel_samples', perf_tail=50,
                         kernel_samples=world*50, mean_kernel_us=duration))
    report = dict(status='completed', world_size=world, case_count=1, result_count=1, config=config, results=[result])
    build.write_json(directory / 'summary.json', report)
    with (directory / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return report


def test_metric_is_pooled_tail_mean_from_summary_csv(tmp_path):
    directory = tmp_path / 'case'
    report_fixture(directory)
    assert suite.validate_summary(directory, 'abc', 30, 1) == dict(dispatch=88.0, combine=66.0)


@pytest.mark.parametrize('problem', ['missing_rank', 'missing_samples', 'wrong_timer', 'mixed_shape', 'wrong_revision', 'wrong_csv_mean', 'unchecked'])
def test_summary_rejects_incomplete_or_mismatched_data(tmp_path, problem):
    directory = tmp_path / 'case'
    report = report_fixture(directory)
    result = report['results'][0]
    if problem == 'missing_rank':
        result['dispatch']['ranks'].pop()
    elif problem == 'missing_samples':
        result['dispatch']['ranks'][0]['kernel_us'].pop()
    elif problem == 'wrong_timer':
        result['comparison']['stage_timer'] = True
    elif problem == 'mixed_shape':
        result['config']['B'] = 32
    elif problem == 'wrong_revision':
        result['comparison']['source_revision'] = 'other'
    elif problem == 'wrong_csv_mean':
        path = directory / 'summary.csv'
        path.write_text(path.read_text().replace('88.0', '87.0'))
    elif problem == 'unchecked':
        result['correctness'] = 'skipped'
    build.write_json(directory / 'summary.json', report)
    with pytest.raises(ValueError):
        suite.validate_summary(directory, 'abc', 30, 1)


def test_parent_deltas_are_not_always_baseline_deltas():
    labels = ('baseline', 'one', 'two')
    entries = [dict(label=label, revision=label, comparison_label=labels[max(0,index-1)],
                    comparison_base=labels[max(0,index-1)], comparison_kind='cumulative_parent' if index else 'baseline')
               for index, label in enumerate(labels)]
    measured = {1: {label: dict(dispatch=value, combine=40) for label, value in [('baseline', 88), ('one', 87), ('two', 89)]}}
    last = suite.comparisons(entries, measured)[-2]
    assert last['baseline_delta_us'] == 1
    assert last['parent_delta_us'] == 2
    assert last['reference_delta_us'] == 2
    assert last['previous_manifest_delta_us'] == 2


def sibling_revision(root, baseline, text='independent candidate\n'):
    git(root, 'checkout', '--detach', baseline)
    (root / 'kernel.h').write_text(text)
    git(root, 'commit', '-qam', 'independent candidate')
    return git(root, 'rev-parse', 'HEAD')


def test_independent_manifest_uses_explicit_reference_and_alternating_order(tmp_path):
    root, baseline, candidate, manifest = repository(tmp_path)
    sibling = sibling_revision(root, baseline)
    build.write_json(manifest, dict(schema=2, comparison_base=baseline, commits=[
        dict(label='baseline', revision=baseline), dict(label='one', revision=candidate),
        dict(label='two', revision=sibling, comparison_base='baseline')]))
    options = args(manifest)
    options.rounds = 4
    plan = suite.make_plan(options, root)
    assert [row['revision'] for row in plan['commits']] == [baseline, candidate, sibling]
    assert all(row['comparison_base'] == baseline for row in plan['commits'])
    assert [row['label'] for row in plan['phases']['run']] == ['baseline','one','two','two','one','baseline'] * 2
    assert [row['execution_position'] for row in plan['phases']['run']] == [1,2,3] * 4
    assert [command[command.index('--revision')+1] for command in plan['phases']['build']] == [baseline,candidate,sibling]
    measured = {round_id: {label: dict(dispatch=value, combine=40) for label, value in [('two',89),('one',87),('baseline',88)]}
                for round_id in range(1,5)}
    rows = suite.comparisons(plan['commits'], measured)
    for row in rows:
        assert row['reference_revision'] == baseline
        assert row['parent_delta_us'] is None and row['parent_delta_pct'] is None
        assert row['reference_delta_us'] == row['baseline_delta_us']
    second = next(row for row in rows if row['round']==2 and row['label']=='two' and row['operation']=='dispatch')
    assert second['execution_position'] == 1
    assert second['reference_delta_us'] == 1
    assert second['previous_manifest_label'] == 'one'
    assert second['previous_manifest_delta_us'] == 2
    options.order = 'manifest'
    assert [row['label'] for row in suite.make_plan(options, root)['phases']['run']] == ['baseline','one','two'] * 4


def test_reference_override_must_be_measured_and_does_not_follow_run_position(tmp_path):
    root, baseline, candidate, manifest = repository(tmp_path)
    sibling = sibling_revision(root, baseline)
    data = dict(schema=2, comparison_base=baseline, commits=[dict(label='baseline',revision=baseline),
        dict(label='one',revision=candidate,comparison_base=sibling), dict(label='two',revision=sibling)])
    build.write_json(manifest, data)
    entries = suite.commits(manifest, root)
    assert entries[1]['comparison_label'] == 'two'
    data['commits'].pop()
    build.write_json(manifest, data)
    with pytest.raises(ValueError, match='must be measured'):
        suite.commits(manifest, root)


def test_legacy_manifest_still_rejects_sibling_candidates(tmp_path):
    root, baseline, candidate, manifest = repository(tmp_path)
    sibling = sibling_revision(root, baseline)
    build.write_json(manifest, dict(commits=[dict(label='baseline',revision=baseline),
        dict(label='one',revision=candidate),dict(label='two',revision=sibling)]))
    with pytest.raises(ValueError, match='cumulative'):
        suite.commits(manifest, root)


def test_comparison_refuses_incomplete_round():
    entries = [dict(label='baseline',revision='a'),dict(label='one',revision='b')]
    with pytest.raises(ValueError, match='every manifest entry'):
        suite.comparisons(entries, {1: {'baseline':dict(dispatch=88,combine=40)}})


def test_sequential_builds_reuse_worktree_but_enter_real_build(tmp_path, monkeypatch):
    root, baseline, candidate, _ = repository(tmp_path)
    real_run = subprocess.run
    observed = []
    def fake_run(command, *pos, **kw):
        if len(command) >= 2 and command[1] == 'scripts/build_shmem.py':
            observed.append(('sdk', Path(kw['cwd'])))
            sdk = Path(kw['cwd']) / build.SDK
            (sdk / 'lib').mkdir(parents=True, exist_ok=True)
            (sdk / 'deepep-sdk.json').write_text('{}')
            (sdk / 'lib/libshmem.so').write_bytes(b'sdk')
            return subprocess.CompletedProcess(command, 0)
        if '-m' in command and 'wheel' in command:
            source = Path(kw['cwd'])
            observed.append(('wheel', source))
            output = Path(command[command.index('--wheel-dir')+1])
            with zipfile.ZipFile(output / 'fixture.whl', 'w') as archive:
                archive.writestr('deep_ep/_C.test.so', git(source, 'rev-parse', 'HEAD').encode())
                archive.writestr('deep_ep/libdeepep_kernels.so', git(source, 'rev-parse', 'HEAD').encode())
            assert '--no-deps' in command and '--no-build-isolation' in command
            assert kw['env']['DEEPEP_LL_STAGE_TIMER'] == 'OFF'
            assert kw['env']['PIP_NO_INDEX'] == '1'
            return subprocess.CompletedProcess(command, 0)
        return real_run(command, *pos, **kw)
    monkeypatch.setattr(build.subprocess, 'run', fake_run)
    monkeypatch.setattr(build, 'check_dependencies', lambda source: None)
    first = build.build(baseline, 30, root=root)
    second = build.build(candidate, 30, root=root)
    assert first != second
    assert [kind for kind, _ in observed] == ['sdk', 'wheel', 'sdk', 'wheel']
    assert len({path for _, path in observed}) == 1
    build.build(baseline, 30, root=root, resume=True)
    assert len(observed) == 4
    source = observed[0][1]
    (source / 'kernel.h').write_text('user work\n')
    with pytest.raises(RuntimeError, match='dirty'):
        build.clean_source(source)
    assert (source / 'kernel.h').read_text() == 'user work\n'


def test_production_preflight_rejects_ablation_identity():
    code = suite.preflight_code(dict(native_sha256='native', kernel_sha256='device', sdk_library_sha256='sdk'), 30)
    assert '19030' in code
    assert '29030' not in code
    assert 'not bool(n.low_latency_stage_timer_enabled)' in code
    assert '/proc/self/maps' in code and 'libdeepep_kernels.so' in code
    assert "=='device'" in code


@pytest.mark.parametrize('escaped', [False, True])
def test_preflight_hashes_actual_device_library_with_spaces(tmp_path, monkeypatch, escaped):
    directory = tmp_path / 'installed package with spaces'
    directory.mkdir()
    native_path, kernel_path, sdk_path = [directory / name for name in ('_C.test.so', 'libdeepep_kernels.so', 'libshmem.so')]
    for path in (native_path, kernel_path, sdk_path):
        path.write_bytes(path.name.encode())
    native = types.SimpleNamespace(__file__=str(native_path), low_latency_stage_timer_enabled=False,
                                   low_latency_core_protocol=19030, low_latency_implementation='g4-r0-prefix',
                                   low_latency_ranks_per_group=4, shmem_library_path=lambda:str(sdk_path))
    module = types.ModuleType('deep_ep._native')
    module.require_native = lambda purpose: native
    monkeypatch.setitem(sys.modules, 'deep_ep._native', module)
    original = Path.read_text
    mapped = kernel_path.as_posix().replace(' ', r'\040') if escaped else kernel_path.as_posix()
    def read_text(path, *args, **kwargs):
        if path.as_posix() == '/proc/self/maps':
            return f'7f-8f r--p 00000000 08:01 99 {mapped}\n'
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', read_text)
    manifest = dict(native_sha256=build.sha(native_path), kernel_sha256=build.sha(kernel_path), sdk_library_sha256=build.sha(sdk_path))
    exec(suite.preflight_code(manifest, 30), {})
    kernel_path.write_bytes(b'wrong candidate binary')
    with pytest.raises(AssertionError):
        exec(suite.preflight_code(manifest, 30), {})


def test_launcher_does_not_inherit_worker_identity_or_debug_flags(tmp_path, monkeypatch):
    options = args(Path('commits.json'))
    for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE'):
        monkeypatch.setenv(key, '99')
    monkeypatch.setenv('DEEPEP_BUILD_TYPE', 'Debug')
    monkeypatch.setenv('DEEPEP_BUILD_MEGAMOE', 'ON')
    env = suite.run_environment(options, dict(revision='abc'), dict(sdk='sdk'), tmp_path)
    assert all(key not in env for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE'))
    assert env['DEEPEP_BUILD_TYPE'] == 'Release'
    assert env['DEEPEP_BUILD_MEGAMOE'] == env['DEEPEP_BUILD_EP_GMM_FUSED'] == 'OFF'
    assert env['DEEPEP_LL_CONTROL_REVISION'] == 'abc'


def test_single_host_readiness_does_not_open_sockets(monkeypatch):
    options = args(Path('commits.json'))
    options.num_nodes = 1
    monkeypatch.setattr(suite.socket, 'socket', lambda *a, **kw: pytest.fail('single-node must not create a socket'))
    suite.ready_barrier(options, {'revision': 'abc'})
    with pytest.raises(RuntimeError, match='NPU busy'):
        suite.ready_barrier(options, {'revision': 'abc'}, error='NPU busy')


@pytest.mark.parametrize('mismatch,idle_error', [(False, None), (True, None), (False, 'NPU busy')])
def test_cpu_readiness_matches_commit_identity(mismatch, idle_error):
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
        probe.listen(1)
        try:
            connection = socket.create_connection(('127.0.0.1', port), timeout=1)
        except PermissionError:
            pytest.skip('This host sandbox blocks even loopback TCP; validate readiness on the Linux test host')
        else:
            connection.close()
    primary, secondary = args(Path('unused')), args(Path('unused'))
    for item in (primary, secondary):
        item.master_addr = '127.0.0.1'
        item.control_port = port
        item.ready_timeout = 5
    primary.node_rank, secondary.node_rank = 0, 1
    errors = []
    generations = range(1 if mismatch or idle_error else 2)
    def coordinator():
        try:
            for generation in generations:
                suite.ready_barrier(primary, dict(revision='abc', generation=generation))
        except Exception as error:
            errors.append(error)
    thread = threading.Thread(target=coordinator)
    thread.start()
    if mismatch or idle_error:
        with pytest.raises(RuntimeError, match='rejected'):
            suite.ready_barrier(secondary, dict(revision='other' if mismatch else 'abc', generation=0), error=idle_error)
    else:
        for generation in generations:
            suite.ready_barrier(secondary, dict(revision='abc', generation=generation))
    thread.join(timeout=6)
    assert not thread.is_alive()
    assert bool(errors) == bool(mismatch or idle_error)
