"""Single-node resume integration: real Git plans, ST reports and interrupted runs.

Device/build commands are replaced by explicit recording fakes. No test imports
Torch, initializes an NPU, installs a package or contacts a remote machine.
"""
import argparse
import copy
import csv
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys
import types

import pytest


_spec = importlib.util.spec_from_file_location('control_result_resume_tests', Path(__file__).with_name('suite.py'))
suite = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(suite)


def git(root, *arguments):
    return subprocess.check_output(['git', '-C', str(root), *arguments], text=True).strip()


@pytest.fixture
def project(tmp_path):
    root = tmp_path / 'resume project with spaces'
    root.mkdir()
    subprocess.run(['git', 'init', str(root)], check=True, capture_output=True)
    git(root, 'config', 'user.name', 'resume fixture')
    git(root, 'config', 'user.email', 'resume@example.invalid')
    (root / '.gitignore').write_text('.tmp/\nresults/\nthird-party/\n', encoding='utf-8')
    (root / 'dependencies.lock.json').write_text('{}\n', encoding='utf-8')
    (root / 'kernel.h').write_text('baseline\n', encoding='utf-8')
    git(root, 'add', '.')
    git(root, 'commit', '-qm', 'baseline')
    baseline = git(root, 'rev-parse', 'HEAD')
    (root / 'kernel.h').write_text('candidate\n', encoding='utf-8')
    git(root, 'commit', '-qam', 'candidate')
    candidate = git(root, 'rev-parse', 'HEAD')
    manifest = root / '.tmp/commits.json'
    manifest.parent.mkdir()
    suite.build.write_json(manifest, dict(schema=2, comparison_base=baseline, commits=[
        dict(label='baseline', revision=baseline), dict(label='candidate', revision=candidate)]))
    options = argparse.Namespace(commits=manifest, send_cores=30, rounds=1, order='alternating',
        build_only=False, run_only=False, devices=','.join(map(str, range(16))), run_prefix='resume-001',
        output_dir=Path('results/test'), num_nodes=1, node_rank=0, master_addr='127.0.0.1',
        master_port=6000, shmem_port=6001, resume=True, resume_results=True, build_workspace=None, offline_deps=None,
        control_port=None, ready_timeout=8, plan=False)
    return root, options


def write_report(directory, revision, world=16, send_cores=30, seed=0):
    """Match the persisted full ST summary (including all 100 raw samples)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = dict(warmup=10, num_iters=100, perf_tail=50, route='normal', seed=seed,
                  skip_check=False, skip_perf_test=False, dispatch_only=False, ll_stage_timer=False)
    identity = dict(source_revision=revision, control_revision=revision, send_cores=send_cores,
        recv_cores=56-send_cores, active_aiv=56, logical_group=4, transport='mte', stage_timer=False,
        core_protocol=19000+send_cores, implementation='g4-r0-prefix',
        sampling_protocol='exp-reduce-stable-tail-v2', world_size=world)
    result = dict(case_id=0, route='normal', config=suite.SHAPE.copy(), comparison=identity,
                  correctness='passed')
    rows = []
    for operation, base in [('dispatch', 88.0), ('combine', 138.0)]:
        ranks = []
        for rank in range(world):
            tail = [base + rank / 100 + sample / 1000 for sample in range(50)]
            ranks.append(dict(rank=rank, samples=50, mean_us=statistics.mean(tail),
                              kernel_us=[base+100+rank]*50 + tail))
        mean = statistics.mean(value for rank in ranks for value in rank['kernel_us'][-50:])
        result[operation] = dict(mean_kernel_us=mean, ranks=ranks, perf_tail=50,
            kernel_samples=world*50, summary_statistics_scope='pooled_rank_tail_kernel_samples')
        rows.append(dict(operation=operation, source_revision=revision, case_id=0, route='normal',
            summary_statistics_scope='pooled_rank_tail_kernel_samples', perf_tail=50,
            kernel_samples=world*50, mean_kernel_us=mean))
    report = dict(status='completed', world_size=world, case_count=1, result_count=1,
                  config=config, results=[result])
    suite.build.write_json(directory / 'summary.json', report)
    with (directory / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return report


def old_results(root, options, completed=None):
    """Pre-resume layout has a suite plan and summaries, but no resume ledger."""
    plan = suite.make_plan(options, root)
    base = root / options.output_dir / options.run_prefix
    base.mkdir(parents=True, exist_ok=True)
    legacy = {key: plan[key] for key in ('commits', 'phases', 'metric', 'sampling', 'send_cores',
              'recv_cores', 'timer', 'order', 'rounds', 'comparison_schema', 'control_protocol', 'control_port')}
    suite.build.write_json(base / f'suite-plan.node-{options.node_rank}.json', legacy)
    for item in plan['phases']['run']:
        if completed is None or (item['label'], item['round']) in completed:
            write_report(item['directory'], item['revision'], options.num_nodes*16, options.send_cores)
    return plan, base


def test_legacy_summary_requires_real_all_rank_stable_tails(tmp_path):
    world = 16
    directory = tmp_path / 'old summary'
    report = write_report(directory, 'revision', world)
    values = suite.validate_summary(directory, 'revision', 30, world//16)
    for operation in ('dispatch', 'combine'):
        assert values[operation] == report['results'][0][operation]['mean_kernel_us']
        assert values[operation] < 150  # The high first fifty samples are excluded.


@pytest.mark.parametrize('problem', [
    'missing_json', 'missing_csv', 'malformed_json', 'not_completed', 'missing_combine',
    'duplicate_rank', 'missing_rank', 'raw99', 'raw101', 'selected49', 'nonfinite',
    'nonpositive', 'wrong_num_iters', 'wrong_tail', 'wrong_world', 'wrong_revision',
    'wrong_split', 'wrong_shape', 'no_correctness', 'csv_missing_combine', 'csv_wrong_mean',
    'top_level_list', 'empty_results'])
def test_invalid_reports_are_never_completed(tmp_path, problem):
    directory = tmp_path / problem
    report = write_report(directory, 'revision')
    result = report['results'][0]
    json_path, csv_path = directory/'summary.json', directory/'summary.csv'
    if problem == 'not_completed': report['status'] = 'running'
    elif problem == 'missing_combine': result.pop('combine')
    elif problem == 'duplicate_rank': result['combine']['ranks'][1]['rank'] = 0
    elif problem == 'missing_rank': result['combine']['ranks'].pop()
    elif problem == 'raw99': result['combine']['ranks'][0]['kernel_us'].pop()
    elif problem == 'raw101': result['dispatch']['ranks'][0]['kernel_us'].append(88)
    elif problem == 'selected49': result['combine']['ranks'][0]['samples'] = 49
    elif problem == 'nonfinite': result['dispatch']['ranks'][0]['kernel_us'][0] = float('nan')
    elif problem == 'nonpositive': result['combine']['ranks'][0]['kernel_us'][0] = 0
    elif problem == 'wrong_num_iters': report['config']['num_iters'] = 99
    elif problem == 'wrong_tail': report['config']['perf_tail'] = 49
    elif problem == 'wrong_world': report['world_size'] = 32
    elif problem == 'wrong_revision': result['comparison']['control_revision'] = 'another'
    elif problem == 'wrong_split': result['comparison']['send_cores'] = 28
    elif problem == 'wrong_shape': result['config']['B'] = 8
    elif problem == 'no_correctness': result['correctness'] = 'skipped'
    elif problem == 'top_level_list': report = []
    elif problem == 'empty_results': report['results'] = []
    suite.build.write_json(json_path, report)
    if problem == 'missing_json': json_path.unlink()
    elif problem == 'missing_csv': csv_path.unlink()
    elif problem == 'malformed_json': json_path.write_text('{truncated', encoding='utf-8')
    elif problem.startswith('csv_'):
        with csv_path.open(newline='', encoding='utf-8') as stream: rows = list(csv.DictReader(stream))
        if problem == 'csv_missing_combine': rows = rows[:1]
        else: rows[0]['mean_kernel_us'] = '1.0'
        with csv_path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    with pytest.raises((ValueError, KeyError, OSError, TypeError)):
        suite.validate_summary(directory, 'revision', 30, 1)


def main_harness(monkeypatch, root, options, fail_once=None):
    """Execute production main; replace only environment/build/device effects."""
    original_plan, original_run, original_schedule = suite.make_plan, subprocess.run, suite.result_schedule
    observed = []
    failures = set([fail_once] if fail_once else [])
    monkeypatch.setattr(suite, 'ROOT', root)
    monkeypatch.setattr(suite, 'make_plan', lambda selected, root=root: original_plan(selected, root))
    monkeypatch.setattr(suite, 'result_schedule', lambda selected, arguments, root=root: original_schedule(selected, arguments, root))
    monkeypatch.setattr(suite, 'parser', lambda: types.SimpleNamespace(
        parse_args=lambda: options, exit=lambda status, message: (_ for _ in ()).throw(ValueError(message))))

    def verify(path, revision, split, root=root):
        assert Path(path) == suite.build.manifest_path(root, revision, split)
        observed.append(('verify', revision))
        return dict(wheel='fixture.whl', sdk='third-party/shmem/install/shmem',
                    native_sha256='a'*64, kernel_sha256='b'*64, sdk_library_sha256='c'*64,
                    sdk_manifest_sha256='d'*64)
    monkeypatch.setattr(suite.build, 'verify_manifest', verify)
    monkeypatch.setattr(suite, 'run_environment', lambda selected, item, manifest, root=root: dict(
        DEEPEP_LL_CONTROL_REVISION=item['revision'], TEST_LABEL=item['label'], TEST_ROUND=str(item.get('round', 0))))

    def fake_run(command, *position, **kwargs):
        command = list(map(str, command))
        if command[0] == 'git': return original_run(command, *position, **kwargs)
        if any(part.replace('\\', '/').endswith('ll_control_opt/build.py') for part in command):
            observed.append(('build', command[command.index('--revision')+1]))
        elif 'venv' in command:
            venv = Path(command[-1]); venv.mkdir(parents=True, exist_ok=True)
            (venv/'pyvenv.cfg').write_text('fixture\n', encoding='utf-8')
            executable = venv / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
            executable.parent.mkdir(); executable.write_bytes(b'not an executable')
            observed.append(('venv', str(venv)))
        elif 'pip' in command and 'install' in command: observed.append(('install', command[0]))
        elif '-c' in command: observed.append(('preflight', command[0]))
        else: raise AssertionError(f'Unexpected external command: {command}')
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(suite.subprocess, 'run', fake_run)
    monkeypatch.setattr(suite, 'require_idle', lambda devices: observed.append(('idle', devices)) or 'all selected devices idle\n')

    def run_st(command, path, environment):
        assert command[command.index('--node-rank')+1] == str(options.node_rank)
        assert command[command.index('--num-nodes')+1] == str(options.num_nodes)
        directory = Path(command[command.index('--output-dir')+1]) / command[command.index('--run-name')+1]
        key = (environment['TEST_LABEL'], int(environment['TEST_ROUND']))
        observed.append(('launch', *key, str(directory)))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open('x', encoding='utf-8') as stream: stream.write('synthetic ST stdout\n')
        if key in failures:
            failures.remove(key)
            directory.mkdir(parents=True, exist_ok=True)
            (directory/'partial.raw').write_bytes(b'preserve interrupted payload')
            raise InterruptedError('synthetic interruption after partial ST output')
        if options.node_rank == 0:
            write_report(directory, environment['DEEPEP_LL_CONTROL_REVISION'], options.num_nodes*16, options.send_cores,
                         seed=int(command[command.index('--seed')+1]) if '--seed' in command else 0)
    monkeypatch.setattr(suite, 'run_logged', run_st)
    return observed


def session_comparisons(base):
    paths = list((Path(base)/'sessions').glob('*/comparison.json'))
    return [(path, json.loads(path.read_text(encoding='utf-8'))) for path in paths]


def test_schedule_imports_old_results_without_done_sidecars(project):
    root, options = project
    plan, _ = old_results(root, options)
    schedule = suite.result_schedule(plan, options, root)
    assert {item['action'] for item in schedule['items']} == {'reuse'}
    assert schedule['legacy_devices_unavailable'] is True
    assert [item['source_directory'] for item in schedule['items']] == [item['directory'] for item in plan['phases']['run']]
    assert all(set(item['values']) == {'dispatch', 'combine'} for item in schedule['items'])
    assert not list(root.rglob('*.done.json'))


def test_original_cumulative_plan_without_reference_or_order_fields(project):
    root, options = project
    data = json.loads(options.commits.read_text(encoding='utf-8'))
    suite.build.write_json(options.commits, dict(commits=data['commits']))
    options.order = 'manifest'; options.rounds = 2
    current, base = old_results(root, options)
    previous_path = base/'suite-plan.node-0.json'
    previous = json.loads(previous_path.read_text(encoding='utf-8'))
    # The original cumulative suite predates explicit references, rounds/order
    # metadata and execution_position. Its commands still prove the run shape.
    previous['commits'] = [{key: row[key] for key in ('label', 'revision')} for row in previous['commits']]
    for field in ('order', 'rounds', 'comparison_schema'): previous.pop(field)
    for item in previous['phases']['run']: item.pop('execution_position')
    suite.build.write_json(previous_path, previous)
    decision = suite.result_schedule(current, options, root)
    assert len(decision['items']) == 4 and all(item['action'] == 'reuse' for item in decision['items'])
    assert decision['legacy_devices_unavailable'] is True
    options.order = 'alternating'
    with pytest.raises(ValueError, match='(?i)order'):
        suite.result_schedule(suite.make_plan(options, root), options, root)


@pytest.mark.parametrize('problem', ['missing_csv', 'bad_json', 'top_level_list', 'empty_results', 'only_done', 'partial_payload'])
def test_partial_or_corrupt_item_gets_new_directory_without_overwrite(project, problem):
    root, options = project
    plan, _ = old_results(root, options, completed={('baseline', 1)})
    candidate = plan['phases']['run'][1]
    directory = Path(candidate['directory'])
    directory.mkdir(parents=True, exist_ok=True)
    if problem in ('missing_csv', 'bad_json', 'top_level_list', 'empty_results'):
        report = write_report(directory, candidate['revision'])
        if problem == 'missing_csv': (directory/'summary.csv').unlink()
        elif problem == 'bad_json': (directory/'summary.json').write_text('{broken', encoding='utf-8')
        elif problem == 'top_level_list': suite.build.write_json(directory/'summary.json', [])
        else:
            report['results'] = []
            suite.build.write_json(directory/'summary.json', report)
    elif problem == 'only_done':
        suite.build.write_json(Path(str(directory)+'.node-0.done.json'), dict(revision=candidate['revision'], exit_code=0))
    else:
        (directory/'raw.bin').write_bytes(b'preserve this exact partial output')
    before = {path: path.read_bytes() for path in directory.parent.rglob('*') if path.is_file()}
    schedule = suite.result_schedule(plan, options, root)
    reused, missing = schedule['items']
    assert reused['action'] == 'reuse' and missing['action'] == 'run'
    assert Path(missing['directory']) != directory
    assert Path(missing['directory']).name.startswith('round-001.attempt-')
    assert not Path(missing['directory']).exists()
    assert all(path.read_bytes() == value for path, value in before.items())


@pytest.mark.parametrize('problem', ['split', 'order', 'sampling', 'manifest', 'world', 'shape'])
def test_old_plan_mismatch_rejected_before_result_reuse(project, problem):
    root, options = project
    plan, base = old_results(root, options)
    path = base/'suite-plan.node-0.json'
    previous = json.loads(path.read_text(encoding='utf-8'))
    if problem == 'split': previous['send_cores'] = 28; previous['recv_cores'] = 28
    elif problem == 'order': previous['order'] = 'manifest'
    elif problem == 'sampling': previous['sampling']['perf_tail'] = 49
    elif problem == 'manifest': previous['commits'][1]['revision'] = 'f'*40
    else:
        for item in previous['phases']['run']:
            command = item['command']
            option, value = ('--num-nodes', '2') if problem == 'world' else ('--hidden', '1024')
            command[command.index(option)+1] = value
    suite.build.write_json(path, previous)
    with pytest.raises((ValueError, RuntimeError)):
        suite.result_schedule(plan, options, root)


def test_recorded_devices_mismatch_and_round_shrink_rejected(project):
    root, options = project
    options.rounds = 2
    previous = suite.make_plan(options, root)
    changed = copy.deepcopy(options)
    changed.devices = ','.join(map(str, reversed(range(16))))
    with pytest.raises((ValueError, RuntimeError)):
        suite.validate_previous_plan(previous, suite.make_plan(changed, root), 0)
    changed = copy.deepcopy(options); changed.rounds = 1
    with pytest.raises((ValueError, RuntimeError)):
        suite.validate_previous_plan(previous, suite.make_plan(changed, root), 0)


def test_multiple_valid_attempts_are_ambiguous(project):
    root, options = project
    plan, _ = old_results(root, options)
    item = plan['phases']['run'][0]
    write_report(Path(str(item['directory'])+'.attempt-duplicate'), item['revision'])
    with pytest.raises((ValueError, RuntimeError), match='(?i)multiple|ambiguous|more than one'):
        suite.result_schedule(plan, options, root)


def test_all_completed_old_results_do_no_build_install_idle_or_launch(project, monkeypatch):
    root, options = project
    _, base = old_results(root, options)
    (base/'comparison.csv').write_text('legacy completed comparison\n', encoding='utf-8')
    suite.build.write_json(base/'comparison.json', dict(legacy_comparison='preserve exactly'))
    before = {path: path.read_bytes() for path in base.rglob('*') if path.is_file()}
    observed = main_harness(monkeypatch, root, options)
    suite.main()
    assert observed == []
    assert all(path.read_bytes() == contents for path, contents in before.items())
    reports = session_comparisons(base)
    assert len(reports) == 1
    rows = reports[0][1]['rows']
    assert len(rows) == 4 and {row['result_status'] for row in rows} == {'reused'}
    assert all(Path(row['source_directory']).is_dir() for row in rows)
    assert (reports[0][0].parent/'comparison.csv').is_file()


def test_completed_files_without_previous_plan_are_not_adopted(project):
    root, options = project
    plan = suite.make_plan(options, root)
    for item in plan['phases']['run']:
        write_report(item['directory'], item['revision'])
    with pytest.raises(ValueError, match='(?i)previous.*plan|identity'):
        suite.result_schedule(plan, options, root)


def test_incompatible_plan_stops_before_build_or_device_work(project, monkeypatch):
    root, options = project
    old_results(root, options)
    options.order = 'manifest'
    observed = main_harness(monkeypatch, root, options)
    with pytest.raises((ValueError, RuntimeError), match='(?i)mismatch|order'):
        suite.main()
    assert observed == []


def test_existing_results_without_resume_are_preserved(project, monkeypatch):
    root, options = project
    _, base = old_results(root, options)
    before = {path: path.read_bytes() for path in base.rglob('*') if path.is_file()}
    options.resume_results = False
    observed = main_harness(monkeypatch, root, options)
    with pytest.raises((ValueError, RuntimeError), match='(?i)exist|fresh|resume'):
        suite.main()
    assert observed == []
    assert all(path.read_bytes() == contents for path, contents in before.items())


def test_partial_round_builds_and_runs_only_missing_revision(project, monkeypatch):
    root, options = project
    options.rounds = 2
    plan, base = old_results(root, options, completed={('baseline', 1), ('candidate', 1), ('candidate', 2)})
    observed = main_harness(monkeypatch, root, options)
    suite.main()
    launches = [row[1:3] for row in observed if row[0] == 'launch']
    assert launches == [('baseline', 2)]
    revision = plan['commits'][0]['revision']
    assert [row[1] for row in observed if row[0] == 'build'] == [revision]
    assert [row[1] for row in observed if row[0] == 'verify'] == [revision]
    assert len([row for row in observed if row[0] == 'install']) == 1
    rows = session_comparisons(base)[0][1]['rows']
    assert len(rows) == 8
    assert {(row['label'], row['round']) for row in rows if row['result_status'] == 'new'} == {('baseline', 2)}
    assert {row['execution_position'] for row in rows if row['round'] == 2 and row['label'] == 'baseline'} == {2}


def test_interruption_reentry_repeat_and_round_extension(project, monkeypatch):
    root, options = project
    options.resume_results = False
    observed = main_harness(monkeypatch, root, options, fail_once=('candidate', 1))
    with pytest.raises((InterruptedError, RuntimeError), match='(?i)interrupt|failed'):
        suite.main()
    base = root/options.output_dir/options.run_prefix
    interrupted = list(base.rglob('partial.raw'))
    assert len(interrupted) == 1
    original = interrupted[0].read_bytes()
    first_launches = [row for row in observed if row[0] == 'launch']
    assert [row[1:3] for row in first_launches] == [('baseline', 1), ('candidate', 1)]
    baseline_summary = Path(first_launches[0][3])/'summary.json'
    baseline_bytes = baseline_summary.read_bytes()

    observed.clear(); options.resume_results = True
    suite.main()
    second = [row for row in observed if row[0] == 'launch']
    assert [row[1:3] for row in second] == [('candidate', 1)]
    assert second[0][3] != first_launches[1][3]
    assert interrupted[0].read_bytes() == original
    assert baseline_summary.read_bytes() == baseline_bytes
    assert len(session_comparisons(base)) == 1

    observed.clear()
    suite.main()
    assert observed == []
    assert len(session_comparisons(base)) == 2

    observed.clear(); options.rounds = 2
    suite.main()
    assert [row[1:3] for row in observed if row[0] == 'launch'] == [('candidate', 2), ('baseline', 2)]
    newest = next(report for _, report in session_comparisons(base) if len(report['rows']) == 8)
    assert {row['result_status'] for row in newest['rows'] if row['round'] == 1} == {'reused'}
    assert {row['result_status'] for row in newest['rows'] if row['round'] == 2} == {'new'}
    assert interrupted[0].read_bytes() == original


def test_multiple_seeds_run_once_and_resume_without_binary_dependencies(project, monkeypatch):
    root, options = project
    options.resume_results = False
    options.seeds = [0, 17]
    observed = main_harness(monkeypatch, root, options)
    suite.main()
    assert len([row for row in observed if row[0]=='build']) == 2
    assert len([row for row in observed if row[0]=='launch']) == 4
    base = root/options.output_dir/options.run_prefix
    original = json.loads((base/'comparison.json').read_text(encoding='utf-8'))
    assert {(row['label'],row['round'],row['seed'],row['operation']) for row in original['rows']} == {
        (label,1,seed,operation) for label in ('baseline','candidate') for seed in (0,17) for operation in ('dispatch','combine')}
    assert {row['measured_route_seed'] for row in original['rows']} == {2,19}
    assert {row['sdk_identity_source'] for row in original['rows']} == {'recorded-build'}
    assert all(row['sdk_manifest_sha256']=='d'*64 for row in original['rows'])
    observed.clear()
    options.resume_results = True
    suite.main()
    assert observed == []
    resumed = session_comparisons(base)[0][1]['rows']
    assert len(resumed)==8 and {row['result_status'] for row in resumed}=={'reused'}
    assert {row['sdk_identity_source'] for row in resumed}=={'recorded-build'}


@pytest.mark.parametrize('node_rank', [0, 1])
def test_result_resume_explicitly_rejects_multiple_nodes(project, node_rank):
    root, options = project
    options.num_nodes = 2
    options.node_rank = node_rank
    with pytest.raises(ValueError, match='(?i)single|one.node|num.nodes.*1|单机'):
        suite.make_plan(options, root)


def test_complete_attempt_without_readable_creation_plan_has_unknown_provenance(project, monkeypatch):
    root, options = project
    plan, base = old_results(root, options, completed={('baseline', 1)})
    # A readable modern root plan establishes current settings, but cannot
    # attest the device mapping of an attempt whose creation plan is damaged.
    suite.build.write_json(base/'suite-plan.node-0.json', plan)
    candidate = next(item for item in plan['phases']['run'] if item['label']=='candidate')
    canonical = Path(candidate['directory'])
    attempt = canonical.with_name(canonical.name+'.attempt-interrupted-session')
    write_report(attempt, candidate['revision'])
    damaged = base/'sessions'/'interrupted-session'/'suite-plan.node-0.json'
    damaged.parent.mkdir(parents=True)
    damaged.write_text('{truncated', encoding='utf-8')
    before = (attempt/'summary.json').read_bytes()
    observed = main_harness(monkeypatch, root, options)
    suite.main()
    assert observed == []
    assert (attempt/'summary.json').read_bytes() == before
    report = session_comparisons(base)[0][1]
    assert report['legacy_devices_unavailable'] is True
    assert len(report['ignored_session_plans']) == 1
    rows = [row for row in report['rows'] if row['label']=='candidate']
    assert {row['source_session_id'] for row in rows} == {'unknown'}
    assert all(row['reference_crosses_sessions'] is None for row in rows)

