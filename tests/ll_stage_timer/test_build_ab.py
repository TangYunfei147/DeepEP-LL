"""CPU tests of the one-SDK, clean-each-wheel, complete A/B build workflow."""
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

SPEC = importlib.util.spec_from_file_location('ll_build_ab', Path(__file__).with_name('build_ab.py'))
build_ab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_ab)


def make_plan(tmp_path, experiment='small', name='group001'):
    args = build_ab.parser().parse_args(['--experiment', experiment, '--run-name', name])
    return build_ab.make_plan(args, root=tmp_path, python='/existing/python', environment={})


@pytest.mark.parametrize('experiment,implementations', [
    ('small', ['cb59', 'cb59-small']), ('pair', ['cb59-stream', 'cb59-pair']),
    ('all', ['cb59', 'cb59-small', 'cb59-stream', 'cb59-pair']),
])
def test_pure_plan_is_complete_and_preserves_wheel_selection(tmp_path, experiment, implementations):
    plan = make_plan(tmp_path, experiment)
    assert [variant['dispatch_impl'] for variant in plan['variants']] == implementations
    assert plan['sdk_build'] == ['/existing/python', str(tmp_path / 'scripts/build_shmem.py')]
    for variant, implementation in zip(plan['variants'], implementations):
        assert variant['build_only'] and not variant['skip_build']
        assert variant['core_split'] == dict(send=28, recv=28)
        assert variant['small_b_prefix_cores'] == 0
        assert variant['node_dir'] == str(tmp_path / 'diagnostic-runs/group001' / implementation / 'node_000')
        assert [stage['name'] for stage in variant['stages']] == ['off', 'sparse']
        assert all(stage['run'] is None and stage['analyze'] is None for stage in variant['stages'])
    assert list(tmp_path.iterdir()) == []


def test_plan_cli_never_creates_files_or_runs_commands(monkeypatch, capsys):
    def forbid(*args, **kwargs):
        raise AssertionError('Plan attempted execution or writes')
    monkeypatch.setattr(build_ab, 'execute', forbid)
    monkeypatch.setattr(Path, 'mkdir', forbid)
    monkeypatch.setattr(subprocess, 'run', forbid)
    assert build_ab.main(['--plan', '--experiment', 'all', '--run-name', 'pure']) == 0
    value = json.loads(capsys.readouterr().out)
    assert len(value['variants']) == 4
    assert not any(argument in build_ab.parser().format_help() for argument in ('--skip-sdk', '--sdk-prepared'))


def mock_commands(monkeypatch, *, fail=None):
    monkeypatch.setattr(build_ab.importlib.util, 'find_spec', lambda name: object())
    seen = []
    def command(argv, plan, env, log):
        log = Path(log)
        seen.append((plan.get('dispatch_impl'), log.name))
        log.write_text('mock execution\n', encoding='utf-8')
        if fail == seen[-1]:
            raise RuntimeError('intentional command failure')
        if '-m' in argv and 'build' in argv:
            (Path(argv[argv.index('--outdir') + 1]) / 'built.whl').write_bytes(b'built wheel')
        assert env['PIP_NO_INDEX'] == '1'
        assert Path(plan['root']) in Path(env['SHMEM_LOG_PATH']).parents
        assert Path(plan['root']) in Path(env['TMPDIR']).parents
    monkeypatch.setattr(build_ab.diagnose, '_command', command)
    return seen


@pytest.mark.parametrize('experiment', ['small', 'pair', 'all'])
def test_sdk_built_once_before_clean_off_and_sparse_for_every_variant(tmp_path, monkeypatch, experiment):
    plan = make_plan(tmp_path, experiment)
    seen = mock_commands(monkeypatch)
    result = build_ab.execute(plan)
    implementations = list(build_ab.EXPERIMENTS[experiment])
    expected = [(implementations[0], 'sdk-build.log')]
    for implementation in implementations:
        expected += [(implementation, log) for log in (
            'venv.log', 'off-clean.log', 'off-build.log', 'sparse-clean.log', 'sparse-build.log',
            'off-install.log', 'off-identity.log', 'sparse-install.log', 'sparse-identity.log')]
    assert seen == expected
    index = json.loads(result.read_text(encoding='utf-8'))
    assert index['wheel_maps'] == {implementation: f'{implementation}/node_000/wheel-map.json' for implementation in implementations}
    for path in index['wheel_maps'].values():
        mapping = result.parent / path
        wheels = json.loads(mapping.read_text(encoding='utf-8'))
        assert set(wheels) == {'off', 'sparse'}
        assert all(not Path(value).is_absolute() and (mapping.parent / value).is_file() for value in wheels.values())
    assert not any('-run' in log or 'analyze' in log for _, log in seen)
    assert not (tmp_path / 'third-party/shmem').exists(), 'workflow must not require SDK manifest/header fixtures'


def test_sdk_failure_prevents_venv_clean_or_wheel_work(tmp_path, monkeypatch):
    plan = make_plan(tmp_path)
    seen = mock_commands(monkeypatch, fail=('cb59', 'sdk-build.log'))
    with pytest.raises(RuntimeError, match='intentional'):
        build_ab.execute(plan)
    assert seen == [('cb59', 'sdk-build.log')]
    assert (Path(plan['group']) / 'logs/sdk-build.log').is_file()
    assert not Path(plan['wheel_maps_output']).exists()
    assert all(not Path(variant['node_dir']).exists() for variant in plan['variants'])


def test_late_build_failure_keeps_earlier_maps_but_never_publishes_group_completion(tmp_path, monkeypatch):
    plan = make_plan(tmp_path)
    seen = mock_commands(monkeypatch, fail=('cb59-small', 'sparse-build.log'))
    with pytest.raises(RuntimeError, match='intentional'):
        build_ab.execute(plan)
    assert (Path(plan['group']) / 'cb59/node_000/wheel-map.json').is_file()
    assert not (Path(plan['group']) / 'cb59-small/node_000/wheel-map.json').exists()
    assert not Path(plan['wheel_maps_output']).exists()
    assert sum(log == 'sdk-build.log' for _, log in seen) == 1


def test_existing_group_is_rejected_without_commands_or_log_changes(tmp_path, monkeypatch):
    plan = make_plan(tmp_path)
    group = Path(plan['group'])
    group.mkdir(parents=True)
    old = group / 'keep.log'
    old.write_text('existing result')
    seen = mock_commands(monkeypatch)
    with pytest.raises(ValueError, match='already exists'):
        build_ab.execute(plan)
    assert seen == [] and old.read_text() == 'existing result'


def test_plan_with_spaces_uses_project_paths_and_rejects_inherited_wrong_split(tmp_path):
    root = tmp_path / 'Project With Spaces'
    plan = make_plan(root, 'pair')
    assert all(Path(variant['node_dir']).is_relative_to(root) for variant in plan['variants'])
    args = build_ab.parser().parse_args(['--experiment', 'small', '--run-name', 'x'])
    with pytest.raises(ValueError, match='requires DEEPEP_LL_SEND_CORES=28'):
        build_ab.make_plan(args, root=root, environment={'DEEPEP_LL_SEND_CORES': '30', 'DEEPEP_LL_RECV_CORES': '26'})
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('output', ['build', 'third-party/shmem/build', 'third-party/shmem/install', 'third-party/shmem/ci'])
def test_group_cannot_be_created_inside_a_directory_that_build_would_clean(tmp_path, output):
    args = build_ab.parser().parse_args(['--experiment', 'all', '--run-name', 'fresh', '--output-root', output])
    with pytest.raises(ValueError, match='cleaned build-artifact'):
        build_ab.make_plan(args, root=tmp_path, environment={})
    assert list(tmp_path.iterdir()) == []
