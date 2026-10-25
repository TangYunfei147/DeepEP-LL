"""CPU contracts for seeded comparisons, legacy results, and SDK cache identity."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import pytest


_spec = importlib.util.spec_from_file_location('seed_sdk_control_suite', Path(__file__).with_name('suite.py'))
suite = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(suite)
build = suite.build
_fixtures = importlib.util.spec_from_file_location('seed_sdk_result_fixtures', Path(__file__).with_name('test_result_resume.py'))
fixtures = importlib.util.module_from_spec(_fixtures)
_fixtures.loader.exec_module(fixtures)


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def test_sdk_sidecar_restores_identity_without_sdk_binaries(tmp_path):
    directory = tmp_path/'old result'
    expected = dict(dependency_id='a'*64, shmem_revision='b'*40)
    legacy = suite.recorded_sdk_identity(directory, 'candidate', expected)
    assert legacy['sdk_identity_source']=='legacy-unavailable'
    assert legacy['sdk_library_sha256'] is None
    record = dict(expected, revision='candidate', **{field:'c'*64 for field in suite.SDK_HASH_FIELDS})
    path = suite.sdk_record_path(directory)
    build.write_json(path, record)
    assert suite.recorded_sdk_identity(directory, 'candidate', expected)['sdk_library_sha256']=='c'*64
    assert not (tmp_path/'third-party/shmem').exists()
    with pytest.raises(ValueError, match='differs'):
        suite.recorded_sdk_identity(directory, 'new-candidate', expected)
    with pytest.raises(ValueError, match='differs'):
        suite.recorded_sdk_identity(directory, 'candidate', dict(expected, shmem_revision='d'*40))


def test_known_old_sdk_is_preserved_not_copied_into_new_pin(tmp_path, monkeypatch):
    donor, source = tmp_path/'old sdk checkout', tmp_path/'new sdk checkout'
    sdk = donor/build.SDK
    (sdk/'lib').mkdir(parents=True)
    build.write_json(sdk/'deepep-sdk.json', dict(revision='a'*40))
    (sdk/'lib/libshmem.so').write_bytes(b'old SDK unchanged')
    source.mkdir()
    monkeypatch.setattr(build, 'required_links', lambda *args, **kwargs: {build.SHMEM.as_posix():'b'*40})
    build.copy_sdk_assets(donor, source)
    assert not (source/build.SDK).exists()
    assert (sdk/'lib/libshmem.so').read_bytes()==b'old SDK unchanged'


def test_new_run_rejects_a_manifest_from_another_dependency_identity():
    expected = dict(dependency_id='a'*64, shmem_revision='b'*40)
    manifest = dict(expected, **{field:'c'*64 for field in suite.SDK_HASH_FIELDS})
    assert suite.manifest_sdk_identity('candidate', expected, manifest)['shmem_revision']=='b'*40
    manifest['shmem_revision']='d'*40
    with pytest.raises(ValueError, match='dependencies'):
        suite.manifest_sdk_identity('candidate', expected, manifest)


def repository(path):
    path.mkdir()
    subprocess.run(['git', 'init', str(path)], check=True, capture_output=True)
    git(path, 'config', 'user.name', 'seed suite test')
    git(path, 'config', 'user.email', 'seed-suite@example.invalid')
    (path / '.gitignore').write_text('.tmp/\nresults/\nthird-party/\n', encoding='utf-8')
    (path / 'dependencies.lock.json').write_text('{}\n', encoding='utf-8')
    (path / 'kernel.h').write_text('baseline\n', encoding='utf-8')
    git(path, 'add', '.')
    git(path, 'commit', '-qm', 'baseline')
    return path, git(path, 'rev-parse', 'HEAD')


@pytest.fixture
def project(tmp_path):
    root, baseline = repository(tmp_path / 'seed project with spaces')
    (root / 'kernel.h').write_text('candidate\n', encoding='utf-8')
    git(root, 'commit', '-qam', 'candidate')
    candidate = git(root, 'rev-parse', 'HEAD')
    manifest = root / '.tmp/commits.json'
    manifest.parent.mkdir()
    manifest.write_text(json.dumps(dict(schema=2, comparison_base=baseline, commits=[
        dict(label='baseline', revision=baseline), dict(label='candidate', revision=candidate)])), encoding='utf-8')
    args = suite.parser().parse_args([
        '--commits', str(manifest), '--devices', ','.join(map(str, range(16))),
        '--num-nodes', '1', '--node-rank', '0', '--master-addr', '127.0.0.1',
        '--master-port', '6000', '--shmem-port', '6001', '--run-prefix', 'seed-001',
        '--output-dir', 'results/seed suite', '--rounds', '2', '--order', 'alternating'])
    return root, args


def test_default_seed_zero_retains_old_result_paths(project):
    root, args = project
    assert args.seeds == [0]
    plan = suite.make_plan(args, root)
    for item in plan['phases']['run']:
        assert item['seed'] == 0
        assert item['measured_route_seed'] == 2
        assert suite._run_key(item) == (item['label'], item['round'], 0)
        assert Path(item['directory']).relative_to(root).parts[-2:] == (
            item['label'], f'round-{item["round"]:03}')
        assert item['command'][item['command'].index('--seed')+1] == '0'


def test_multiseed_plan_pairs_all_candidates_and_builds_only_once(project):
    root, args = project
    args.seeds = [0, 7, 23]
    plan = suite.make_plan(args, root)
    assert len(plan['phases']['build']) == 2
    assert len(plan['phases']['install']) == 2
    runs = plan['phases']['run']
    assert len(runs) == 12
    assert len({suite._run_key(item) for item in runs}) == 12
    for round_id in (1, 2):
        for seed in args.seeds:
            selected = [item for item in runs if item['round'] == round_id and item['seed'] == seed]
            assert [item['label'] for item in selected] == (
                ['baseline', 'candidate'] if round_id == 1 else ['candidate', 'baseline'])
            assert [item['execution_position'] for item in selected] == [1, 2]
            for item in selected:
                assert item['measured_route_seed'] == seed+2
                assert item['command'][item['command'].index('--seed')+1] == str(seed)
                assert Path(item['directory']).parts[-3:] == (
                    item['label'], f'seed-{seed}', f'round-{round_id:03}')


@pytest.mark.parametrize('seeds', [[], [0, 0]])
def test_plan_rejects_empty_or_duplicate_seed_keys(project, seeds):
    root, args = project
    args.seeds = seeds
    with pytest.raises(ValueError):
        suite.make_plan(args, root)


def test_comparisons_never_pair_distinct_routing_seeds(project):
    root, args = project
    entries = suite.make_plan(args, root)['commits']
    measured = {
        (1, 0): {'baseline': dict(dispatch=100, combine=200), 'candidate': dict(dispatch=90, combine=180)},
        (1, 7): {'baseline': dict(dispatch=70, combine=120), 'candidate': dict(dispatch=77, combine=132)},
    }
    rows = suite.comparisons(entries, measured, 'alternating')
    assert len(rows) == 8
    for row in rows:
        assert row['measured_route_seed'] == row['seed']+2
    candidates = {(row['seed'], row['operation']): row for row in rows if row['label'] == 'candidate'}
    assert candidates[(0, 'dispatch')]['reference_delta_us'] == -10
    assert candidates[(7, 'dispatch')]['reference_delta_us'] == 7
    assert candidates[(0, 'combine')]['reference_delta_us'] == -20
    assert candidates[(7, 'combine')]['reference_delta_us'] == 12
    legacy = suite.comparisons(entries, {1: measured[(1, 0)]}, 'alternating')
    assert all(row['seed'] == 0 and row['measured_route_seed'] == 2 for row in legacy)


def test_command_identity_normalizes_only_missing_seed_zero(project):
    root, args = project
    command = suite.make_plan(args, root)['phases']['run'][0]['command']
    old = list(command)
    index = old.index('--seed')
    del old[index:index+2]
    other = list(command)
    other[other.index('--seed')+1] = '7'
    assert suite._command_identity(old) == suite._command_identity(command)
    assert suite._command_identity(other) != suite._command_identity(command)


def test_old_seed_zero_plan_without_seed_fields_can_resume(project):
    root, args = project
    current = suite.make_plan(args, root)
    previous = copy.deepcopy(current)
    previous.pop('seeds', None)
    previous['sampling'] = dict(warmup=10, num_iters=100, perf_tail=50)
    for item in previous['phases']['run']:
        item.pop('seed', None)
        item.pop('measured_route_seed', None)
        index = item['command'].index('--seed')
        del item['command'][index:index+2]
    assert suite.validate_previous_plan(previous, current, 0)['previous_rounds'] == 2
    args.seeds = [0, 7]
    with pytest.raises(ValueError):
        suite.validate_previous_plan(previous, suite.make_plan(args, root), 0)


@pytest.mark.parametrize('requested,stored,accepted', [(7, 7, True), (7, 0, False), (0, 7, False)])
def test_summary_seed_matches_requested_routes(tmp_path, requested, stored, accepted):
    directory = tmp_path / 'summary'
    report = fixtures.write_report(directory, 'revision')
    report['config'].update(seed=stored, correctness_repeats=3)
    report['results'][0]['comparison'].update(seed=stored, correctness_repeats=3)
    build.write_json(directory / 'summary.json', report)
    if accepted:
        result = suite.validate_summary(directory, 'revision', 30, 1, seed=requested)
        assert set(result) == {'dispatch', 'combine'}
    else:
        with pytest.raises(ValueError):
            suite.validate_summary(directory, 'revision', 30, 1, seed=requested)


def test_dependency_key_uses_committed_lock_and_gitlinks_not_kernel_or_dirty_tree(tmp_path):
    root, first = repository(tmp_path / 'identity project')
    original = build.dependency_identity(root, first)
    (root / 'kernel.h').write_text('candidate\n', encoding='utf-8')
    git(root, 'commit', '-qam', 'kernel candidate')
    second = git(root, 'rev-parse', 'HEAD')
    assert build.dependency_identity(root, second) == original
    (root / 'dependencies.lock.json').write_text('{"uncommitted":true}\n', encoding='utf-8')
    assert build.dependency_identity(root, second) == original
    (root / 'dependencies.lock.json').write_text(json.dumps(dict(shmem=dict(revision=first))), encoding='utf-8')
    git(root, 'commit', '-qam', 'new dependency lock')
    third = git(root, 'rev-parse', 'HEAD')
    changed = build.dependency_identity(root, third)
    assert changed['dependency_id'] != original['dependency_id']
    git(root, 'update-index', '--add', '--cacheinfo', f'160000,{first},third-party/shmem')
    git(root, 'commit', '-qm', 'new dependency gitlink')
    fourth = git(root, 'rev-parse', 'HEAD')
    linked = build.dependency_identity(root, fourth)
    assert linked['dependency_id'] != changed['dependency_id']
    assert linked['shmem_revision'] == first


def test_default_source_reuses_matching_legacy_but_isolates_changed_dependencies(tmp_path):
    root, first = repository(tmp_path / 'source cache project')
    legacy = root / build.STATE / 'source-s30'
    legacy.parent.mkdir(parents=True)
    subprocess.run(['git', '-C', str(root), 'worktree', 'add', '--detach', str(legacy), first], check=True, capture_output=True)
    assert build.default_source(root, first, 30) == legacy
    (root / 'dependencies.lock.json').write_text('{"sdk":"new"}\n', encoding='utf-8')
    git(root, 'commit', '-qam', 'new sdk')
    second = git(root, 'rev-parse', 'HEAD')
    selected = build.default_source(root, second, 30)
    identity = build.dependency_identity(root, second)
    assert selected != legacy
    assert selected.name == 'source-s30-deps-'+identity['dependency_id'][:12]
    assert git(legacy, 'rev-parse', 'HEAD') == first


def test_donor_search_skips_root_with_different_shmem_pin(tmp_path):
    dependency, old_pin = repository(tmp_path / 'dependency origin')
    (dependency / 'kernel.h').write_text('new sdk\n', encoding='utf-8')
    git(dependency, 'commit', '-qam', 'new sdk')
    new_pin = git(dependency, 'rev-parse', 'HEAD')
    root, _ = repository(tmp_path / 'donor project')
    git(root, 'update-index', '--add', '--cacheinfo', f'160000,{old_pin},third-party/shmem')
    git(root, 'commit', '-qm', 'old pinned sdk')
    target_revision = git(root, 'rev-parse', 'HEAD')
    source = root / build.STATE / 'new target'
    cache = root / build.STATE / 'old valid cache'
    source.parent.mkdir(parents=True)
    for path in (source, cache):
        subprocess.run(['git', '-C', str(root), 'worktree', 'add', '--detach', str(path), target_revision], check=True, capture_output=True)
    for base, revision in ((root, new_pin), (cache, old_pin)):
        (base / build.SHMEM).parent.mkdir(parents=True, exist_ok=True)
        # Copy a local test repository: no fetch/clone transport or remote access.
        shutil.copytree(dependency, base / build.SHMEM, dirs_exist_ok=True)
        git(base / build.SHMEM, 'checkout', '--detach', revision)
    sdk = cache / build.SDK
    sdk.mkdir(parents=True)
    (sdk / 'deepep-sdk.json').write_text('{}\n', encoding='utf-8')
    manifest = root / build.STATE / 'wheels' / 'registered' / 'manifest.json'
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(dict(source=str(cache))), encoding='utf-8')
    assert build.dependency_donor(root, source) == cache
    assert git(root / build.SHMEM, 'rev-parse', 'HEAD') == new_pin
