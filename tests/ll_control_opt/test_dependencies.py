"""Exercise offline dependency preparation with real local Git repositories."""
import importlib.util
import json
from pathlib import Path
import subprocess
import zipfile

import pytest


_spec = importlib.util.spec_from_file_location('control_dependency_build', Path(__file__).with_name('build.py'))
build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build)


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True, stderr=subprocess.PIPE).strip()


def init(path):
    path.mkdir(parents=True)
    subprocess.run(['git', 'init', str(path)], check=True, capture_output=True)
    git(path, 'config', 'user.name', 'dependency fixture')
    git(path, 'config', 'user.email', 'fixture@example.invalid')
    git(path, 'config', 'core.autocrlf', 'false')
    return path


def commit(path, message='fixture'):
    git(path, 'add', '.')
    git(path, 'commit', '-qm', message)
    return git(path, 'rev-parse', 'HEAD')


def clone(origin, target):
    completed = subprocess.run(['git', 'clone', '--no-hardlinks', str(origin), str(target)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    return target


def sdk(root):
    """Presence fixture only; the production preparer must validate its contents."""
    directory = root / build.SDK
    (directory / 'lib').mkdir(parents=True)
    (directory / 'deepep-sdk.json').write_text('{"fixture": "requires validation"}\n', encoding='utf-8')
    (directory / 'lib/libshmem.so').write_bytes(b'sdk fixture')
    return directory


def header(root):
    path = root / build.JSON_HEADER
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(f'#define NLOHMANN_JSON_VERSION_{key} {value}'
                              for key, value in [('MAJOR', 3), ('MINOR', 11), ('PATCH', 3)]), encoding='utf-8')
    return path


@pytest.fixture
def project(tmp_path):
    seed = init(tmp_path / 'dependency seed')
    (seed / '.gitignore').write_text('/install/\n/3rdparty/\n', encoding='utf-8')
    (seed / 'source.h').write_text('original\n', encoding='utf-8')
    revision = commit(seed)
    root = init(tmp_path / 'project with spaces')
    (root / '.gitignore').write_text('/.tmp/\n/.ablation/\n', encoding='utf-8')
    (root / 'dependencies.lock.json').write_text(json.dumps({'shmem': {'revision': revision}}), encoding='utf-8')
    (root / '.gitmodules').write_text(
        '[submodule "shmem"]\n\tpath = third-party/shmem\n\turl = local-shmem\n'
        '[submodule "catlass"]\n\tpath = third-party/catlass\n\turl = local-catlass\n', encoding='utf-8')
    git(root, 'add', '.')
    for path in ('third-party/shmem', 'third-party/catlass'):
        git(root, 'update-index', '--add', '--cacheinfo', f'160000,{revision},{path}')
    git(root, 'commit', '-qm', 'project with pinned dependencies')
    return root, seed, revision


def worktree(root, relative='.tmp/build source'):
    path = (root / relative).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', '-C', str(root), 'worktree', 'add', '--detach', str(path), 'HEAD'],
                   check=True, capture_output=True)
    return path


def bundle(tmp_path, seed, revision, entries=None):
    directory = tmp_path / 'offline bundle'
    directory.mkdir()
    artifact = directory / 'shmem.bundle'
    git(seed, 'bundle', 'create', str(artifact), 'HEAD')
    main = dict(path='third-party/shmem', revision=revision, file=artifact.name, sha256=build.sha(artifact))
    data = dict(entries=[main] if entries is None else entries(main), assets=[])
    (directory / 'deps.json').write_text(json.dumps(data), encoding='utf-8')
    return directory, data


def test_local_core_build_requires_shmem_but_not_catlass(project):
    root, seed, revision = project
    clone(seed, root / build.SHMEM)
    original_sdk = sdk(root)
    source = worktree(root)
    build.copy_local_dependencies(root, source)
    build.check_dependencies(source)
    assert git(source / build.SHMEM, 'rev-parse', 'HEAD') == revision
    assert not (source / 'third-party/catlass/.git').exists()
    assert not (source / build.JSON_HEADER).exists()
    assert (source / build.SDK / 'lib/libshmem.so').read_bytes() == b'sdk fixture'
    assert (original_sdk / 'lib/libshmem.so').read_bytes() == b'sdk fixture'


def test_cold_sdk_requires_json_but_installed_sdk_defers_full_validation(project):
    root, seed, _ = project
    clone(seed, root / build.SHMEM)
    with pytest.raises(RuntimeError, match='Cold SDK build.*json'):
        build.check_dependencies(root)
    json_header = header(root)
    build.check_dependencies(root)
    json_header.unlink()
    sdk(root)
    # This is only dependency staging: build_shmem.py remains responsible for
    # rejecting the intentionally invalid fixture identity before any wheel.
    build.check_dependencies(root)


@pytest.mark.parametrize('problem', ['wrong_revision', 'dirty', 'foreign_worktree'])
def test_existing_shmem_is_rejected_without_overwrite(project, problem):
    root, seed, revision = project
    target = clone(seed, root / build.SHMEM)
    if problem == 'wrong_revision':
        git(target, 'config', 'user.name', 'fixture')
        git(target, 'config', 'user.email', 'fixture@example.invalid')
        (target / 'source.h').write_text('different commit\n', encoding='utf-8')
        commit(target)
    elif problem == 'dirty':
        (target / 'source.h').write_text('user edits\n', encoding='utf-8')
    else:
        git(target, 'config', 'core.worktree', str(seed))
    before = (target / 'source.h').read_bytes()
    with pytest.raises(RuntimeError):
        build.check_checkout(target, revision)
    assert (target / 'source.h').read_bytes() == before


def test_nonempty_dependency_is_not_overwritten(project):
    root, seed, _ = project
    clone(seed, root / build.SHMEM)
    source = worktree(root)
    target = source / build.SHMEM
    target.mkdir(parents=True, exist_ok=True)
    (target / 'user.txt').write_text('preserve me', encoding='utf-8')
    with pytest.raises(RuntimeError, match='nonempty'):
        build.copy_local_dependencies(root, source)
    assert (target / 'user.txt').read_text(encoding='utf-8') == 'preserve me'


def test_offline_bundle_ignores_missing_optional_catlass_artifacts(project, tmp_path):
    root, seed, revision = project
    source = worktree(root)
    def entries(main):
        return [dict(path='third-party/catlass', revision=revision, file='missing-catlass.bundle', sha256='unused'),
                dict(path='third-party/catlass/tests/dependency', revision=revision,
                     file='missing-catlass-child.bundle', sha256='unused'), main]
    uploaded, _ = bundle(tmp_path, seed, revision, entries)
    build.unpack_dependencies(uploaded, source)
    sdk(source)
    build.check_dependencies(source)
    assert git(source / build.SHMEM, 'rev-parse', 'HEAD') == revision
    assert not (source / 'third-party/catlass/.git').exists()


@pytest.mark.parametrize('problem', ['path_escape', 'absolute_path', 'artifact_escape', 'unregistered_child',
                                    'wrong_pin', 'wrong_hash'])
def test_offline_bundle_rejects_unsafe_or_unpinned_required_dependencies(project, tmp_path, problem):
    root, seed, revision = project
    source = worktree(root)
    uploaded, data = bundle(tmp_path, seed, revision)
    entry = data['entries'][0]
    if problem == 'path_escape':
        entry['path'] = '../escaped'
    elif problem == 'absolute_path':
        entry['path'] = str((tmp_path / 'escaped').resolve())
    elif problem == 'artifact_escape':
        entry['file'] = '../escaped.bundle'
    elif problem == 'unregistered_child':
        data['entries'].append(dict(entry, path='third-party/shmem/not-a-submodule'))
    elif problem == 'wrong_pin':
        entry['revision'] = '0' * 40
    else:
        entry['sha256'] = '0' * 64
    (uploaded / 'deps.json').write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(ValueError):
        build.unpack_dependencies(uploaded, source)
    assert not (tmp_path / 'escaped').exists()


def test_offline_existing_dirty_shmem_is_not_silently_reused(project, tmp_path):
    root, seed, revision = project
    source = worktree(root)
    target = clone(seed, source / build.SHMEM)
    (target / 'source.h').write_text('user edits\n', encoding='utf-8')
    uploaded, _ = bundle(tmp_path, seed, revision)
    with pytest.raises(RuntimeError, match='tracked edits'):
        build.unpack_dependencies(uploaded, source)
    assert (target / 'source.h').read_text(encoding='utf-8') == 'user edits\n'


@pytest.mark.parametrize('cache_kind', ['ablation', 'control'])
@pytest.mark.parametrize('root_has_source', [False, True])
def test_registered_cache_sdk_is_reused_with_or_without_root_source(project, cache_kind, root_has_source):
    root, seed, revision = project
    if root_has_source:
        clone(seed, root / build.SHMEM)
    cache_relative = '.ablation/sources/ready-replay' if cache_kind == 'ablation' else '.tmp/ll-control-opt/previous'
    cache = worktree(root, cache_relative)
    clone(seed, cache / build.SHMEM)
    sdk(cache)
    manifests = root / ('.ablation/wheels/ready-replay' if cache_kind == 'ablation'
                        else '.tmp/ll-control-opt/wheels/previous')
    manifests.mkdir(parents=True)
    (manifests / 'manifest.json').write_text(json.dumps({'source': cache_relative}), encoding='utf-8')
    source = worktree(root)
    assert (root / build.SHMEM / '.git').exists() == root_has_source
    assert build.dependency_donor(root, source) == cache
    build.copy_local_dependencies(root, source)
    build.check_dependencies(source)
    assert git(source / build.SHMEM, 'rev-parse', 'HEAD') == revision
    assert (source / build.SDK / 'lib/libshmem.so').read_bytes() == b'sdk fixture'


def test_existing_source_from_failed_attempt_can_copy_root_sdk(project):
    root, seed, revision = project
    clone(seed, root / build.SHMEM)
    sdk(root)
    source = worktree(root)
    clone(seed, source / build.SHMEM)
    previous = source / '.tmp/previous-attempt'
    previous.mkdir(parents=True)
    (previous / 'build.log').write_text('retained failure evidence', encoding='utf-8')
    assert not (source / build.SDK).exists()
    build.copy_local_dependencies(root, source)
    build.check_dependencies(source)
    assert git(source / build.SHMEM, 'rev-parse', 'HEAD') == revision
    assert (source / build.SDK / 'lib/libshmem.so').read_bytes() == b'sdk fixture'
    assert (previous / 'build.log').read_text(encoding='utf-8') == 'retained failure evidence'


def test_manifest_cannot_select_foreign_repository_as_cache(project):
    root, seed, _ = project
    foreign = clone(root, root / '.ablation/sources/foreign')
    clone(seed, foreign / build.SHMEM)
    sdk(foreign)
    manifests = root / '.ablation/wheels/foreign'
    manifests.mkdir(parents=True)
    (manifests / 'manifest.json').write_text(json.dumps({'source': '.ablation/sources/foreign'}), encoding='utf-8')
    with pytest.raises(RuntimeError, match='Pinned SHMEM source unavailable'):
        build.dependency_donor(root, worktree(root))


def test_existing_sdk_directory_is_preserved_even_when_incomplete(project):
    root, seed, _ = project
    clone(seed, root / build.SHMEM)
    sdk(root)
    source = worktree(root)
    clone(seed, source / build.SHMEM)
    partial = source / build.SDK
    partial.mkdir(parents=True)
    (partial / 'user.txt').write_text('partial SDK evidence', encoding='utf-8')
    build.copy_sdk_assets(root, source)
    assert not (partial / 'deepep-sdk.json').exists()
    assert (partial / 'user.txt').read_text(encoding='utf-8') == 'partial SDK evidence'


def test_dependency_staging_never_bypasses_real_sdk_validation(project, monkeypatch):
    root, seed, _ = project
    clone(seed, root / build.SHMEM)
    sdk(root)
    original_run = subprocess.run
    invoked = []
    def run(command, *args, **kwargs):
        if len(command) > 1 and command[1] == 'scripts/build_shmem.py':
            invoked.append('sdk validation')
            return subprocess.CompletedProcess(command, 19)
        if 'wheel' in command:
            pytest.fail('A rejected SDK must never reach wheel compilation')
        return original_run(command, *args, **kwargs)
    monkeypatch.setattr(build.subprocess, 'run', run)
    with pytest.raises(subprocess.CalledProcessError) as caught:
        build.build('HEAD', 30, root=root)
    assert caught.value.returncode == 19
    assert invoked == ['sdk validation']
    assert not build.manifest_path(root, git(root, 'rev-parse', 'HEAD'), 30).exists()


@pytest.fixture
def sibling_build(tmp_path, monkeypatch):
    """Keep Git/cache operations real; replace only Linux SDK/wheel commands."""
    root = init(tmp_path / 'sibling project with spaces')
    (root / '.gitignore').write_text('/.tmp/\n/.cache/\n/.logs/\n/third-party/\n', encoding='utf-8')
    (root / 'dependencies.lock.json').write_text('{}\n', encoding='utf-8')
    (root / 'kernel.h').write_text('baseline\n', encoding='utf-8')
    baseline = commit(root, 'common baseline')
    (root / 'kernel.h').write_text('candidate a\n', encoding='utf-8')
    first = commit(root, 'independent candidate a')
    git(root, 'checkout', '--detach', baseline)
    (root / 'kernel.h').write_text('candidate b\n', encoding='utf-8')
    second = commit(root, 'independent candidate b')
    observed = []
    original_run = subprocess.run

    def run(command, *args, **kwargs):
        if len(command) > 1 and command[1] == 'scripts/build_shmem.py':
            source = Path(kwargs['cwd'])
            observed.append(('sdk', git(source, 'rev-parse', 'HEAD'), source))
            if not (source / build.SDK).exists():
                sdk(source)
            return subprocess.CompletedProcess(command, 0)
        if '-m' in command and 'wheel' in command:
            source = Path(kwargs['cwd'])
            revision = git(source, 'rev-parse', 'HEAD')
            observed.append(('wheel', revision, source))
            assert '--no-deps' in command and '--no-build-isolation' in command
            assert kwargs['env']['DEEPEP_LL_STAGE_TIMER'] == 'OFF'
            output = Path(command[command.index('--wheel-dir') + 1])
            with zipfile.ZipFile(output / 'fixture.whl', 'w') as archive:
                archive.writestr('deep_ep/_C.fixture.so', revision.encode())
                archive.writestr('deep_ep/libdeepep_kernels.so', (source / 'kernel.h').read_bytes())
            return subprocess.CompletedProcess(command, 0)
        return original_run(command, *args, **kwargs)

    monkeypatch.setattr(build.subprocess, 'run', run)
    return root, baseline, first, second, observed


def test_sibling_switches_rebuild_once_then_resume_in_any_round_order(sibling_build):
    root, baseline, first, second, observed = sibling_build
    first_manifest = build.build(first, 30, root=root)
    source = observed[0][2]
    cache = source / '.cache/native-object.fixture'
    cache.write_bytes(b'preserved compiler cache')
    paths = {first: first_manifest}
    # A sibling transition and a transition back to the common parent must
    # both enter the normal build, rather than needing a fresh worktree.
    for revision in (second, baseline):
        paths[revision] = build.build(revision, 30, root=root)
        assert git(source, 'rev-parse', 'HEAD') == revision
        assert cache.read_bytes() == b'preserved compiler cache'
    assert [(kind, revision) for kind, revision, _ in observed] == [
        (kind, revision) for revision in (first, second, baseline) for kind in ('sdk', 'wheel')]
    assert len({location for _, _, location in observed}) == 1
    before = {path: path.read_bytes() for path in paths.values()}
    for revision in (baseline, first, second, second, baseline, first):
        assert build.build(revision, 30, root=root, resume=True) == paths[revision]
        build.verify_manifest(paths[revision], revision, 30, root)
    assert len(observed) == 6
    assert {path: path.read_bytes() for path in paths.values()} == before
    assert git(source, 'rev-parse', 'HEAD') == baseline
    assert git(root, 'rev-parse', 'HEAD') == second


@pytest.mark.parametrize('dirty_kind', ['tracked', 'staged', 'untracked'])
def test_sibling_switch_refuses_dirty_worktree_before_checkout(sibling_build, dirty_kind):
    root, _, first, second, observed = sibling_build
    build.build(first, 30, root=root)
    source = observed[0][2]
    path = source / ('user-new.h' if dirty_kind == 'untracked' else 'kernel.h')
    path.write_text('user changes must survive\n', encoding='utf-8')
    if dirty_kind == 'staged':
        git(source, 'add', path.name)
    status = git(source, 'status', '--porcelain')
    with pytest.raises(RuntimeError, match='dirty build worktree'):
        build.build(second, 30, root=root)
    assert git(source, 'rev-parse', 'HEAD') == first
    assert git(source, 'status', '--porcelain') == status
    assert path.read_text(encoding='utf-8') == 'user changes must survive\n'
    assert len(observed) == 2
    assert not build.manifest_path(root, second, 30).exists()


@pytest.mark.parametrize('dependency_change', ['lock', 'gitlink'])
def test_sibling_cache_rejects_changed_dependency_identity_before_checkout(sibling_build, dependency_change):
    root, _, first, _, observed = sibling_build
    build.build(first, 30, root=root)
    source = observed[0][2]
    if dependency_change == 'lock':
        (root / 'dependencies.lock.json').write_text('{"changed": true}\n', encoding='utf-8')
        changed = commit(root, 'different SDK identity')
        expected = 'Dependency identity changed'
    else:
        git(root, 'update-index', '--add', '--cacheinfo', f'160000,{first},third-party/shmem')
        git(root, 'commit', '-qm', 'different pinned dependency')
        changed = git(root, 'rev-parse', 'HEAD')
        expected = 'Pinned Git dependencies changed'
    with pytest.raises(RuntimeError, match=expected):
        build.build(changed, 30, root=root, workspace=source)
    assert git(source, 'rev-parse', 'HEAD') == first
    assert len(observed) == 2
    assert not build.manifest_path(root, changed, 30).exists()


def test_default_build_isolates_changed_dependencies_and_preserves_old_sdk(sibling_build):
    root, _, first, _, observed = sibling_build
    old_manifest = build.build(first, 30, root=root)
    old_source = observed[0][2]
    old_sdk = (old_source/build.SDK/'deepep-sdk.json').read_bytes()
    (root/'dependencies.lock.json').write_text('{"changed": true}\n', encoding='utf-8')
    changed = commit(root, 'independent SDK cache identity')
    new_manifest = build.build(changed, 30, root=root)
    new_source = observed[2][2]
    assert new_source != old_source
    assert git(old_source, 'rev-parse', 'HEAD') == first
    assert (old_source/build.SDK/'deepep-sdk.json').read_bytes() == old_sdk
    old = json.loads(old_manifest.read_text(encoding='utf-8'))
    new = json.loads(new_manifest.read_text(encoding='utf-8'))
    assert old['dependency_id'] != new['dependency_id']
    assert len(observed) == 4


def test_later_round_rejects_corrupt_finished_wheel_without_overwriting(sibling_build):
    root, _, first, second, observed = sibling_build
    manifest_path = build.build(first, 30, root=root)
    build.build(second, 30, root=root)
    identity = json.loads(manifest_path.read_text(encoding='utf-8'))
    wheel = build.resolve(manifest_path.parent, identity['wheel'])
    wheel.write_bytes(b'damaged wheel retained for inspection')
    with pytest.raises(RuntimeError, match='corrupt'):
        build.build(first, 30, root=root, resume=True)
    assert wheel.read_bytes() == b'damaged wheel retained for inspection'
    assert len(observed) == 4
    assert git(observed[0][2], 'rev-parse', 'HEAD') == second
