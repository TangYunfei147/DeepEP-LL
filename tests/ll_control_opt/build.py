"""Build committed control-path candidates in one persistent, offline worktree."""
import argparse
import atexit
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[2]
STATE = Path('.tmp/ll-control-opt')
SDK = Path('third-party/shmem/install/shmem')


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def resolve(root, path):
    path = Path(path)
    return (path if path.is_absolute() else root / path).resolve()


def manifest_path(root, revision, send_cores):
    return root / STATE / 'wheels' / f'{revision}-off-s{send_cores}' / 'manifest.json'


def verify_manifest(path, revision, send_cores, root=ROOT):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    expected = (revision, send_cores, 56 - send_cores, 'OFF', 4, 'Release')
    actual = tuple(data.get(k) for k in ('revision', 'send_cores', 'recv_cores', 'timer', 'logical_group', 'build_type'))
    if actual != expected:
        raise RuntimeError(f'Wheel manifest identity mismatch: {path}')
    wheel = resolve(Path(path).parent, data['wheel'])
    if not wheel.is_relative_to(Path(path).parent.resolve()) or sha(wheel) != data['wheel_sha256']:
        raise RuntimeError(f'Wheel missing, unsafe, or corrupt: {wheel}')
    sdk = resolve(root, data['sdk'])
    if sha(sdk / 'deepep-sdk.json') != data['sdk_manifest_sha256'] or sha(sdk / 'lib/libshmem.so') != data['sdk_library_sha256']:
        raise RuntimeError(f'Matching SDK changed or missing: {sdk}')
    return data


def clean_source(source):
    edits = git(source, 'status', '--porcelain', '--untracked-files=normal', '--ignore-submodules=dirty')
    if edits:
        raise RuntimeError(f'Refuse to change a dirty build worktree: {source}\n{edits}\nNo reset, clean, or stash was performed.')


def common_dir(source):
    return resolve(source, git(source, 'rev-parse', '--git-common-dir'))


def links(source):
    result = {}
    for line in git(source, 'ls-tree', '-r', 'HEAD').splitlines():
        mode, _, rest = line.split(' ', 2)
        if mode == '160000':
            revision, path = rest.split('\t', 1)
            result[path] = revision
    return result


def acquire_lock(path):
    """A stale lock is explicit evidence to inspect, never silently removed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(f'pid={os.getpid()}\n')
    def release():
        path.unlink(missing_ok=True)
    atexit.register(release)
    return release


def environment(source, send_cores):
    env = dict(os.environ, DEEPEP_NPU_ARCH='Ascend950', DEEPEP_LL_STAGE_TIMER='OFF',
               DEEPEP_LL_SEND_CORES=str(send_cores), DEEPEP_LL_RECV_CORES=str(56 - send_cores),
               DEEPEP_LL_RANKS_PER_GROUP='4', PIP_NO_INDEX='1', GIT_ALLOW_PROTOCOL='file',
               DEEPEP_BUILD_TYPE='Release', DEEPEP_BUILD_MEGAMOE='OFF', DEEPEP_MEGAMOE_TIMER='OFF',
               DEEPEP_BUILD_EP_GMM_FUSED='OFF', DEEPEP_LINEINFO='OFF',
               PYTHONDONTWRITEBYTECODE='1')
    for key in ('SHMEM_ROOT', 'CATLASS_ROOT'):
        env.pop(key, None)
    paths = {'TMPDIR': '.tmp', 'TMP': '.tmp', 'TEMP': '.tmp', 'PIP_CACHE_DIR': '.cache/pip',
             'XDG_CACHE_HOME': '.cache', 'CCACHE_DIR': '.cache/ccache',
             'TORCH_EXTENSIONS_DIR': '.cache/torch_extensions', 'ASCEND_CACHE_PATH': '.cache/ascend',
             'SHMEM_LOG_PATH': '.logs/shmem', 'ASCEND_WORK_PATH': '.logs/ascend',
             'ASCEND_PROCESS_LOG_PATH': '.logs/ascend'}
    for key, relative in paths.items():
        directory = source / relative
        directory.mkdir(parents=True, exist_ok=True)
        env[key] = str(directory)
    return env


def copy_local_dependencies(donor, source):
    """Copy pinned repositories via local Git; preserve the donor and its SDK."""
    for relative, revision in links(source).items():
        origin, target = donor / relative, source / relative
        if (target / '.git').exists():
            if git(target, 'rev-parse', 'HEAD') != revision:
                raise RuntimeError(f'Existing dependency has the wrong revision: {target}')
        else:
            if not (origin / '.git').exists() or git(origin, 'rev-parse', 'HEAD') != revision:
                raise RuntimeError(f'Pinned local dependency unavailable: {origin}; initialize it or pass --offline-deps. No network fallback.')
            if git(origin, 'diff', 'HEAD', '--name-only', '--ignore-submodules=dirty'):
                raise RuntimeError(f'Local dependency has tracked edits: {origin}')
            if target.exists() and any(target.iterdir()):
                raise RuntimeError(f'Refuse to overwrite nonempty dependency: {target}')
            subprocess.run(['git', 'clone', '--no-hardlinks', '--no-checkout', str(origin), str(target)], check=True)
            subprocess.run(['git', '-C', str(target), 'checkout', '--detach', revision], check=True)
        copy_local_dependencies(origin, target)
    json_rel = Path('third-party/shmem/3rdparty/json/single_include/nlohmann/json.hpp')
    if (donor / json_rel).is_file() and not (source / json_rel).exists():
        (source / json_rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(donor / json_rel, source / json_rel)
    if (donor / SDK / 'deepep-sdk.json').is_file() and not (source / SDK).exists():
        shutil.copytree(donor / SDK, source / SDK, symlinks=False)


def unpack_dependencies(bundle, source):
    """Read the existing ll_ablation/deps.py bundle format without network access."""
    bundle = bundle.resolve()
    data = json.loads((bundle / 'deps.json').read_text(encoding='utf-8'))
    for entry in data['entries']:
        relative = Path(entry['path'])
        target, artifact = source / relative, bundle / entry['file']
        if relative.is_absolute() or '..' in relative.parts or artifact.parent != bundle:
            raise ValueError('Unsafe offline dependency path')
        if target.resolve() != target or not target.is_relative_to(source) or sha(artifact) != entry['sha256']:
            raise ValueError('Unsafe or corrupt offline dependency')
        parent = target.parent
        while parent != source and not (parent / '.git').exists():
            parent = parent.parent
        if links(parent).get(target.relative_to(parent).as_posix()) != entry['revision']:
            raise ValueError('Offline dependency differs from pinned Git revision')
        if (target / '.git').exists():
            if git(target, 'rev-parse', 'HEAD') != entry['revision']:
                raise ValueError('Existing dependency differs from uploaded bundle')
            continue
        if target.exists() and any(target.iterdir()):
            raise ValueError(f'Refuse to overwrite {target}')
        target.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'init', str(target)], check=True)
        subprocess.run(['git', '-C', str(target), 'fetch', str(artifact), 'HEAD'], check=True)
        subprocess.run(['git', '-C', str(target), 'checkout', '--detach', entry['revision']], check=True)
    for asset in data.get('assets', []):
        if asset['path'] != 'third-party/shmem/3rdparty/json/single_include/nlohmann/json.hpp':
            raise ValueError('Unknown offline dependency asset')
        artifact, target = bundle / asset['file'], source / asset['path']
        if artifact.parent != bundle or sha(artifact) != asset['sha256']:
            raise ValueError('Offline asset digest mismatch')
        if target.exists():
            if sha(target) != asset['sha256']:
                raise ValueError('Existing offline asset differs; not overwritten')
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(artifact, target)


def check_dependencies(source):
    for relative, revision in links(source).items():
        target = source / relative
        if not (target / '.git').exists() or git(target, 'rev-parse', 'HEAD') != revision:
            raise RuntimeError(f'Uninitialized or mismatched dependency: {target}')
        check_dependencies(target)
    if (source / 'third-party/shmem').exists():
        header = source / 'third-party/shmem/3rdparty/json/single_include/nlohmann/json.hpp'
        if not header.is_file():
            raise RuntimeError('Uploaded/local nlohmann json header missing; no network fallback')


def build(revision, send_cores, *, root=ROOT, workspace=None, offline_deps=None, resume=False):
    revision = git(root, 'rev-parse', revision + '^{commit}')
    canonical = manifest_path(root, revision, send_cores)
    if canonical.exists():
        if not resume:
            raise RuntimeError(f'Wheel already exists; use --resume: {canonical}')
        verify_manifest(canonical, revision, send_cores, root)
        print(f'Reusing verified wheel: {canonical}', flush=True)
        return canonical
    source = resolve(root, workspace or STATE / f'source-s{send_cores}')
    if source == root.resolve() or source.is_symlink():
        raise RuntimeError('Build workspace must be a separate real worktree, never the main checkout')
    source.parent.mkdir(parents=True, exist_ok=True)
    release = acquire_lock(root / STATE / f'build-s{send_cores}.lock')
    try:
        if source.exists():
            if not (source / '.git').exists() or common_dir(source) != common_dir(root):
                raise RuntimeError('Build workspace must be a worktree of this repository')
            clean_source(source)
            previous = git(source, 'rev-parse', 'HEAD')
            if subprocess.run(['git', '-C', str(root), 'merge-base', '--is-ancestor', previous, revision]).returncode:
                raise RuntimeError(f'Cache workspace {previous} cannot fast-forward to {revision}; use a fresh --build-workspace')
            if git(source, 'show', previous + ':dependencies.lock.json') != git(root, 'show', revision + ':dependencies.lock.json'):
                raise RuntimeError('Dependency identity changed; use a fresh build workspace')
            subprocess.run(['git', '-C', str(source), 'checkout', '--no-recurse-submodules', '--detach', revision], check=True)
        else:
            subprocess.run(['git', '-C', str(root), 'worktree', 'add', '--detach', str(source), revision], check=True)
        canonical.parent.mkdir(parents=True, exist_ok=True)
        output = canonical.parent / ('attempt-' + uuid.uuid4().hex[:12])
        output.mkdir()
        env = environment(source, send_cores)
        timings = []
        def run(command):
            started, code = time.perf_counter(), None
            print('Running ' + ' '.join(map(str, command)) + f' (log: {output / "build.log"})', flush=True)
            try:
                with (output / 'build.log').open('a', encoding='utf-8') as log:
                    log.write('\n' + repr(command) + '\n')
                    log.flush()
                    code = subprocess.run(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
                if code:
                    raise subprocess.CalledProcessError(code, command)
            finally:
                timings.append(dict(command=command, seconds=time.perf_counter() - started, exit_code=code))
                write_json(output / 'timings.json', timings)
        if offline_deps:
            unpack_dependencies(resolve(root, offline_deps), source)
        else:
            copy_local_dependencies(root, source)
        check_dependencies(source)
        # Validates all SDK hashes even when reusing its existing installation.
        run([sys.executable, 'scripts/build_shmem.py'])
        # Always enter the real build: CMake/Ninja depfiles decide which objects need rebuilding.
        run([sys.executable, '-m', 'pip', 'wheel', '-v', '--no-deps', '--no-build-isolation', '--wheel-dir', str(output), '.'])
        wheels = list(output.glob('*.whl'))
        if len(wheels) != 1:
            raise RuntimeError('Expected exactly one full wheel')
        with zipfile.ZipFile(wheels[0]) as archive:
            natives = [name for name in archive.namelist() if Path(name).name.startswith('_C.') and name.endswith('.so')]
            kernels = [name for name in archive.namelist() if Path(name).name == 'libdeepep_kernels.so']
            if len(natives) != 1 or len(kernels) != 1:
                raise RuntimeError('Cannot uniquely identify the wheel extension and kernel library')
            native_sha = hashlib.sha256(archive.read(natives[0])).hexdigest()
            kernel_sha = hashlib.sha256(archive.read(kernels[0])).hexdigest()
        data = dict(revision=revision, send_cores=send_cores, recv_cores=56 - send_cores, timer='OFF',
                    logical_group=4, build_type='Release', wheel=wheels[0].relative_to(canonical.parent).as_posix(),
                    wheel_sha256=sha(wheels[0]), native_sha256=native_sha, kernel_sha256=kernel_sha,
                    source=os.path.relpath(source, root), sdk=os.path.relpath(source / SDK, root),
                    sdk_manifest_sha256=sha(source / SDK / 'deepep-sdk.json'),
                    sdk_library_sha256=sha(source / SDK / 'lib/libshmem.so'),
                    python=sys.version, cann=os.environ.get('ASCEND_HOME_PATH'))
        with canonical.open('x', encoding='utf-8') as stream:
            json.dump(data, stream, indent=2, sort_keys=True)
            stream.write('\n')
        print(canonical, flush=True)
        return canonical
    finally:
        release()


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--send-cores', type=int, default=30)
    parser.add_argument('--build-workspace', type=Path)
    parser.add_argument('--offline-deps', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.send_cores <= 55:
        parser.error('send-cores must be 1..55; total is fixed at56')
    build(args.revision, args.send_cores, workspace=args.build_workspace,
          offline_deps=args.offline_deps, resume=args.resume)


if __name__ == '__main__':
    main()
