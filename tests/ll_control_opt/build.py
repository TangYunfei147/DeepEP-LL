"""Build committed control-path candidates in one persistent, offline worktree."""
import argparse
import atexit
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import stat
import subprocess
import sys
import time
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[2]
STATE = Path('.tmp/ll-control-opt')
SDK = Path('third-party/shmem/install/shmem')
SHMEM = Path('third-party/shmem')
JSON_HEADER = SHMEM / '3rdparty/json/single_include/nlohmann/json.hpp'


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


def dependency_identity(root, revision):
    """Read committed dependency inputs, never a dirty checkout's current pin."""
    lock = json.loads(git(root, 'show', revision+':dependencies.lock.json'))
    gitlinks = [line for line in git(root, 'ls-tree', '-r', revision).splitlines() if line.startswith('160000 ')]
    encoded = json.dumps(dict(lock=lock, gitlinks=gitlinks), sort_keys=True, separators=(',', ':')).encode('utf-8')
    return dict(dependency_id=hashlib.sha256(encoded).hexdigest(),
                shmem_revision=lock.get('shmem', {}).get('revision'))


def default_source(root, revision, send_cores):
    identity = dependency_identity(root, revision)
    legacy = root / STATE / f'source-s{send_cores}'
    if (legacy / '.git').exists():
        try:
            if common_dir(legacy) == common_dir(root) and dependency_identity(legacy, 'HEAD') == identity:
                return legacy
        except (OSError, ValueError, subprocess.CalledProcessError):
            pass
    return root / STATE / f'source-s{send_cores}-deps-{identity["dependency_id"][:12]}'


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
    identity = dependency_identity(root, revision)
    recorded = json.loads((sdk / 'deepep-sdk.json').read_text(encoding='utf-8'))
    if identity['shmem_revision'] is not None and recorded.get('revision') != identity['shmem_revision']:
        raise RuntimeError('SDK revision differs from the tested commit dependency lock')
    for field, value in identity.items():
        if field in data and data[field] != value:
            raise RuntimeError(f'Wheel dependency identity mismatch: {field}')
    data.update(identity)  # Legacy manifests are enriched in memory, never overwritten.
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


def _process_identity(pid):
    """Return alive/dead/unknown plus a best-effort process birth identity.

    Never use os.kill(pid, 0) on Windows: unlike POSIX, that API can terminate
    the target. A reused, currently live PID is conservatively kept locked.
    """
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel.GetProcessTimes.argtypes = (wintypes.HANDLE, *([ctypes.POINTER(wintypes.FILETIME)] * 4))
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION only.
        if not handle:
            return ('dead', None) if ctypes.get_last_error() == 87 else ('unknown', None)
        try:
            code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                return 'unknown', None
            if code.value != 259:  # STILL_ACTIVE
                return 'dead', None
            times = [wintypes.FILETIME() for _ in range(4)]
            if kernel.GetProcessTimes(handle, *(ctypes.byref(value) for value in times)):
                return 'alive', f'windows:{times[0].dwHighDateTime:08x}{times[0].dwLowDateTime:08x}'
            return 'alive', None
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)  # POSIX existence query; no signal is delivered.
    except ProcessLookupError:
        return 'dead', None
    except (OSError, OverflowError):
        return 'unknown', None
    try:
        # comm may contain spaces and parentheses; starttime is field 22.
        process = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        started = process[19]
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        return 'alive', f'linux:{boot}:{started}'
    except (OSError, ValueError, IndexError):
        return 'alive', None


def _lock_snapshot(path):
    if path.is_symlink():
        raise RuntimeError(f'Refuse a symlink lock: {path}')
    entry = path.lstat()
    if not stat.S_ISREG(entry.st_mode) or entry.st_size > 4096:
        raise RuntimeError(f'Invalid lock file: {path}')
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > 4096:
            raise RuntimeError(f'Invalid lock file: {path}')
        contents = stream.read(4097)
        after = os.fstat(stream.fileno())
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
    if identity(before) != identity(after):
        raise RuntimeError(f'Lock changed during inspection: {path}')
    return identity(after), contents


def _same_lock(path, snapshot):
    try:
        return _lock_snapshot(path) == snapshot
    except (OSError, RuntimeError):
        return False


def _dead_lock_owner(contents, path):
    try:
        text = contents.decode('utf-8').strip()
        legacy = re.fullmatch(r'pid=([1-9][0-9]*)', text)
        if legacy:
            pid = int(legacy.group(1))
        else:
            owner = json.loads(text)
            if (not isinstance(owner, dict) or set(owner) != {'schema', 'pid', 'host', 'start', 'token'} or
                    type(owner['schema']) is not int or owner['schema'] != 1 or
                    not isinstance(owner['host'], str) or not owner['host'] or
                    not isinstance(owner['token'], str) or not re.fullmatch('[0-9a-f]{32}', owner['token']) or
                    (owner['start'] is not None and not isinstance(owner['start'], str))):
                raise ValueError('invalid owner identity')
            if owner['host'] != socket.gethostname():
                raise RuntimeError(f'Lock belongs to another host; preserved: {path}')
            pid = owner['pid']
        if isinstance(pid, bool) or not isinstance(pid, int) or not 1 <= pid <= 0x7fffffff:
            raise ValueError('invalid PID')
    except (UnicodeError, ValueError, TypeError, KeyError) as error:
        raise RuntimeError(f'Cannot recover malformed lock; preserved: {path}') from error
    state, _ = _process_identity(pid)
    if state != 'dead':
        reason = f'PID {pid} is alive' if state == 'alive' else f'cannot confirm PID {pid} is dead'
        raise RuntimeError(f'Cannot recover lock: {reason}; preserved: {path}')
    return pid


def _acquire_os_guard(path):
    """Keep the guard inode forever: unlinking it would permit dual owners."""
    if path.is_symlink():
        raise RuntimeError(f'Refuse a symlink lock guard: {path}')
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    stream = os.fdopen(descriptor, 'r+b', buffering=0)
    try:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise OSError('Lock guard must be a regular file')
        if os.name == 'nt':
            import msvcrt
            if os.fstat(stream.fileno()).st_size == 0:
                stream.write(b'\0')
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        stream.close()
        raise FileExistsError(f'Lock guard is held or inaccessible; no recovery attempted: {path}') from error
    return stream


def acquire_lock(path, *, recover_dead=False):
    """Recover a proven dead owner only on explicit suite result resumption.

    The default build behavior still refuses any existing marker. The OS guard
    serializes recovery and stays held for the entire ownership lifetime; a
    crashed process releases it automatically. Legacy pid-only markers are
    recoverable only when that local PID is definitively absent.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    guard = _acquire_os_guard(path.with_name(path.name + '.guard'))
    try:
        if path.exists() or path.is_symlink():
            if not recover_dead:
                raise FileExistsError(f'Lock already exists; preserved: {path}')
            old = _lock_snapshot(path)
            pid = _dead_lock_owner(old[1], path)
            if not _same_lock(path, old):
                raise RuntimeError(f'Lock changed before recovery; preserved: {path}')
            path.unlink()
            print(f'Recovering lock of confirmed exited PID {pid}: {path}', flush=True)
        owner = dict(schema=1, pid=os.getpid(), host=socket.gethostname(),
                     start=_process_identity(os.getpid())[1], token=uuid.uuid4().hex)
        with path.open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(owner, sort_keys=True) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        snapshot = _lock_snapshot(path)
    except BaseException:
        guard.close()
        raise
    released = False
    def release():
        nonlocal released
        if released:
            return
        released = True
        try:
            # A delayed atexit callback must not remove a replacement owner.
            if _same_lock(path, snapshot):
                path.unlink()
        finally:
            guard.close()
            atexit.unregister(release)
    atexit.register(release)
    return release


def environment(source, send_cores):
    env = dict(os.environ, DEEPEP_NPU_ARCH='Ascend950', DEEPEP_LL_STAGE_TIMER='OFF',
               DEEPEP_LL_SEND_CORES=str(send_cores), DEEPEP_LL_RECV_CORES=str(56 - send_cores),
               DEEPEP_LL_RANKS_PER_GROUP='4', PIP_NO_INDEX='1', GIT_ALLOW_PROTOCOL='file',
               DEEPEP_BUILD_TYPE='Release', DEEPEP_BUILD_MEGAMOE='OFF', DEEPEP_MEGAMOE_TIMER='OFF',
               DEEPEP_BUILD_EP_GMM_FUSED='OFF', DEEPEP_BUILD_HT='OFF', DEEPEP_LINEINFO='OFF',
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


def shared_pull_probe_mode(source):
    header=Path(source)/'csrc/kernels/dispatch/low_latency/common/ll_shared_pull_probe_config.h'
    if not header.exists():
        return 0
    match=re.search(r'^#define DEEPEP_LL_SHARED_PULL_PROBE_MODE ([012])$',
                    header.read_text(encoding='utf-8'),re.M)
    if not match:
        raise ValueError('Cannot identify committed shared-pull prototype mode')
    return int(match.group(1))


def source_protocol_revision(source, filename, macro, default, allowed):
    """Read source-controlled protocol fields; old pinned sources omit them."""
    header=Path(source)/'csrc/kernels/dispatch/low_latency/common'/filename
    if not header.exists():
        return default
    text=header.read_text(encoding='utf-8')
    if not re.search(r'^\s*#\s*define\s+'+re.escape(macro)+r'\b',text,re.M):
        return default
    match=re.search(r'^#define '+re.escape(macro)+r' ([0-9]+)$',text,re.M)
    if not match or int(match.group(1)) not in allowed:
        raise ValueError(f'Cannot identify committed {macro}')
    return int(match.group(1))


def prefix_ready_revision(source):
    return source_protocol_revision(source,'ll_core_config.h',
        'DEEPEP_LL_PREFIX_READY_REVISION',0,(0,1))


def shared_pull_wire_revision(source):
    return source_protocol_revision(source,'ll_shared_pull_probe_config.h',
        'DEEPEP_LL_SHARED_PULL_WIRE_REVISION',1,(1,2))


def protocol_base(mode, prefix, wire):
    if (type(mode) is not int or mode not in (0,1,2) or
            type(prefix) is not int or prefix not in (0,1) or
            type(wire) is not int or wire not in (1,2) or
            (prefix==0 and wire!=1)):
        raise ValueError('Invalid shared-pull/prefix protocol combination')
    return (19000,23000,24000)[mode] if prefix==0 else (25000,26000,27000 if wire==1 else 28000)[mode]


def source_protocol_identity(source):
    identity=dict(shared_pull_probe_mode=shared_pull_probe_mode(source),
                  prefix_ready_revision=prefix_ready_revision(source),
                  shared_pull_wire_revision=shared_pull_wire_revision(source))
    protocol_base(*identity.values())
    return identity


def required_links(source, project=True):
    dependencies = links(source)
    # This suite fixes MegaMoE and fused builds OFF. CATLASS (including its
    # test/DSL submodules) is not part of the core LL build graph.
    return {p:r for p,r in dependencies.items() if not project or p==SHMEM.as_posix()}


def check_checkout(path, revision):
    if not (path / '.git').exists() or Path(git(path, 'rev-parse', '--show-toplevel')).resolve()!=path.resolve():
        raise RuntimeError(f'Pinned local dependency unavailable: {path}')
    if git(path, 'rev-parse', 'HEAD') != revision:
        raise RuntimeError(f'Existing dependency has the wrong revision: {path}')
    if git(path, 'diff', 'HEAD', '--name-only', '--ignore-submodules=dirty'):
        raise RuntimeError(f'Dependency has tracked edits: {path}; no changes were overwritten')


def dependency_donor(root, source):
    """Prefer existing sources; consult only registered project-local caches."""
    revision=required_links(source).get(SHMEM.as_posix())
    if revision is None:return root
    existing=(source/SHMEM/'.git').exists()
    if existing:
        check_checkout(source/SHMEM,revision)
        if (source/SDK/'deepep-sdk.json').is_file():return source
    if (root/SHMEM/'.git').exists() and git(root/SHMEM,'rev-parse','HEAD') == revision:
        check_checkout(root/SHMEM,revision)
        if sdk_revision_suitable(root, revision):return root
        candidates=[root]
    else:candidates=[]
    manifests=list((root/'.ablation/wheels').glob('*/manifest.json'))
    manifests+=list((root/STATE/'wheels').glob('*/manifest.json'))
    for manifest in sorted(manifests):
        try:
            info=json.loads(manifest.read_text(encoding='utf-8'))
            candidate=resolve(root,info['source'])
            if not any((root/base).resolve().is_relative_to(root.resolve()) and
                       candidate.is_relative_to((root/base).resolve()) for base in (Path('.ablation/sources'),STATE)):
                continue
            if not (candidate/'.git').exists() or common_dir(candidate)!=common_dir(root):continue
            check_checkout(candidate/SHMEM,revision)
        except (OSError,ValueError,KeyError,RuntimeError,subprocess.CalledProcessError):
            continue
        if candidate not in candidates:candidates.append(candidate)
    if not candidates:
        if existing:return source
        raise RuntimeError(f'Pinned SHMEM source unavailable in {root} or registered local caches; '
                           'initialize third-party/shmem or use --offline-deps. No network fallback.')
    candidates.sort(key=lambda path:not sdk_revision_suitable(path, revision))
    print(f'Reusing SHMEM dependency source from {candidates[0]} (kernel sources unchanged)',flush=True)
    return candidates[0]


def sdk_revision_suitable(source, revision):
    manifest = source/SDK/'deepep-sdk.json'
    if not manifest.is_file():
        return False
    # This is donor selection, not SDK qualification. Unknown/invalid legacy
    # identities still go through the existing complete preparer validation.
    try:
        recorded = json.loads(manifest.read_text(encoding='utf-8')).get('revision')
    except (OSError, ValueError, AttributeError):
        return True
    return recorded is None or recorded == revision


def copy_sdk_assets(donor, source):
    if (donor/JSON_HEADER).is_file() and not (source/JSON_HEADER).exists():
        (source/JSON_HEADER).parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(donor/JSON_HEADER,source/JSON_HEADER)
    if (donor/SDK/'deepep-sdk.json').is_file() and not (source/SDK).exists():
        requested = required_links(source).get(SHMEM.as_posix())
        if requested is not None and not sdk_revision_suitable(donor, requested):
            print(f'Preserving other-revision SDK in {donor}; preparing a separate SDK in {source}', flush=True)
            return
        # scripts/build_shmem.py validates the copied identity and all hashes
        # before pip builds anything. Existing/incomplete SDKs are never replaced.
        shutil.copytree(donor/SDK,source/SDK,symlinks=False)


def copy_local_dependencies(donor, source, *, project=True):
    """Copy only required pinned repositories; preserve optional repositories."""
    if project:donor=dependency_donor(donor,source)
    for relative, revision in required_links(source,project).items():
        origin, target = donor / relative, source / relative
        if (target / '.git').exists():
            check_checkout(target,revision)
        else:
            check_checkout(origin,revision)
            if target.exists() and any(target.iterdir()):
                raise RuntimeError(f'Refuse to overwrite nonempty dependency: {target}')
            subprocess.run(['git', 'clone', '--no-hardlinks', '--no-checkout', str(origin), str(target)], check=True)
            subprocess.run(['git', '-C', str(target), 'checkout', '--no-recurse-submodules', '--detach', revision], check=True)
        copy_local_dependencies(origin, target,project=False)
    if project:copy_sdk_assets(donor,source)


def unpack_dependencies(bundle, source):
    """Read the existing ll_ablation/deps.py bundle format without network access."""
    bundle = bundle.resolve()
    data = json.loads((bundle / 'deps.json').read_text(encoding='utf-8'))
    for entry in data['entries']:
        relative = Path(entry['path'])
        target, artifact = source / relative, bundle / entry['file']
        if relative.is_absolute() or '..' in relative.parts or artifact.parent != bundle:
            raise ValueError('Unsafe offline dependency path')
        if not relative.is_relative_to(SHMEM):
            continue  # Optional CATLASS entries/artifacts are not required.
        if target.resolve() != target or not target.is_relative_to(source) or sha(artifact) != entry['sha256']:
            raise ValueError('Unsafe or corrupt offline dependency')
        parent = target.parent
        while parent != source and not (parent / '.git').exists():
            parent = parent.parent
        if links(parent).get(target.relative_to(parent).as_posix()) != entry['revision']:
            raise ValueError('Offline dependency differs from pinned Git revision')
        if (target / '.git').exists():
            check_checkout(target,entry['revision'])
            continue
        if target.exists() and any(target.iterdir()):
            raise ValueError(f'Refuse to overwrite {target}')
        target.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'init', str(target)], check=True)
        subprocess.run(['git', '-C', str(target), 'fetch', str(artifact), 'HEAD'], check=True)
        subprocess.run(['git', '-C', str(target), 'checkout', '--no-recurse-submodules', '--detach', entry['revision']], check=True)
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


def check_dependencies(source, *, project=True):
    for relative, revision in required_links(source,project).items():
        target = source / relative
        check_checkout(target,revision)
        check_dependencies(target,project=False)
    if project and SHMEM.as_posix() in required_links(source):
        # The standard SDK preparer fully validates installed SDKs without
        # building them. JSON source is needed only for a cold SDK build.
        if (source/SDK/'deepep-sdk.json').is_file():return
        header = source / JSON_HEADER
        if not header.is_file():
            raise RuntimeError('Cold SDK build needs uploaded/local nlohmann json header; no network fallback')
        contents=header.read_text(encoding='utf-8')
        if any(not re.search(r'#define\s+NLOHMANN_JSON_VERSION_'+part+r'\s+'+str(value)+r'\b',contents)
               for part,value in (('MAJOR',3),('MINOR',11),('PATCH',3))):
            raise RuntimeError('Cold SDK build requires nlohmann json3.11.3')


def build(revision, send_cores, *, root=ROOT, workspace=None, offline_deps=None, resume=False):
    revision = git(root, 'rev-parse', revision + '^{commit}')
    identity = dependency_identity(root, revision)
    canonical = manifest_path(root, revision, send_cores)
    if canonical.exists():
        if not resume:
            raise RuntimeError(f'Wheel already exists; use --resume: {canonical}')
        verify_manifest(canonical, revision, send_cores, root)
        print(f'Reusing verified wheel: {canonical}', flush=True)
        return canonical
    source = resolve(root, workspace) if workspace else default_source(root, revision, send_cores).resolve()
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
            # A detached checkout may move between sibling candidates. It never
            # discards user edits; the normal native build below still checks
            # all dependency timestamps/configuration after each switch.
            previous_links = git(source, 'ls-tree', '-r', previous).splitlines()
            target_links = git(root, 'ls-tree', '-r', revision).splitlines()
            if [line for line in previous_links if line.startswith('160000 ')] != [line for line in target_links if line.startswith('160000 ')]:
                raise RuntimeError('Pinned Git dependencies changed; use a fresh build workspace')
            if dependency_identity(source, previous) != identity:
                raise RuntimeError('Dependency identity changed; use a separate matching build workspace')
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
            if SHMEM.as_posix() in required_links(source):
                copy_sdk_assets(dependency_donor(root,source),source)
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
                    **source_protocol_identity(source),
                    **identity,
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
