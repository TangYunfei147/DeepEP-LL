"""CPU-only ownership/recovery checks; these tests never terminate a process."""
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import pytest


MODULE = Path(__file__).with_name('build.py').resolve()
_spec = importlib.util.spec_from_file_location('control_resume_lock_build', MODULE)
build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build)


def owner(pid, **changes):
    result = dict(schema=1, pid=pid, host=socket.gethostname(), start=None, token='a' * 32)
    result.update(changes)
    return json.dumps(result).encode()


def test_default_build_lock_never_recovers_existing_marker(tmp_path, monkeypatch):
    path = tmp_path / 'build.lock'
    original = b'pid=99999\n'
    path.write_bytes(original)
    monkeypatch.setattr(build, '_process_identity', lambda pid: pytest.fail('default must not probe old owner'))
    with pytest.raises(FileExistsError):
        build.acquire_lock(path)
    assert path.read_bytes() == original


@pytest.mark.parametrize('contents', [b'pid=99999\n', owner(99999)])
def test_explicit_resume_recovers_confirmed_dead_owner(tmp_path, monkeypatch, contents):
    path = tmp_path / 'suite.lock'
    path.write_bytes(contents)
    monkeypatch.setattr(build, '_process_identity', lambda pid: ('dead', None))
    release = build.acquire_lock(path, recover_dead=True)
    try:
        current = json.loads(path.read_text(encoding='utf-8'))
        assert current['pid'] == os.getpid() and current['host'] == socket.gethostname()
        assert len(current['token']) == 32
        with pytest.raises(FileExistsError):
            build.acquire_lock(path, recover_dead=True)
    finally:
        release()
    assert not path.exists()
    assert path.with_name('suite.lock.guard').is_file()
    release()  # Repeated finally/atexit cleanup must be harmless.


@pytest.mark.parametrize('state', ['alive', 'unknown'])
def test_live_or_inaccessible_owner_is_preserved(tmp_path, monkeypatch, state):
    path = tmp_path / 'suite.lock'
    original = owner(99999, start='old process incarnation')
    path.write_bytes(original)
    # A different live start identity still does not authorize PID recycling.
    monkeypatch.setattr(build, '_process_identity', lambda pid: (state, 'new process incarnation'))
    with pytest.raises(RuntimeError, match='preserved'):
        build.acquire_lock(path, recover_dead=True)
    assert path.read_bytes() == original


@pytest.mark.parametrize('contents', [b'', b'pid=0\n', b'pid=-3\n', b'pid=no\n', b'pid=99999\nextra=yes',
                                      b'\xff', b'{}', b'[]', owner(True), owner(99999, token='wrong')])
def test_malformed_owner_never_triggers_recovery(tmp_path, monkeypatch, contents):
    path = tmp_path / 'suite.lock'
    path.write_bytes(contents)
    monkeypatch.setattr(build, '_process_identity', lambda pid: pytest.fail('malformed lock must not probe a PID'))
    with pytest.raises(RuntimeError, match='malformed'):
        build.acquire_lock(path, recover_dead=True)
    assert path.read_bytes() == contents


def test_foreign_host_marker_is_preserved_even_when_pid_not_local(tmp_path, monkeypatch):
    path = tmp_path / 'suite.lock'
    original = owner(99999, host=socket.gethostname() + '-different-host')
    path.write_bytes(original)
    monkeypatch.setattr(build, '_process_identity', lambda pid: pytest.fail('foreign owner must not be queried locally'))
    with pytest.raises(RuntimeError, match='another host'):
        build.acquire_lock(path, recover_dead=True)
    assert path.read_bytes() == original


def test_owner_change_during_liveness_check_prevents_recovery(tmp_path, monkeypatch):
    path = tmp_path / 'suite.lock'
    path.write_bytes(b'pid=99999\n')
    replacement = b'pid=12345\n'
    def changed(pid):
        path.write_bytes(replacement)
        return 'dead', None
    monkeypatch.setattr(build, '_process_identity', changed)
    with pytest.raises(RuntimeError, match='changed before recovery'):
        build.acquire_lock(path, recover_dead=True)
    assert path.read_bytes() == replacement


def test_release_cannot_remove_replacement_owner(tmp_path):
    path = tmp_path / 'suite.lock'
    release = build.acquire_lock(path)
    replacement = owner(os.getpid(), token='b' * 32)
    path.write_bytes(replacement)
    release()
    release()
    assert path.read_bytes() == replacement


def test_guard_still_excludes_other_owner_if_marker_was_removed(tmp_path):
    path = tmp_path / 'suite.lock'
    release = build.acquire_lock(path)
    try:
        path.unlink()
        with pytest.raises(FileExistsError, match='guard'):
            build.acquire_lock(path, recover_dead=True)
    finally:
        release()
    later = build.acquire_lock(path)
    later()


def test_real_current_process_is_queried_without_signalling_it(monkeypatch):
    if os.name == 'nt':
        monkeypatch.setattr(build.os, 'kill', lambda *args: pytest.fail('Windows process query must not use os.kill'))
    state, started = build._process_identity(os.getpid())
    assert state == 'alive'
    assert started is None or isinstance(started, str)


CHILD = r'''
import contextlib, importlib.util, io, os, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location('lock_child_build', sys.argv[1])
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)
try:
    with contextlib.redirect_stdout(io.StringIO()):
        release = build.acquire_lock(Path(sys.argv[2]), recover_dead=sys.argv[3]=='recover')
except (FileExistsError, RuntimeError):
    print('blocked', flush=True)
else:
    print('held:' + str(os.getpid()), flush=True)
    if sys.argv[3]=='exit-without-cleanup':
        os._exit(0)
    sys.stdin.readline()
    release()
'''


def abandoned_lock(path):
    completed = subprocess.run([sys.executable, '-c', CHILD, str(MODULE), str(path), 'exit-without-cleanup'],
                               capture_output=True, text=True, timeout=10)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip().startswith('held:')
    previous = json.loads(path.read_text(encoding='utf-8'))
    assert build._process_identity(previous['pid'])[0] == 'dead'
    return previous


def test_abrupt_process_exit_releases_guard_for_explicit_recovery(tmp_path):
    path = tmp_path / 'suite.lock'
    old = abandoned_lock(path)
    with pytest.raises(FileExistsError):
        build.acquire_lock(path)
    release = build.acquire_lock(path, recover_dead=True)
    try:
        new = json.loads(path.read_text(encoding='utf-8'))
        assert new['pid'] == os.getpid() and new['token'] != old['token']
    finally:
        release()


def test_two_resume_processes_cannot_both_own_recovered_lock(tmp_path):
    path = tmp_path / 'suite.lock'
    abandoned_lock(path)
    children = []
    try:
        for _ in range(2):
            children.append(subprocess.Popen([sys.executable, '-c', CHILD, str(MODULE), str(path), 'recover'],
                                             stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE, text=True))
        answers = [child.stdout.readline().strip() for child in children]
        assert sum(answer == 'blocked' for answer in answers) == 1
        winners = [answer for answer in answers if answer.startswith('held:')]
        assert len(winners) == 1
        # Windows venv's redirector may be the Popen PID; the lock belongs to
        # its actual Python worker, whose PID is returned explicitly.
        assert json.loads(path.read_text(encoding='utf-8'))['pid'] == int(winners[0].split(':')[1])
    finally:
        for child in children:
            try:
                child.stdin.write('\n')
                child.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
            try:
                child.stdin.close()
            except OSError:
                pass
        for child in children:
            assert child.wait(timeout=10) == 0, child.stderr.read()
    assert not path.exists()
