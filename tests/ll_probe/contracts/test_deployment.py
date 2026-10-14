"""CPU contracts for real rank topology, safe planning, and portable launch paths."""
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import time

import pytest

PROBE = Path(__file__).resolve().parents[1]
REPO = PROBE.parents[1]
sys.path.insert(0, str(PROBE))
import deployment


def config_file(tmp_path, per_host=16, nodes=1, extra=''):
    path = tmp_path / 'deployment.yaml'
    ips = ''.join(f'  - 192.0.2.{10+i}\n' for i in range(nodes))
    path.write_text(f'ips:\n{ips}rank_num_per_server: {per_host}\n'
                    'port: 18679\nshmem_port: 18680\n' + extra)
    return path


@pytest.mark.parametrize('world', [2, 4, 8, 16])
@pytest.mark.parametrize('kind', ['empty', 'tiling', 'comm', 'metadata'])
def test_single_host_real_world(tmp_path, world, kind):
    cfg = deployment.load_config(config_file(tmp_path, world))
    job = deployment.make_plan(cfg, test=kind, run_id='contract')
    assert job['world'] == world
    assert job['local_world_size'] == world
    assert job['ranks'] == list(range(world))
    assert job['node_index'] == 0
    assert job['master_addr'] == '192.0.2.10'
    assert job['master_port'] == 18679
    assert job['endpoint'] == 'tcp://192.0.2.10:18680'
    assert job['status'] != 'unsupported'


@pytest.mark.parametrize('kind', ['empty', 'tiling', 'comm', 'metadata', 'native', 'network'])
def test_32_ranks_are_two_real_hosts(tmp_path, kind):
    cfg = deployment.load_config(config_file(tmp_path, 16, 2))
    jobs = [deployment.make_plan(cfg, node_index=i, test=kind, run_id='same-job') for i in (0, 1)]
    assert [j['ranks'] for j in jobs] == [list(range(16)), list(range(16, 32))]
    assert all(j['world'] == 32 and j['local_world_size'] == 16 for j in jobs)
    assert jobs[0]['endpoint'] == jobs[1]['endpoint']
    assert jobs[0]['master_addr'] == jobs[1]['master_addr']
    assert jobs[0]['run_id'] == jobs[1]['run_id'] == 'same-job'


@pytest.mark.parametrize('world', [2, 4, 8])
def test_native_small_world_is_explicitly_unsupported(tmp_path, world):
    cfg = deployment.load_config(config_file(tmp_path, world))
    assert deployment.make_plan(cfg, test='native', run_id='contract')['status'] == 'unsupported'


@pytest.mark.parametrize('per_host,nodes,node', [(32, 1, 0), (16, 2, 2), (16, 1, -1), (3, 1, 0)])
def test_invalid_topology_rejected(tmp_path, per_host, nodes, node):
    with pytest.raises(ValueError):
        cfg = deployment.load_config(config_file(tmp_path, per_host, nodes))
        deployment.make_plan(cfg, node_index=node, run_id='contract')


def test_a5_validation_is_exactly_one_host_two_ranks(tmp_path):
    cfg = deployment.load_config(config_file(tmp_path, 2))
    job = deployment.make_plan(cfg, validation=True, run_id='validation')
    assert job['ranks'] == [0, 1] and job['world'] == 2
    for per_host, nodes in [(4, 1), (16, 1), (2, 2), (16, 2)]:
        cfg = deployment.load_config(config_file(tmp_path, per_host, nodes))
        with pytest.raises(ValueError):
            deployment.make_plan(cfg, validation=True, run_id='validation')


@pytest.mark.parametrize('extra', ['unknown_option: 1\n', 'world_size: 32\n',
                                  'port: 18680\n', 'shmem_port: 18679\n'])
def test_unknown_or_ambiguous_config_rejected(tmp_path, extra):
    with pytest.raises(ValueError):
        cfg = deployment.load_config(config_file(tmp_path, extra=extra))
        deployment.make_plan(cfg, run_id='contract')


@pytest.mark.parametrize('text', ['[]', '- 192.0.2.10\n- 192.0.2.10\n',
                                  '- $(touch unexpected)\n', '- 0.0.0.0\n'])
def test_yaml_host_validation(tmp_path, text):
    path = tmp_path / 'bad.yaml'
    path.write_text(text)
    with pytest.raises(ValueError):
        deployment.make_plan(deployment.load_config(path))


@pytest.mark.parametrize('run_id', ['../escape', '/absolute', 'with spaces', ''])
def test_run_identity_cannot_escape_output_root(tmp_path, run_id):
    cfg = deployment.load_config(config_file(tmp_path))
    with pytest.raises(ValueError):
        deployment.make_plan(cfg, run_id=run_id)


def test_relative_output_root_cannot_escape_checkout(tmp_path):
    cfg = deployment.load_config(config_file(tmp_path, extra='output_root: ../outside\n'))
    with pytest.raises(ValueError):
        deployment.make_plan(cfg, run_id='contract')


@pytest.fixture
def portable_runner(tmp_path):
    """Copy only source into a differently named checkout; run from unrelated cwd."""
    root = tmp_path / 'renamed project with spaces'
    for source in PROBE.rglob('*'):
        relative = source.relative_to(PROBE)
        if source.is_file() and source.suffix in ('.py', '.sh', '.yaml') and not set(relative.parts) & {'build', 'snapshot', '__pycache__'}:
            target = root / 'tests/ll_probe' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    for source in (REPO / 'scripts').glob('project_*'):
        if source.is_file():
            target = root / 'scripts' / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    native = root / 'tests/elastic/test_ep_low_latency.py'
    native.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO / 'tests/elastic/test_ep_low_latency.py', native)
    cwd = tmp_path / 'unrelated caller'
    cwd.mkdir()
    guard = tmp_path / 'cpu import guard'
    guard.mkdir()
    # Fail hard if planning imports device modules or starts any subprocess,
    # including an accidental device launcher that later returns success.
    (guard / 'sitecustomize.py').write_text(
        'import os,sys\n'
        'class BlockDeviceImports:\n'
        ' def find_spec(self, fullname, path=None, target=None):\n'
        '  if fullname.split(".")[0] in ("torch", "torch_npu", "acl"):\n'
        '   os._exit(97)\n'
        'sys.meta_path.insert(0, BlockDeviceImports())\n'
        'def audit(event,args):\n'
        ' if event in ("subprocess.Popen", "os.system", "os.exec", "os.posix_spawn"):\n'
        '  os._exit(98)\n'
        'sys.addaudithook(audit)\n')
    env = os.environ.copy()
    for key in ('PROBE_ROOT', 'DEEPEP_PROJECT_ROOT', 'RANK', 'LOCAL_RANK', 'WORLD_SIZE',
                'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
        env.pop(key, None)
    env.update(PYTHONPATH=str(guard), PYTHONDONTWRITEBYTECODE='1')
    cfg = config_file(root, 16, 2)
    return root, cwd, env, cfg


def run_portable(fixture, *args):
    root, cwd, env, cfg = fixture
    return subprocess.run([sys.executable, str(root / 'tests/ll_probe/run.py'),
                           '--config', cfg.name, *args], cwd=cwd, env=env,
                          text=True, capture_output=True, timeout=20)


@pytest.mark.parametrize('kind', ['empty', 'tiling', 'comm', 'metadata', 'native', 'network'])
def test_plan_is_cpu_only_from_spaced_checkout(portable_runner, kind):
    result = run_portable(portable_runner, '--test', kind, '--node-index', '1', '--plan')
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan['world'] == 32 and plan['ranks'] == list(range(16, 32))
    assert plan['endpoint'] == 'tcp://192.0.2.10:18680'
    command = plan['command']
    if kind != 'network':
        for option, value in [('--nnodes', '2'), ('--nproc-per-node', '16'), ('--node-rank', '1')]:
            assert command[command.index(option) + 1] == value
    assert any('renamed project with spaces' in arg for arg in command)


@pytest.mark.parametrize('argument', ['--num-processes', '--nproc-per-node', '--endpoint', '--world-size'])
def test_unknown_arguments_cannot_override_topology(portable_runner, argument):
    result = run_portable(portable_runner, '--test', 'comm', '--plan', argument, '123')
    assert result.returncode not in (0, 97, 98), result.stdout + result.stderr


def test_default_without_execution_mode_is_cpu_plan(portable_runner):
    result = run_portable(portable_runner, '--test', 'comm', '--run-id', 'no-authorization')
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)['run_id'] == 'no-authorization'


def test_validation_cannot_be_target_mode(portable_runner):
    result = run_portable(portable_runner, '--test', 'comm', '--validation', '--target-run', '--plan')
    assert result.returncode not in (0, 97, 98), result.stdout + result.stderr


@pytest.mark.parametrize('kind', ['empty', 'tiling', 'comm', 'metadata'])
def test_validation_defaults_are_tiny(portable_runner, kind):
    root = portable_runner[0]
    config_file(root, 2)
    result = run_portable(portable_runner, '--test', kind, '--validation', '--plan')
    assert result.returncode == 0, result.stderr
    command = json.loads(result.stdout)['command']
    for option, value in [('--warmup', '2'), ('--samples', '3'), ('--tail', '2'),
                          ('--delay-rows', '64'), ('--delay-cols', '64')]:
        assert command[command.index(option) + 1] == value


@pytest.mark.parametrize('option,value', [('--delay-rows', '65'), ('--delay-cols', '65'),
                                         ('--samples', '4'), ('--warmup', '3')])
def test_validation_cannot_expand_reference_load(portable_runner, option, value):
    config_file(portable_runner[0], 2)
    result = run_portable(portable_runner, '--test', 'comm', '--validation', '--plan', '--', option, value)
    assert result.returncode not in (0, 97, 98), result.stdout + result.stderr


def test_run_directory_symlink_cannot_escape_checkout(tmp_path, monkeypatch):
    # Model a checkout and its sibling explicitly: pytest's base directory may
    # itself be inside the real checkout, so tmp_path is not necessarily outside.
    checkout = tmp_path / 'fake checkout'
    output_root = checkout / 'results'
    output_root.mkdir(parents=True)
    outside = tmp_path / 'outside checkout'
    outside.mkdir()
    (output_root / 'escaped-run').symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(deployment, 'REPO_ROOT', checkout)
    cfg = deployment.load_config(config_file(tmp_path))
    cfg['output_root'] = str(output_root)
    with pytest.raises(ValueError):
        deployment.make_plan(cfg, run_id='escaped-run')


@pytest.mark.parametrize('argument', ['--world-size', '--endpoint', '--num-processes', '--output-dir', '--kind'])
def test_forwarded_options_cannot_override_deployment(portable_runner, argument):
    result = run_portable(portable_runner, '--test', 'comm', '--plan', '--', argument, '123')
    assert result.returncode not in (0, 97, 98), result.stdout + result.stderr


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux process-group supervision')
@pytest.mark.parametrize('timeout_case', [False, True])
def test_supervisor_cleans_descendants_on_failure_and_timeout(tmp_path, timeout_case):
    supervise = runpy.run_path(str(PROBE / 'run.py'))['supervise']
    child_pid_path = tmp_path / 'child.pid'
    # Only CPU Python processes. A descendant ignores TERM so that failure to
    # clean the entire process group is observable after its leader exits.
    child = 'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)'
    leader = ('import subprocess,sys,time,pathlib\n'
              f'p=subprocess.Popen([sys.executable,"-c",{child!r}])\n'
              f'pathlib.Path({str(child_pid_path)!r}).write_text(str(p.pid))\n'
              + ('time.sleep(60)\n' if timeout_case else 'time.sleep(.1); sys.exit(7)\n'))
    try:
        if timeout_case:
            with pytest.raises(subprocess.TimeoutExpired):
                supervise([sys.executable, '-c', leader], os.environ.copy(), tmp_path, 1)
        else:
            assert supervise([sys.executable, '-c', leader], os.environ.copy(), tmp_path, 5) == 7
        assert child_pid_path.exists(), 'CPU descendant did not start'
        pid = int(child_pid_path.read_text())
        def alive():
            try:
                state = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0]
                return state not in ('Z', 'X')
            except FileNotFoundError:
                return False
        deadline = time.monotonic() + 2
        while alive() and time.monotonic() < deadline:
            time.sleep(.02)
        assert not alive(), 'Supervisor left its CPU descendant running'
    finally:
        # Ensure even a broken implementation does not leave the fixture alive.
        if child_pid_path.exists():
            try:
                os.kill(int(child_pid_path.read_text()), 9)
            except ProcessLookupError:
                pass
