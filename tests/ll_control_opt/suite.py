"""Build every committed candidate, then run the unchanged LL ST sampling protocol."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import socket
import statistics
import subprocess
import sys
import time

_spec = importlib.util.spec_from_file_location('ll_control_build', Path(__file__).with_name('build.py'))
build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build)
ROOT = build.ROOT
SHAPE = dict(B=256, M=256, H=3584, num_topk=8, num_experts=384,
             dtype='fp8', do_cpu_sync=False, fp8_scale_dtype='int32')


def safe_name(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value):
        raise ValueError(f'Unsafe name: {value!r}')
    return value


def commits(path, root=ROOT):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    items = data['commits'] if isinstance(data, dict) else data
    schema = data.get('schema', 1) if isinstance(data, dict) else 1
    if type(schema) is not int or schema not in (1, 2):
        raise ValueError('Unsupported commit manifest schema; expected1 or2')
    if not isinstance(items, list) or len(items) < 2 or any(not isinstance(item, dict) for item in items):
        raise ValueError('Manifest must contain baseline and candidate commit objects')
    if schema == 1 and ((isinstance(data, dict) and 'comparison_base' in data) or any('comparison_base' in item for item in items)):
        raise ValueError('Explicit comparison_base requires manifest schema2')
    result, labels, revisions = [], set(), set()
    for item in items:
        label = safe_name(item['label'])
        revision = build.git(root, 'rev-parse', item['revision'] + '^{commit}')
        if label in labels or revision in revisions:
            raise ValueError('Commit labels and revisions must be unique')
        if schema == 1 and result and subprocess.run(['git', '-C', str(root), 'merge-base', '--is-ancestor', result[-1]['revision'], revision]).returncode:
            raise ValueError('Candidates must be a cumulative, ancestor-ordered commit chain')
        labels.add(label)
        revisions.add(revision)
        result.append(dict(label=label, revision=revision))
    if result[0]['label'] != 'baseline':
        raise ValueError('The first commit must have label baseline')
    by_label = {entry['label']: entry for entry in result}
    by_revision = {entry['revision']: entry for entry in result}
    for index, (entry, item) in enumerate(zip(result, items)):
        if schema == 1:
            reference = result[max(0, index-1)]
        else:
            value = item.get('comparison_base', data.get('comparison_base', result[0]['revision']))
            if not isinstance(value, str) or not value:
                raise ValueError('comparison_base must be a measured label or Git revision')
            revision = by_label[value]['revision'] if value in by_label else build.git(root, 'rev-parse', value+'^{commit}')
            if revision not in by_revision:
                raise ValueError(f'Comparison reference for {entry["label"]} must be measured in this manifest')
            reference = by_revision[revision]
            if (index == 0) != (reference['revision'] == entry['revision']):
                raise ValueError('Only baseline may reference itself; baseline must use its own revision')
        entry.update(comparison_base=reference['revision'], comparison_label=reference['label'],
                     comparison_kind='baseline' if index == 0 else ('cumulative_parent' if schema == 1 else 'explicit_reference'))
    return result


def execution_entries(entries, round_id, order):
    if order not in ('manifest', 'alternating'):
        raise ValueError('order must be manifest or alternating')
    return list(reversed(entries)) if order == 'alternating' and round_id % 2 == 0 else list(entries)


def device_ids(value):
    parts = (value or '').split(',')
    if len(parts) != 16 or any(not part.isdecimal() for part in parts) or len(set(map(int, parts))) != 16:
        raise ValueError('Supply exactly16 distinct local device IDs with --devices or ASCEND_RT_VISIBLE_DEVICES')
    return parts


def require_idle(devices):
    output = subprocess.check_output(['npu-smi', 'info'], text=True)
    empty = {int(value) for value in re.findall(r'No running processes found in NPU\s+(\d+)', output)}
    if not set(map(int, device_ids(devices))).issubset(empty):
        raise RuntimeError('Selected NPU devices are busy or not reported idle: no launch; no processes were stopped')
    return output


def st_command(python, args, directory, root=ROOT):
    return [str(python), '-I', str(root / 'tests/elastic/test_ep_low_latency.py'),
            '--num-processes', '16', '--num-nodes', str(args.num_nodes), '--node-rank', str(args.node_rank),
            '--master-addr', args.master_addr, '--master-port', str(args.master_port),
            '--shmem-port', str(args.shmem_port), '--num-tokens', '256',
            '--num-max-tokens-per-rank', '256', '--hidden', '3584', '--num-topk', '8',
            '--num-experts', '384', '--dtype', 'fp8', '--do-cpu-sync', '0',
            '--output-dir', str(directory.parent), '--run-name', directory.name]


def validate_summary(directory, revision, send_cores, num_nodes):
    """Use summary.csv mean_kernel_us; reject mixed cases or incomplete stable tails."""
    directory = Path(directory)
    report = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
    world = 16 * num_nodes
    if report.get('status') != 'completed' or report.get('world_size') != world or report.get('case_count') != 1 or report.get('result_count') != 1:
        raise ValueError(f'Incomplete or mixed ST run: {directory}')
    config = report['config']
    for key, value in dict(warmup=10, num_iters=100, perf_tail=50, route='normal', seed=0,
                           skip_check=False, skip_perf_test=False, dispatch_only=False, ll_stage_timer=False).items():
        if config.get(key) != value:
            raise ValueError(f'Unexpected sampling setting {key}: {config.get(key)!r}')
    result = report['results'][0]
    if result['case_id'] != 0 or result['route'] != 'normal' or result['config'] != SHAPE or result.get('correctness') != 'passed':
        raise ValueError('Unexpected case, route, or shape')
    identity = result.get('comparison', {})
    for key, value in dict(source_revision=revision, control_revision=revision, send_cores=send_cores, recv_cores=56-send_cores,
                           active_aiv=56, logical_group=4, transport='mte', stage_timer=False,
                           sampling_protocol='exp-reduce-stable-tail-v2', world_size=world).items():
        if identity.get(key) != value:
            raise ValueError(f'Runtime identity mismatch for {key}')
    with (directory / 'summary.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 2 or {row['operation'] for row in rows} != {'dispatch', 'combine'}:
        raise ValueError('Expected exactly one dispatch and one combine row')
    values = {}
    for row in rows:
        operation = row['operation']
        if row.get('summary_statistics_scope') != 'pooled_rank_tail_kernel_samples' or int(row['perf_tail']) != 50 or int(row['kernel_samples']) != world * 50:
            raise ValueError('Missing samples or wrong summary metric scope')
        if row['source_revision'] != revision or int(row['case_id']) != 0 or row['route'] != 'normal':
            raise ValueError('Mixed revision/case/route in summary.csv')
        perf = result[operation]
        ranks = perf['ranks']
        if len(ranks) != world or {r['rank'] for r in ranks} != set(range(world)):
            raise ValueError('Missing or duplicate performance ranks')
        pooled = []
        for rank in ranks:
            raw = rank['kernel_us']
            if rank['samples'] != 50 or len(raw) != 100 or any(not math.isfinite(v) or v <= 0 for v in raw):
                raise ValueError('Incomplete, nonfinite, or nonpositive raw kernel samples')
            pooled.extend(raw[-50:])
        value = float(row['mean_kernel_us'])
        if not math.isfinite(value) or value <= 0 or not math.isclose(value, statistics.mean(pooled), rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError('summary.csv mean_kernel_us disagrees with the selected raw samples')
        if not math.isclose(value, perf['mean_kernel_us'], rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError('JSON/CSV performance mismatch')
        values[operation] = value
    return values


def comparisons(entries, measured, order='alternating'):
    rows = []
    for round_id, values in sorted(measured.items()):
        if set(values) != {entry['label'] for entry in entries}:
            raise ValueError('Every comparison round must contain every manifest entry')
        positions = {entry['label']: position for position, entry in enumerate(execution_entries(entries, round_id, order), 1)}
        for index, entry in enumerate(entries):
            for operation in ('dispatch', 'combine'):
                value = values[entry['label']][operation]
                baseline = values[entries[0]['label']][operation]
                reference = values[entry['comparison_label']][operation]
                previous = entries[index-1] if index else None
                previous_value = values[previous['label']][operation] if previous else None
                cumulative = entry['comparison_kind'] == 'cumulative_parent'
                rows.append(dict(round=round_id, label=entry['label'], revision=entry['revision'], operation=operation,
                                 execution_position=positions[entry['label']], comparison_kind=entry['comparison_kind'],
                                 baseline_label=entries[0]['label'], baseline_revision=entries[0]['revision'],
                                 mean_kernel_us=value, baseline_delta_us=value-baseline,
                                 baseline_delta_pct=100*(value/baseline-1),
                                 reference_label=entry['comparison_label'], reference_revision=entry['comparison_base'],
                                 reference_delta_us=value-reference, reference_delta_pct=100*(value/reference-1),
                                 parent_delta_us=value-reference if cumulative else None,
                                 parent_delta_pct=100*(value/reference-1) if cumulative else None,
                                 previous_manifest_label=previous['label'] if previous else None,
                                 previous_manifest_revision=previous['revision'] if previous else None,
                                 previous_manifest_delta_us=value-previous_value if previous else None,
                                 previous_manifest_delta_pct=100*(value/previous_value-1) if previous else None))
    return rows


def run_environment(args, entry, manifest, root=ROOT):
    env = build.environment(root, args.send_cores)
    for key in ('CUDA_VISIBLE_DEVICES', 'ASCEND_RT_VISIBLE_DEVICES', 'ASCEND_VISIBLE_DEVICES', 'PROBE_VISIBLE_DEVICES'):
        env[key] = args.devices
    env.update(DEEPEP_LL_RUN_REVISION=entry['revision'], DEEPEP_LL_RUN_DIRTY='False',
               DEEPEP_LL_CONTROL_REVISION=entry['revision'], DEEPEP_LL_EXPECT_IMPLEMENTATION='g4-r0-prefix',
               DEEPEP_LL_BACKEND='mte')
    for key in ('LL_ABLATION_MODE', 'LL_ABLATION_MANIFEST', 'RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE'):
        env.pop(key, None)
    sdk = build.resolve(root, manifest['sdk'])
    env['LD_LIBRARY_PATH'] = str(sdk / 'lib') + ':' + env.get('LD_LIBRARY_PATH', '')
    return env


def preflight_code(manifest, send_cores):
    return ('from deep_ep._native import require_native; import hashlib,re; from pathlib import Path; '
            'n=require_native("control-path candidate preflight"); '
            f'assert hashlib.sha256(Path(n.__file__).read_bytes()).hexdigest()=={manifest["native_sha256"]!r}; '
            'assert not bool(n.low_latency_stage_timer_enabled); '
            f'assert n.low_latency_core_protocol=={19000+send_cores}; '
            'assert n.low_latency_implementation=="g4-r0-prefix"; '
            'g=n.low_latency_ranks_per_group; assert (g(16) if callable(g) else g)==4; '
            'paths={re.sub(r"\\\\([0-7]{3})",lambda m:chr(int(m.group(1),8)),line.split(None,5)[5]) for line in Path("/proc/self/maps").read_text().splitlines() if len(line.split(None,5))==6}; '
            'kernels=[Path(p) for p in paths if p.endswith("/libdeepep_kernels.so")]; '
            'assert len(kernels)==1, "Expected exactly one loaded DeepEP kernel library"; '
            f'assert hashlib.sha256(kernels[0].read_bytes()).hexdigest()=={manifest["kernel_sha256"]!r}; '
            f'assert hashlib.sha256(Path(n.shmem_library_path()).read_bytes()).hexdigest()=={manifest["sdk_library_sha256"]!r}')


def control_port(args):
    if args.control_port is not None:
        port = args.control_port
    else:
        port = args.master_port + 1
        if port == args.shmem_port:
            port += 1
    if not 0 < port < 65536 or port in (args.master_port, args.shmem_port):
        raise ValueError('Choose a --control-port distinct from MASTER_PORT and SHMEM_PORT in1..65535')
    return port


def _receive_json(connection, deadline):
    data = bytearray()
    while not data.endswith(b'\n'):
        connection.settimeout(max(0.001, deadline-time.monotonic()))
        block = connection.recv(4096)
        if not block:
            raise RuntimeError('Readiness peer disconnected')
        data.extend(block)
        if len(data) > 16384:
            raise RuntimeError('Oversized readiness message')
    return json.loads(data.decode('utf-8'))


def _send_json(connection, value):
    connection.sendall((json.dumps(value, sort_keys=True)+'\n').encode('utf-8'))


def ready_barrier(args, identity, error=None):
    """CPU-only rendezvous before ST: slower hosts may still be compiling."""
    if args.num_nodes == 1:
        if error:
            raise RuntimeError(error)
        return
    port = control_port(args)
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    deadline = time.monotonic() + args.ready_timeout
    print(f'Waiting for {args.num_nodes} prepared hosts on {args.master_addr}:{port}, timeout={args.ready_timeout}s', flush=True)
    if args.node_rank:
        connection = None
        while time.monotonic() < deadline:
            try:
                connection = socket.create_connection((args.master_addr, port), timeout=min(2, max(0.01, deadline-time.monotonic())))
                break
            except OSError:
                time.sleep(min(1, max(0, deadline-time.monotonic())))
        if connection is None:
            raise RuntimeError('Readiness coordinator unavailable; ST was not launched')
        with connection:
            _send_json(connection, dict(node_rank=args.node_rank, fingerprint=fingerprint, error=error))
            reply = _receive_json(connection, deadline)
            if reply != dict(ready=True, fingerprint=fingerprint):
                raise RuntimeError(f'Host readiness rejected: {reply}')
            _send_json(connection, dict(ack=args.node_rank))
        return
    peers = {}
    failures = {0: error} if error else {}
    family = socket.AF_INET6 if ':' in args.master_addr else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('::' if family == socket.AF_INET6 else '0.0.0.0', port))
        server.listen(args.num_nodes)
        try:
            while len(peers) < args.num_nodes-1:
                server.settimeout(max(0.001, deadline-time.monotonic()))
                connection, _ = server.accept()
                try:
                    hello = _receive_json(connection, deadline)
                    rank = hello.get('node_rank')
                    if type(rank) is not int or not 0 < rank < args.num_nodes or rank in peers or hello.get('fingerprint') != fingerprint:
                        _send_json(connection, dict(error='Node/commit/split/round/run identity mismatch'))
                        raise RuntimeError('Hosts have mismatched run identities; ST was not launched')
                    peers[rank] = connection
                    if hello.get('error'):
                        failures[rank] = str(hello['error'])[:2048]
                except BaseException:
                    connection.close()
                    raise
            # No new connection may enter this completed generation. The next
            # rendezvous reopens the listener; early clients retry normally.
            server.close()
            if failures:
                raise RuntimeError(f'Hosts failed readiness; no ST launch: {failures}')
            for connection in peers.values():
                _send_json(connection, dict(ready=True, fingerprint=fingerprint))
            for rank, connection in peers.items():
                if _receive_json(connection, deadline) != dict(ack=rank):
                    raise RuntimeError('Missing readiness acknowledgement')
        except BaseException as exception:
            for connection in peers.values():
                try:
                    _send_json(connection, dict(error='Readiness failed; ST was not launched: '+str(exception)[:2048]))
                except OSError:
                    pass
            raise
        finally:
            for connection in peers.values():
                connection.close()


def run_logged(command, path, env):
    with Path(path).open('x', encoding='utf-8') as log:
        log.write(json.dumps(command)+'\n')
        log.flush()
        with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding='utf-8', errors='replace', bufsize=1) as process:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument('--commits', type=Path, default=Path('tests/ll_control_opt/commits.json'))
    result.add_argument('--devices', default=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'))
    result.add_argument('--send-cores', type=int, default=30)
    result.add_argument('--run-prefix')
    result.add_argument('--output-dir', type=Path, default=Path('results/ll-control-opt'))
    result.add_argument('--num-nodes', type=int, default=os.environ.get('NNODES'))
    result.add_argument('--node-rank', type=int, default=os.environ.get('NODE_RANK'))
    result.add_argument('--master-addr', default=os.environ.get('MASTER_ADDR'))
    result.add_argument('--master-port', type=int, default=os.environ.get('MASTER_PORT'))
    result.add_argument('--shmem-port', type=int, default=os.environ.get('SHMEM_PORT'))
    result.add_argument('--control-port', type=int, help='CPU readiness port; default first port after MASTER_PORT that is not SHMEM_PORT')
    result.add_argument('--ready-timeout', type=float, default=3600, help='Seconds to wait for all machines to finish preparation before each ST launch')
    result.add_argument('--rounds', type=int, default=1, help='Repeat complete commit sweeps; no changes to each ST sampling protocol')
    result.add_argument('--order', choices=('alternating', 'manifest'), default='alternating',
                        help='alternating: odd rounds use manifest order, even rounds reverse it; manifest: repeat fixed order')
    modes = result.add_mutually_exclusive_group()
    modes.add_argument('--build-only', action='store_true')
    modes.add_argument('--run-only', action='store_true')
    result.add_argument('--resume', action='store_true', help='Reuse verified wheels; never skip or overwrite old performance runs')
    result.add_argument('--build-workspace', type=Path)
    result.add_argument('--offline-deps', type=Path)
    result.add_argument('--plan', action='store_true')
    return result


def make_plan(args, root=ROOT):
    if not 1 <= args.send_cores <= 55 or args.rounds < 1 or not math.isfinite(args.ready_timeout) or args.ready_timeout <= 0:
        raise ValueError('send-cores must be 1..55, rounds must be positive')
    entries = commits(build.resolve(root, args.commits), root)
    if not args.build_only:
        device_ids(args.devices)
        safe_name(args.run_prefix or '')
        if args.num_nodes is None or args.node_rank is None or args.num_nodes < 1 or not 0 <= args.node_rank < args.num_nodes:
            raise ValueError('Explicit NNODES and NODE_RANK required; no implicit single-machine launch')
        if not args.master_addr or any(port is None or not 0 < port < 65536 for port in (args.master_port, args.shmem_port)):
            raise ValueError('Explicit MASTER_ADDR, MASTER_PORT, and SHMEM_PORT required')
        if args.num_nodes not in (1, 2, 4, 8):
            raise ValueError('This native LL suite supports EP16/32/64/128: NNODES must be1,2,4,8')
        if args.num_nodes > 1:
            control_port(args)
    phases = dict(build=[], install=[], run=[])
    for entry in entries:
        path = build.manifest_path(root, entry['revision'], args.send_cores)
        if not args.run_only:
            command = [sys.executable, str(root / 'tests/ll_control_opt/build.py'), '--revision', entry['revision'],
                       '--send-cores', str(args.send_cores)]
            if args.resume:
                command.append('--resume')
            if args.build_workspace:
                command += ['--build-workspace', str(build.resolve(root, args.build_workspace))]
            if args.offline_deps:
                command += ['--offline-deps', str(build.resolve(root, args.offline_deps))]
            phases['build'].append(command)
        if args.build_only:
            continue
        venv = root / build.STATE / 'envs' / f'{entry["revision"]}-s{args.send_cores}'
        python = venv / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
        phases['install'].append(dict(manifest=str(path), environment=str(venv), python=str(python)))
        for round_id in range(1, args.rounds+1):
            directory = build.resolve(root, args.output_dir) / args.run_prefix / entry['label'] / f'round-{round_id:03}'
            phases['run'].append(dict(label=entry['label'], revision=entry['revision'], round=round_id,
                                      directory=str(directory), command=st_command(python, args, directory, root)))
    positions = {round_id: {entry['label']: index for index, entry in enumerate(execution_entries(entries, round_id, args.order), 1)}
                 for round_id in range(1, args.rounds+1)}
    for item in phases['run']:
        item['execution_position'] = positions[item['round']][item['label']]
    phases['run'].sort(key=lambda item: (item['round'], item['execution_position']))
    return dict(commits=entries, phases=phases, metric='summary.csv: pooled mean_kernel_us; dispatch and combine separately',
                sampling=dict(warmup=10, num_iters=100, perf_tail=50), send_cores=args.send_cores,
                recv_cores=56-args.send_cores, timer='OFF',
                order=args.order, rounds=args.rounds, comparison_schema=2,
                control_protocol='ll-control-ready-v1',
                control_port=control_port(args) if not args.build_only and args.num_nodes>1 else None)


def main():
    cli = parser()
    args = cli.parse_args()
    try:
        plan = make_plan(args)
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        cli.exit(2, str(error) + '\n')
    if args.plan:
        print(json.dumps(plan, indent=2))
        return
    release = build.acquire_lock(ROOT / build.STATE / 'suite.lock')
    try:
        # Fail before any launch if any previous output would be reused.
        for item in plan['phases']['run']:
            if (args.node_rank == 0 and Path(item['directory']).exists()) or any(Path(item['directory'] + f'.node-{args.node_rank}' + suffix).exists() for suffix in ('.done.json', '.log', '.idle.txt')):
                raise RuntimeError('Previous performance output exists; choose a fresh --run-prefix. --resume only reuses wheels.')
        for command in plan['phases']['build']:
            subprocess.run(command, cwd=ROOT, check=True)
        if args.build_only:
            print('All wheels are built. No NPU devices were used.', flush=True)
            return
        manifests = {}
        for entry, install in zip(plan['commits'], plan['phases']['install']):
            path = Path(install['manifest'])
            info = build.verify_manifest(path, entry['revision'], args.send_cores)
            manifests[entry['label']] = info
            venv = Path(install['environment'])
            if not venv.exists():
                subprocess.run([sys.executable, '-m', 'venv', '--system-site-packages', str(venv)], check=True)
            if not (venv / 'pyvenv.cfg').is_file() or not Path(install['python']).is_file():
                raise RuntimeError(f'Incomplete local virtual environment: {venv}; preserve it and choose a fresh checkout')
            env = run_environment(args, entry, info)
            wheel = build.resolve(path.parent, info['wheel'])
            subprocess.run([install['python'], '-I', '-m', 'pip', 'install', '--no-index', '--no-deps', '--force-reinstall', str(wheel)], cwd=venv, env=env, check=True)
            subprocess.run([install['python'], '-I', '-c', preflight_code(info, args.send_cores)], cwd=venv, env=env, check=True)
        base = build.resolve(ROOT, args.output_dir) / args.run_prefix
        base.mkdir(parents=True, exist_ok=True)
        build.write_json(base / f'suite-plan.node-{args.node_rank}.json', plan)
        measured = {}
        for item in plan['phases']['run']:
            directory = Path(item['directory'])
            directory.parent.mkdir(parents=True, exist_ok=True)
            env = run_environment(args, item, manifests[item['label']])
            identity = dict(protocol=plan['control_protocol'], commits=plan['commits'], run_prefix=args.run_prefix,
                            label=item['label'], revision=item['revision'], round=item['round'], rounds=args.rounds,
                            order=plan['order'], execution_position=item['execution_position'],
                            send_cores=args.send_cores, num_nodes=args.num_nodes, shape=SHAPE)
            ready_barrier(args, dict(identity, phase='prepared'))
            idle_error = None
            try:
                idle = require_idle(args.devices)
                (directory.parent / f'{directory.name}.node-{args.node_rank}.idle.txt').write_text(idle, encoding='utf-8')
            except (OSError, RuntimeError, subprocess.CalledProcessError) as exception:
                idle_error = str(exception)[:2048]
            ready_barrier(args, dict(identity, phase='idle'), error=idle_error)
            print(f'Run {item["label"]}, round{item["round"]}, node{args.node_rank}; stdout remains visible.', flush=True)
            run_logged(item['command'], str(directory)+f'.node-{args.node_rank}.log', env)
            build.write_json(Path(str(directory) + f'.node-{args.node_rank}.done.json'), dict(revision=item['revision'], exit_code=0))
            if args.node_rank == 0:
                values = validate_summary(directory, item['revision'], args.send_cores, args.num_nodes)
                measured.setdefault(item['round'], {})[item['label']] = values
        if args.node_rank != 0:
            print('All ST commands completed. Aggregated summary is written by node0.', flush=True)
            return
        rows = comparisons(plan['commits'], measured, plan['order'])
        with (base / 'comparison.csv').open('x', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        build.write_json(base / 'comparison.json', dict(comparison_schema=2, metric=plan['metric'], order=plan['order'], rows=rows,
                         note='Negative delta is faster. References come from the manifest, never execution order. parent_delta is populated only for legacy cumulative manifests; previous_manifest_delta is descriptive, not a causal attribution. FP8 configuration uses BF16 combine. Alternating order reduces position imbalance but does not eliminate temporal drift.'))
        print(base / 'comparison.csv', flush=True)
    finally:
        release()


if __name__ == '__main__':
    main()
