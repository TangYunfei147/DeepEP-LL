"""Build every committed candidate, then run the unchanged LL ST sampling protocol."""
import argparse
import csv
import hashlib
import importlib.util
import io
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
import uuid

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


def selected_seeds(args):
    seeds = list(getattr(args, 'seeds', [0]))
    if not seeds or any(type(seed) is not int or not 0 <= seed <= 2147483645 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('seeds must be distinct integers in0..2147483645')
    return seeds


def st_command(python, args, directory, root=ROOT, seed=0):
    return [str(python), '-I', str(root / 'tests/elastic/test_ep_low_latency.py'),
            '--num-processes', '16', '--num-nodes', str(args.num_nodes), '--node-rank', str(args.node_rank),
            '--master-addr', args.master_addr, '--master-port', str(args.master_port),
            '--shmem-port', str(args.shmem_port), '--num-tokens', '256',
            '--num-max-tokens-per-rank', '256', '--hidden', '3584', '--num-topk', '8',
            '--num-experts', '384', '--dtype', 'fp8', '--do-cpu-sync', '0', '--seed', str(seed),
            '--output-dir', str(directory.parent), '--run-name', directory.name]


def validate_summary(directory, revision, send_cores, num_nodes, seed=0):
    """Use summary.csv mean_kernel_us; reject mixed cases or incomplete stable tails."""
    directory = Path(directory)
    report = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
    world = 16 * num_nodes
    if not isinstance(report, dict) or report.get('status') != 'completed' or report.get('world_size') != world or report.get('case_count') != 1 or report.get('result_count') != 1:
        raise ValueError(f'Incomplete or mixed ST run: {directory}')
    if not isinstance(report.get('results'), list) or len(report['results']) != 1 or not isinstance(report['results'][0], dict) or not isinstance(report.get('config'), dict):
        raise ValueError(f'Malformed ST report structure: {directory}')
    config = report['config']
    for key, value in dict(warmup=10, num_iters=100, perf_tail=50, route='normal', seed=seed,
                           skip_check=False, skip_perf_test=False, dispatch_only=False, ll_stage_timer=False).items():
        if config.get(key) != value:
            raise ValueError(f'Unexpected sampling setting {key}: {config.get(key)!r}')
    if config.get('correctness_repeats', 3) != 3:
        raise ValueError('Expected3 correctness routes; otherwise the measured routing seed changes')
    result = report['results'][0]
    if result['case_id'] != 0 or result['route'] != 'normal' or result['config'] != SHAPE or result.get('correctness') != 'passed':
        raise ValueError('Unexpected case, route, or shape')
    identity = result.get('comparison', {})
    if identity.get('seed', seed) != seed or identity.get('correctness_repeats', 3) != 3:
        raise ValueError('Routing seed/correctness-repeat identity mismatch')
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


def _run_key(item):
    return item['label'], item['round'], item.get('seed', 0)


def _command_identity(command):
    """Retain ST semantics while allowing a new process port/output location."""
    starts = [index for index, value in enumerate(command) if Path(value).name == 'test_ep_low_latency.py']
    if len(starts) != 1:
        raise ValueError('Previous plan does not identify the standard LL ST entry')
    options = command[starts[0]+1:]
    ignored = {'--master-port', '--shmem-port', '--output-dir', '--run-name'}
    result = []
    seed = 0
    seen_seed = False
    index = 0
    while index < len(options):
        if options[index] == '--seed':
            if seen_seed or index+1 >= len(options):
                raise ValueError('Incomplete seed in previous command')
            seed = int(options[index+1])
            seen_seed = True
            index += 2
        elif options[index] in ignored:
            if index+1 >= len(options):
                raise ValueError('Incomplete command in previous plan')
            index += 2
        else:
            result.append(options[index])
            index += 1
    return result + ['--seed', str(seed)]


def validate_previous_plan(previous, current, node_rank):
    """Accept an old saved plan, but never change the experiment on resume."""
    saved_entries = previous.get('commits', [])
    normalized = []
    for index, entry in enumerate(saved_entries):
        entry = dict(entry)
        if not any(field in entry for field in ('comparison_base', 'comparison_label', 'comparison_kind')):
            reference = saved_entries[max(0, index-1)]
            entry.update(comparison_base=reference['revision'], comparison_label=reference['label'],
                         comparison_kind='cumulative_parent' if index else 'baseline')
        normalized.append(entry)
    if normalized != current['commits']:
        raise ValueError('Resume plan mismatch: commits/references')
    if previous.get('order', 'manifest') != current['order']:
        raise ValueError('Resume plan mismatch: order (legacy plans used --order manifest)')
    if previous.get('seeds', [0]) != current.get('seeds', [0]):
        raise ValueError('Resume plan mismatch: seeds; use a new run prefix for a different seed set')
    if 'dependencies' in previous and previous['dependencies'] != current.get('dependencies'):
        raise ValueError('Resume plan mismatch: committed SDK dependency identities')
    for field in ('metric', 'sampling', 'send_cores', 'recv_cores', 'timer'):
        if previous.get(field) != current.get(field):
            raise ValueError(f'Resume plan mismatch: {field}')
    old_runs = previous.get('phases', {}).get('run', [])
    new_runs = {_run_key(item): item for item in current['phases']['run']}
    old_rounds = previous.get('rounds', max((item['round'] for item in old_runs), default=0))
    if not isinstance(old_rounds, int) or old_rounds < 1 or old_rounds > current['rounds']:
        raise ValueError('Resume may increase rounds, but cannot decrease previously planned rounds')
    expected = {(entry['label'], round_id, seed) for entry in current['commits']
                for round_id in range(1, old_rounds+1) for seed in current.get('seeds', [0])}
    if len(old_runs) != len(expected) or {_run_key(item) for item in old_runs} != expected:
        raise ValueError('Previous plan has missing or duplicate round/candidate entries')
    positions = {}
    for item in old_runs:
        target = new_runs[_run_key(item)]
        pair = item['round'], item.get('seed', 0)
        positions[pair] = positions.get(pair, 0)+1
        if item['revision'] != target['revision'] or item.get('execution_position', positions[pair]) != target['execution_position']:
            raise ValueError(f'Resume plan mismatch: revision/execution_position for {_run_key(item)}')
        if _command_identity(item['command']) != _command_identity(target['command']):
            raise ValueError(f'Resume ST shape/topology/argument mismatch for {_run_key(item)}')
        command = item['command']
        if int(command[command.index('--node-rank')+1]) != node_rank:
            raise ValueError('Previous plan belongs to a different node rank')
    identity = previous.get('run_identity')
    if identity is not None and identity != current.get('run_identity'):
        raise ValueError('Resume device/host/topology identity mismatch')
    return dict(previous_rounds=old_rounds, legacy_devices_unavailable=identity is None)


def previous_plans(base, node_rank):
    base = Path(base)
    paths = [base / f'suite-plan.node-{node_rank}.json']
    paths += sorted((base / 'sessions').glob(f'*/suite-plan.node-{node_rank}.json'))
    result, ignored = [], []
    for path in paths:
        if path.is_file():
            try:
                saved = json.loads(path.read_text(encoding='utf-8'))
                if not isinstance(saved, dict):
                    raise ValueError('Saved suite plan must be a JSON object')
            except (OSError, ValueError) as error:
                if path == paths[0]:
                    raise ValueError(f'Original suite plan is unreadable: {path}') from error
                ignored.append(dict(path=str(path), reason=str(error)[:1024]))
            else:
                result.append(dict(path=str(path), plan=saved))
    if not result:
        raise ValueError(f'No previous suite plan for node{node_rank}: cannot establish result identity')
    if ignored:
        result[0]['ignored_session_plans'] = ignored
    return result


def _write_new_file(path, content):
    """Publish a complete new file atomically without replacing an older file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    try:
        with temporary.open('x', encoding='utf-8', newline='') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # Fails if path already exists; same-directory hard link.
    finally:
        temporary.unlink(missing_ok=True)


def write_new_json(path, value):
    _write_new_file(path, json.dumps(value, indent=2, sort_keys=True)+'\n')


def _attempt_directories(canonical):
    canonical = Path(canonical)
    pattern = re.compile(re.escape(canonical.name)+r'(?:\.attempt-[A-Za-z0-9_-]+)?$')
    paths = [canonical] + sorted(canonical.parent.glob(canonical.name+'.attempt-*'))
    result = []
    for path in paths:
        if not pattern.fullmatch(path.name) or not path.exists():
            continue
        if path.is_symlink():
            raise ValueError(f'Result directory must not be a symlink: {path}')
        result.append(path)
    return result


def _source_session(directory, history):
    for record in history:
        saved = record['plan']
        for item in saved.get('result_schedule', {}).get('items', []):
            if item.get('action') == 'run' and Path(item['directory']) == Path(directory):
                return saved.get('session_id', 'unknown')
    return 'unknown' if '.attempt-' in Path(directory).name else 'legacy'


SDK_HASH_FIELDS = ('native_sha256', 'kernel_sha256', 'sdk_manifest_sha256', 'sdk_library_sha256')


def sdk_record_path(directory, node_rank=0):
    return Path(str(directory)+f'.node-{node_rank}.build.json')


def recorded_sdk_identity(directory, revision, expected):
    """Read persisted provenance only; resuming never opens an old SDK binary."""
    path = sdk_record_path(directory)
    if not path.is_file():
        return dict(expected, revision=revision, sdk_identity_source='legacy-unavailable',
                    **{field: None for field in SDK_HASH_FIELDS})
    record = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(record, dict) or record.get('revision') != revision or any(record.get(field) != value for field, value in expected.items()):
        raise ValueError('Saved SDK identity differs from the tested commit')
    if any(not isinstance(record.get(field), str) or not re.fullmatch(r'[0-9a-f]{64}', record[field]) for field in SDK_HASH_FIELDS):
        raise ValueError('Saved SDK identity has incomplete library/manifest hashes')
    return dict(record, sdk_identity_source='recorded-build')


def manifest_sdk_identity(revision, expected, manifest):
    record = dict(expected, revision=revision, **{field: manifest[field] for field in SDK_HASH_FIELDS})
    if any(field in manifest and manifest[field] != value for field, value in expected.items()):
        raise ValueError('Installed wheel dependencies differ from the planned commit')
    if any(not re.fullmatch(r'[0-9a-f]{64}', record[field]) for field in SDK_HASH_FIELDS):
        raise ValueError('Invalid wheel/SDK hash in build manifest')
    return dict(record, sdk_identity_source='recorded-build')


def result_schedule(plan, args, root=ROOT):
    """Validate saved single-node results and choose complete items for reuse."""
    if args.num_nodes != 1 or args.node_rank != 0:
        raise ValueError('Result reuse supports only a single-node run at node rank0')
    resume = bool(getattr(args, 'resume_results', False))
    base = build.resolve(root, args.output_dir) / args.run_prefix
    history = previous_plans(base, 0) if resume else []
    compatibility = [dict(path=record['path'], **validate_previous_plan(record['plan'], plan, 0)) for record in history]
    session_id = uuid.uuid4().hex
    items = []
    for planned in plan['phases']['run']:
        canonical = Path(planned['directory'])
        complete, rejected = [], []
        for directory in _attempt_directories(canonical) if resume else []:
            try:
                values = validate_summary(directory, planned['revision'], args.send_cores, args.num_nodes, seed=planned.get('seed', 0))
                sdk_identity = recorded_sdk_identity(directory, planned['revision'], plan.get('dependencies', {}).get(planned['revision'], {}))
            except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError, csv.Error, OverflowError) as error:
                rejected.append(dict(directory=str(directory), reason=str(error)[:2048]))
            else:
                complete.append((directory, values, sdk_identity))
        if len(complete) > 1:
            raise ValueError(f'Ambiguous complete attempts for {_run_key(planned)}: '+', '.join(str(item[0]) for item in complete))
        choice = dict(label=planned['label'], revision=planned['revision'], round=planned['round'],
                      seed=planned.get('seed', 0), measured_route_seed=planned.get('seed', 0)+2,
                      execution_position=planned['execution_position'], rejected_attempts=rejected)
        if complete:
            directory, values, sdk_identity = complete[0]
            choice.update(action='reuse', directory=str(directory), source_directory=str(directory),
                          values=values, source_session_id=_source_session(directory, history), sdk_identity=sdk_identity)
        else:
            directory = canonical.with_name(canonical.name+'.attempt-'+session_id) if resume else canonical
            choice.update(action='run', directory=str(directory), source_directory=str(directory),
                          source_session_id=session_id)
        items.append(choice)
    return dict(session_id=session_id, resume_results=resume, items=items, compatibility=compatibility,
                ignored_session_plans=[ignored for record in history for ignored in record.get('ignored_session_plans', [])],
                legacy_devices_unavailable=any(record['legacy_devices_unavailable'] for record in compatibility) or
                    any(item['action']=='reuse' and item['source_session_id']=='unknown' for item in items))


def apply_result_schedule(plan, schedule):
    """Apply recovered attempt names without changing the planned ST arguments."""
    local = {_run_key(item): item for item in plan['phases']['run']}
    if len(schedule['items']) != len(local) or {_run_key(item) for item in schedule['items']} != set(local):
        raise ValueError('Coordinator result schedule has missing or duplicate entries')
    result = []
    for choice in schedule['items']:
        original = local[_run_key(choice)]
        if any(choice.get(field) != original.get(field) for field in ('revision', 'execution_position')):
            raise ValueError('Coordinator selected a different revision/order')
        if choice['action'] not in ('run', 'reuse'):
            raise ValueError('Unknown result action')
        selected = Path(choice['directory']).name
        canonical = Path(original['directory'])
        if not re.fullmatch(re.escape(canonical.name)+r'(?:\.attempt-[A-Za-z0-9_-]+)?', selected):
            raise ValueError('Unsafe result attempt name')
        directory = canonical.with_name(selected)
        item = dict(original, **choice)
        item['directory'] = str(directory)
        item['command'] = list(original['command'])
        for option, value in (('--output-dir', str(directory.parent)), ('--run-name', directory.name)):
            item['command'][item['command'].index(option)+1] = value
        result.append(item)
    if [_run_key(item) for item in result] != [_run_key(item) for item in plan['phases']['run']]:
        raise ValueError('Coordinator changed execution order')
    return result


def comparisons(entries, measured, order='alternating'):
    rows = []
    paired = {(key if isinstance(key, tuple) else (key, 0)): value for key, value in measured.items()}
    if len(paired) != len(measured):
        raise ValueError('Duplicate round/seed measurements')
    for (round_id, seed), values in sorted(paired.items()):
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
                rows.append(dict(round=round_id, seed=seed, requested_seed=seed, measured_route_seed=seed+2,
                                 label=entry['label'], revision=entry['revision'], operation=operation,
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
    result.add_argument('--seeds', type=int, nargs='+', default=[0],
                        help='Requested uniform-route seeds; default0. Performance uses seed+2 after3 correctness routes')
    result.add_argument('--order', choices=('alternating', 'manifest'), default='alternating',
                        help='alternating: odd rounds use manifest order, even rounds reverse it; manifest: repeat fixed order')
    modes = result.add_mutually_exclusive_group()
    modes.add_argument('--build-only', action='store_true')
    modes.add_argument('--run-only', action='store_true')
    result.add_argument('--resume', action='store_true', help='Reuse verified wheels; never skip or overwrite old performance runs')
    result.add_argument('--resume-results', action='store_true',
                        help='Single-node only: validate/reuse complete results at the same run prefix; retry incomplete items in new attempt directories')
    result.add_argument('--build-workspace', type=Path)
    result.add_argument('--offline-deps', type=Path)
    result.add_argument('--plan', action='store_true')
    return result


def make_plan(args, root=ROOT):
    if not 1 <= args.send_cores <= 55 or args.rounds < 1 or not math.isfinite(args.ready_timeout) or args.ready_timeout <= 0:
        raise ValueError('send-cores must be 1..55, rounds must be positive')
    entries = commits(build.resolve(root, args.commits), root)
    seeds = selected_seeds(args)
    dependencies = {entry['revision']: build.dependency_identity(root, entry['revision']) for entry in entries}
    resume_results = bool(getattr(args, 'resume_results', False))
    if resume_results and (args.build_only or args.num_nodes != 1 or args.node_rank != 0):
        raise ValueError('--resume-results requires a single-node run (--num-nodes 1 --node-rank 0); it cannot be combined with --build-only')
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
            if args.resume or resume_results:
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
            for seed in seeds:
                parent = build.resolve(root, args.output_dir) / args.run_prefix / entry['label']
                if seeds != [0]:
                    parent /= f'seed-{seed}'
                directory = parent / f'round-{round_id:03}'
                phases['run'].append(dict(label=entry['label'], revision=entry['revision'], round=round_id,
                                          seed=seed, measured_route_seed=seed+2, directory=str(directory),
                                          command=st_command(python, args, directory, root, seed=seed)))
    positions = {round_id: {entry['label']: index for index, entry in enumerate(execution_entries(entries, round_id, args.order), 1)}
                 for round_id in range(1, args.rounds+1)}
    for item in phases['run']:
        item['execution_position'] = positions[item['round']][item['label']]
    phases['run'].sort(key=lambda item: (item['round'], seeds.index(item['seed']), item['execution_position']))
    return dict(commits=entries, phases=phases, metric='summary.csv: pooled mean_kernel_us; dispatch and combine separately',
                sampling=dict(warmup=10, num_iters=100, perf_tail=50), send_cores=args.send_cores,
                recv_cores=56-args.send_cores, timer='OFF',
                order=args.order, rounds=args.rounds, comparison_schema=2,
                seeds=seeds, routing=dict(route='normal', correctness_repeats=3, measured_seed_offset=2),
                dependencies=dependencies,
                resume_results=resume_results,
                run_identity=None if args.build_only else dict(shape=SHAPE, num_nodes=args.num_nodes, node_rank=args.node_rank,
                    devices=device_ids(args.devices), local_processes=16, master_addr=args.master_addr, hostname=socket.gethostname()),
                control_protocol='ll-control-ready-v1',
                control_port=control_port(args) if not args.build_only and args.num_nodes>1 else None)


def main():
    cli = parser()
    args = cli.parse_args()
    try:
        plan = make_plan(args, root=ROOT)
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        cli.exit(2, str(error) + '\n')
    if args.plan:
        if args.resume_results:
            schedule = result_schedule(plan, args, root=ROOT)
            items = apply_result_schedule(plan, schedule)
            needed = {item['revision'] for item in items if item['action'] == 'run'}
            plan['result_schedule'] = schedule
            plan['phases']['build'] = [command for command in plan['phases']['build'] if command[command.index('--revision')+1] in needed]
            plan['phases']['install'] = [install for entry, install in zip(plan['commits'], plan['phases']['install']) if entry['revision'] in needed]
            plan['phases']['run'] = items
        print(json.dumps(plan, indent=2))
        return
    release = build.acquire_lock(ROOT / build.STATE / 'suite.lock', recover_dead=args.resume_results)
    try:
        if args.build_only:
            for command in plan['phases']['build']:
                subprocess.run(command, cwd=ROOT, check=True)
            print('All wheels are built. No NPU devices were used.', flush=True)
            return
        base = build.resolve(ROOT, args.output_dir) / args.run_prefix
        if args.resume_results:
            schedule = result_schedule(plan, args, root=ROOT)
            items = apply_result_schedule(plan, schedule)
            output = base / 'sessions' / schedule['session_id']
            output.mkdir(parents=True, exist_ok=False)
        else:
            # Fresh runs retain the existing behavior; resume never overwrites them.
            for item in plan['phases']['run']:
                if (args.node_rank == 0 and Path(item['directory']).exists()) or any(Path(item['directory'] + f'.node-{args.node_rank}' + suffix).exists() for suffix in ('.done.json', '.log', '.idle.txt')):
                    raise RuntimeError('Previous performance output exists; use --resume-results for a single-node continuation or a fresh --run-prefix.')
            if (base / f'suite-plan.node-{args.node_rank}.json').exists() or (args.node_rank == 0 and any((base / name).exists() for name in ('comparison.csv', 'comparison.json'))):
                raise RuntimeError('Previous suite output exists; use --resume-results or a fresh --run-prefix')
            session_id = uuid.uuid4().hex
            items = [dict(item, action='run', source_directory=item['directory'], source_session_id=session_id,
                          rejected_attempts=[]) for item in plan['phases']['run']]
            schedule = dict(session_id=session_id, resume_results=False, items=items, compatibility=[],
                            legacy_devices_unavailable=False, ignored_session_plans=[])
            output = base
            output.mkdir(parents=True, exist_ok=True)
        saved_plan = dict(plan, session_id=schedule['session_id'], result_schedule=schedule)
        write_new_json(output / f'suite-plan.node-{args.node_rank}.json', saved_plan)
        needed = {item['label'] for item in items if item['action'] == 'run'}
        needed_revisions = {entry['revision'] for entry in plan['commits'] if entry['label'] in needed}
        # Complete prior results require neither an installed wheel nor a working SDK.
        for command in plan['phases']['build']:
            if command[command.index('--revision')+1] in needed_revisions:
                subprocess.run(command, cwd=ROOT, check=True)
        manifests = {}
        for entry, install in zip(plan['commits'], plan['phases']['install']):
            if entry['label'] not in needed:
                continue
            path = Path(install['manifest'])
            info = build.verify_manifest(path, entry['revision'], args.send_cores, root=ROOT)
            manifests[entry['label']] = info
            venv = Path(install['environment'])
            if not venv.exists():
                subprocess.run([sys.executable, '-m', 'venv', '--system-site-packages', str(venv)], check=True)
            if not (venv / 'pyvenv.cfg').is_file() or not Path(install['python']).is_file():
                raise RuntimeError(f'Incomplete local virtual environment: {venv}; preserve it and choose a fresh checkout')
            env = run_environment(args, entry, info, root=ROOT)
            wheel = build.resolve(path.parent, info['wheel'])
            subprocess.run([install['python'], '-I', '-m', 'pip', 'install', '--no-index', '--no-deps', '--force-reinstall', str(wheel)], cwd=venv, env=env, check=True)
            subprocess.run([install['python'], '-I', '-c', preflight_code(info, args.send_cores)], cwd=venv, env=env, check=True)
        measured = {}
        executed = 0
        for item in items:
            if item['action'] == 'reuse':
                measured.setdefault((item['round'], item.get('seed', 0)), {})[item['label']] = item['values']
                item['invocation_execution_position'] = None
                print(f'Reusing complete {item["label"]}, round{item["round"]}, seed{item.get("seed", 0)}: {item["source_directory"]}', flush=True)
                continue
            executed += 1
            item['invocation_execution_position'] = executed
            directory = Path(item['directory'])
            directory.parent.mkdir(parents=True, exist_ok=True)
            if (args.node_rank == 0 and directory.exists()) or any(Path(str(directory)+f'.node-{args.node_rank}'+suffix).exists() for suffix in ('.done.json', '.log', '.idle.txt')):
                raise RuntimeError(f'Result attempt already exists and will not be overwritten: {directory}')
            env = run_environment(args, item, manifests[item['label']], root=ROOT)
            identity = dict(protocol=plan['control_protocol'], commits=plan['commits'], run_prefix=args.run_prefix,
                            label=item['label'], revision=item['revision'], round=item['round'], rounds=args.rounds,
                            seed=item.get('seed', 0), seeds=plan['seeds'], dependencies=plan['dependencies'],
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
            item['sdk_identity'] = manifest_sdk_identity(item['revision'], plan['dependencies'][item['revision']], manifests[item['label']])
            write_new_json(sdk_record_path(directory, args.node_rank), item['sdk_identity'])
            print(f'Run {item["label"]}, round{item["round"]}, seed{item.get("seed", 0)} (measured route seed{item.get("seed", 0)+2}), node{args.node_rank}; stdout remains visible.', flush=True)
            run_logged(item['command'], str(directory)+f'.node-{args.node_rank}.log', env)
            write_new_json(Path(str(directory) + f'.node-{args.node_rank}.done.json'),
                           dict(revision=item['revision'], seed=item.get('seed', 0), exit_code=0, session_id=schedule['session_id']))
            if args.node_rank == 0:
                values = validate_summary(directory, item['revision'], args.send_cores, args.num_nodes, seed=item.get('seed', 0))
                measured.setdefault((item['round'], item.get('seed', 0)), {})[item['label']] = values
        if args.node_rank != 0:
            print('All ST commands completed. Aggregated summary is written by node0.', flush=True)
            return
        rows = comparisons(plan['commits'], measured, plan['order'])
        sources = {_run_key(item): item for item in items}
        for row in rows:
            source = sources[(row['label'], row['round'], row['seed'])]
            reference = sources[(row['reference_label'], row['round'], row['seed'])]
            row.update(result_status='reused' if source['action']=='reuse' else 'new',
                       source_directory=source['source_directory'], source_session_id=source['source_session_id'],
                       invocation_session_id=schedule['session_id'], invocation_execution_position=source['invocation_execution_position'],
                       reference_crosses_sessions=None if 'unknown' in (source['source_session_id'], reference['source_session_id']) else
                           source['source_session_id'] != reference['source_session_id'])
            for field in ('dependency_id', 'shmem_revision', 'sdk_manifest_sha256', 'sdk_library_sha256', 'sdk_identity_source'):
                row[field] = source['sdk_identity'].get(field)
                row['reference_'+field] = reference['sdk_identity'].get(field)
            current_sdk, reference_sdk = row['sdk_library_sha256'], row['reference_sdk_library_sha256']
            row['sdk_library_matches_reference'] = current_sdk == reference_sdk if current_sdk and reference_sdk else None
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        _write_new_file(output / 'comparison.csv', stream.getvalue())
        write_new_json(output / 'comparison.json', dict(comparison_schema=2, metric=plan['metric'], order=plan['order'], rows=rows,
                       seeds=plan['seeds'], dependencies=plan['dependencies'], routing=plan['routing'],
                       session_id=schedule['session_id'], resume_results=args.resume_results,
                       legacy_devices_unavailable=schedule['legacy_devices_unavailable'],
                       ignored_session_plans=schedule['ignored_session_plans'],
                       note='Negative delta is faster. References come from the manifest, never execution order. parent_delta is populated only for legacy cumulative manifests; previous_manifest_delta is descriptive, not a causal attribution. Resumed measurements can span separate sampling sessions; inspect source_session_id/reference_crosses_sessions before interpreting small deltas. Legacy device mappings were not recorded and cannot be retroactively verified. FP8 configuration uses BF16 combine. Alternating order does not eliminate temporal drift.'))
        print(output / 'comparison.csv', flush=True)
    finally:
        release()


if __name__ == '__main__':
    main()
