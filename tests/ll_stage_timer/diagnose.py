"""Run the small-B OFF baseline and separately built timer profiles.

Load the target's Python/CANN environment first (for example, source
/home/tyf/0env.sh), then invoke this file with explicit --devices and launcher
topology arguments. No dependencies are downloaded, no shared Python is changed.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
PROFILES = ('overview', 'prepare', 'prefix', 'send', 'recv')
LIMIT = 60000
CONTROLLED = {'--output-dir', '--output', '--run-name', '--num-tokens',
              '--num-max-tokens-per-rank', '--hidden', '--num-topk', '--num-experts',
              '--dtype', '--do-cpu-sync', '--warmup', '--num-iters', '--perf-tail',
              '--ll-stage-timer', '--ll-expert-timer', '--ll-profile-capture', '--ll-timer-iters', '--ll-timer-mode',
              '--skip-perf-test', '--skip-check', '--plan', '--config', '--test-first-only'}


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--run-name', help='Fresh single directory name, identical on every host')
    p.add_argument('--output-root', default='diagnostic-runs',
                   help='Project-root output directory, outside source-fingerprint trees')
    p.add_argument('--bs', type=int, nargs='+', default=[1, 2, 4, 8, 16, 24, 32])
    p.add_argument('--m', type=int, default=32)
    p.add_argument('--hidden', type=int, default=3584)
    p.add_argument('--topk', type=int, default=8)
    p.add_argument('--experts', type=int, default=384)
    p.add_argument('--dtype', choices=('fp8', 'bf16', 'fp16'), default='fp8')
    p.add_argument('--cpu-sync', type=int, choices=(0, 1), default=0)
    p.add_argument('--profiles', choices=PROFILES + ('full',), nargs='+', default=list(PROFILES))
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--num-iters', type=int, default=100)
    p.add_argument('--perf-tail', type=int, default=50)
    p.add_argument('--settle-iters', type=int, default=100)
    p.add_argument('--diag-iters', type=int, default=5)
    p.add_argument('--plan', action='store_true', help='Print all commands without creating files/importing NPU/building')
    p.add_argument('--skip-build', action='store_true', help='Require --wheel-map; native identity still checked')
    p.add_argument('--wheel-map', type=Path, help='JSON mapping off/profile -> existing local wheel path')
    p.add_argument('--defer-analysis', action='store_true', help='Collect rank-local files before --analyze-existing')
    p.add_argument('--analyze-existing', type=Path, help='Only analyze a completed suite after collecting all ranks')
    p.add_argument('--transfer-mode', choices=('summary',), default='summary',
                   help='One bounded SEND_FIRST.txt; only necessary follow-ups, '
                        'at most 10 transfers including this report')
    return p


def _within_project(value, root):
    path = Path(value)
    path = (path if path.is_absolute() else root / path).resolve()
    if path == root or root not in path.parents:
        raise ValueError('Output paths must be inside this checkout')
    # build_support.source_fingerprint traverses these source trees. Runtime
    # wheels/venvs/results there would alter the hash after the wheel was built.
    relative = path.relative_to(root)
    scanned = ('deep_ep', 'csrc', 'cmake', 'scripts', 'tests', 'benchmarks')
    if relative.parts[0] in scanned or relative.parts[:2] in (
            ('experiments', 'megamoe'), ('experiments', 'ep_gmm_fused')):
        raise ValueError('Use a project-root output directory outside source-fingerprint trees '
                         '(deep_ep/csrc/cmake/scripts/tests/benchmarks and experiment sources)')
    return path


def _forward_value(argv, option, default=None):
    values = []
    for index, token in enumerate(argv):
        if token == option:
            if index + 1 >= len(argv):
                raise ValueError(f'{option} requires a value')
            values.append(argv[index + 1])
        elif token.startswith(option + '='):
            values.append(token.split('=', 1)[1])
    if len(values) > 1:
        raise ValueError(f'Duplicate launcher option: {option}')
    return values[0] if values else default


def make_plan(args, forwarded, root=ROOT, python=sys.executable):
    """Pure planning: no filesystem creation or subprocess/device operations."""
    _validate_transfer_mode(args.transfer_mode)
    root = Path(root).resolve()
    if not args.run_name or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', args.run_name):
        raise ValueError('--run-name must be a fresh ASCII directory name (1..96 characters)')
    forwarded = list(forwarded)
    if forwarded and forwarded[0] == '--':
        forwarded = forwarded[1:]
    for token in forwarded:
        if token.split('=', 1)[0] in CONTROLLED:
            raise ValueError(f'{token} is controlled by diagnose.py; use its matrix/profile options')
    devices = _forward_value(forwarded, '--devices')
    if not devices or not re.fullmatch(r'\d+(,\d+)*', devices):
        raise ValueError('Supply explicit launcher --devices (16 distinct devices per host)')
    ids = [int(value) for value in devices.split(',')]
    if len(ids) != 16 or len(set(ids)) != 16:
        raise ValueError('Launcher requires 16 distinct devices per host')
    nodes = int(_forward_value(forwarded, '--num-nodes', '1'))
    node = int(_forward_value(forwarded, '--node-rank', '0'))
    processes = int(_forward_value(forwarded, '--num-processes', '16'))
    if nodes < 1 or not 0 <= node < nodes or processes != 16:
        raise ValueError('Require 16 local processes and 0 <= node-rank < num-nodes')
    if any(key in os.environ for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE')):
        raise ValueError('Run once per host, outside torchrun; unset inherited rank variables')
    if not args.bs or len(set(args.bs)) != len(args.bs) or any(not 1 <= b <= 32 for b in args.bs):
        raise ValueError('--bs must contain distinct values in 1..32')
    if not max(args.bs) <= args.m <= 32 or min(args.hidden, args.topk, args.experts) < 1:
        raise ValueError('Require max(B) <= M <= 32 and positive H/K/E')
    if args.warmup < 0 or min(args.num_iters, args.settle_iters, args.diag_iters) < 1:
        raise ValueError('Require warmup>=0 and positive iteration counts')
    if not 0 <= args.perf_tail <= args.num_iters:
        raise ValueError('Require 0 <= perf-tail <= num-iters')
    if len(set(args.profiles)) != len(args.profiles):
        raise ValueError('Duplicate profiles')
    if args.skip_build != bool(args.wheel_map):
        raise ValueError('--skip-build and --wheel-map must be supplied together')
    suite = _within_project(Path(args.output_root) / args.run_name, root)
    node_dir = suite / f'node_{node:03d}'
    venv = node_dir / 'venv'
    executable = venv / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    matrix = ['--num-tokens', *map(str, args.bs), '--num-max-tokens-per-rank', str(args.m),
              '--hidden', str(args.hidden), '--num-topk', str(args.topk), '--num-experts', str(args.experts),
              '--dtype', args.dtype, '--do-cpu-sync', str(args.cpu_sync), '--dispatch-only',
              '--warmup', str(args.warmup)]
    profiles = ['off', *args.profiles]
    stages = []
    for profile in profiles:
        timer = profile != 'off'
        wheel_dir = node_dir / 'wheels' / profile
        results = suite / 'runs' / profile
        run = [str(executable), str(root / 'tests/ll_stage_timer/run.py'), *forwarded,
               *matrix, '--output-dir', str(suite / 'runs'), '--run-name', profile,
               '--num-iters', str(args.settle_iters if timer else args.num_iters)]
        if timer:
            run += ['--ll-stage-timer', '--ll-profile-capture', '--ll-timer-mode', 'observe', '--ll-timer-iters', str(args.diag_iters),
                    '--skip-perf-test']
        else:
            run += ['--perf-tail', str(args.perf_tail)]
        stages.append(dict(name=profile, timer=timer, profile=profile if timer else 'full',
                           wheel_dir=str(wheel_dir), results=str(results),
                           build=[str(executable), '-m', 'build', '--wheel', '--no-isolation',
                                  '--outdir', str(wheel_dir)],
                           install=[str(executable), '-m', 'pip', 'install', '--no-index', '--no-deps',
                                    '--force-reinstall', '{wheel}'], run=run,
                           analyze=[str(executable), str(root / 'tests/ll_stage_timer/analyze.py'), str(results),
                                    '--op', 'dispatch', '--output-dir', str(suite / 'analysis' / profile)] if timer else None))
    return dict(format='ll-staged-diagnosis-v1', root=str(root), suite=str(suite), node=node, nodes=nodes,
                python=python, venv=str(venv), venv_python=str(executable), node_dir=str(node_dir),
                create_venv=[python, '-m', 'venv', '--system-site-packages', str(venv)],
                device_visibility=devices, matrix=dict(B=args.bs, M=args.m, H=args.hidden, K=args.topk,
                                                       E=args.experts, dtype=args.dtype, cpu_sync=args.cpu_sync),
                stages=stages, skip_build=args.skip_build, wheel_map=str(args.wheel_map) if args.wheel_map else None,
                defer_analysis=args.defer_analysis, transfer_mode=args.transfer_mode,
                note='Run the same profiles/matrix in the same order on all hosts. ON profiles are separate captures and cannot be summed or merged into one timeline.')


def _environment(plan, stage=None):
    env = dict(os.environ)
    work = Path(plan['node_dir'])
    for variable, suffix in {'TMPDIR':'tmp', 'TMP':'tmp', 'TEMP':'tmp', 'PIP_CACHE_DIR':'cache/pip',
                             'XDG_CACHE_HOME':'cache', 'ASCEND_CACHE_PATH':'cache/ascend',
                             'ASCEND_WORK_PATH':'logs/ascend', 'ASCEND_PROCESS_LOG_PATH':'logs/ascend',
                             'SHMEM_LOG_PATH':'logs/shmem'}.items():
        path = work / suffix
        path.mkdir(parents=True, exist_ok=True)
        env[variable] = str(path)
    for name in ('CUDA_VISIBLE_DEVICES', 'ASCEND_RT_VISIBLE_DEVICES', 'ASCEND_VISIBLE_DEVICES', 'PROBE_VISIBLE_DEVICES'):
        env[name] = plan['device_visibility']
    env.update(PIP_NO_INDEX='1', PIP_DISABLE_PIP_VERSION_CHECK='1', PYTHONDONTWRITEBYTECODE='1',
               DEEPEP_NPU_ARCH='Ascend950')
    env.pop('SHMEM_ROOT', None)
    env.pop('CATLASS_ROOT', None)
    library = str(Path(plan['root']) / 'third-party/shmem/install/shmem/lib')
    env['LD_LIBRARY_PATH'] = os.pathsep.join([library, env.get('LD_LIBRARY_PATH', '')])
    if stage:
        env['DEEPEP_LL_STAGE_TIMER'] = 'ON' if stage['timer'] else 'OFF'
        env['DEEPEP_LL_TIMER_PROFILE'] = stage['profile']
        env['DEEPEP_LL_EXPECT_TIMER_PROFILE'] = stage['profile']
    return env


def _command(command, plan, env, log):
    print('+ ' + json.dumps(command), flush=True)
    with Path(log).open('x', encoding='utf-8') as stream:
        result = subprocess.run(command, cwd=plan['root'], env=env, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Command failed ({result.returncode}); see {log}')


def _wheel_map(plan):
    if not plan['skip_build']:
        return {}
    path = Path(plan['wheel_map'])
    path = path if path.is_absolute() else Path(plan['root']) / path
    values = json.loads(path.read_text(encoding='utf-8'))
    result = {}
    for stage in plan['stages']:
        value = values.get(stage['name'])
        if not isinstance(value, str):
            raise ValueError(f"Wheel map is missing {stage['name']}")
        wheel = Path(value)
        wheel = wheel if wheel.is_absolute() else path.parent / wheel
        if wheel.suffix != '.whl' or not wheel.is_file():
            raise ValueError(f'Wheel does not exist: {wheel}')
        result[stage['name']] = str(wheel.resolve())
    return result


def _verify_code(stage, implementation):
    # The installed-wheel helper prevents checkout PYTHONPATH shadowing.
    profile_id = ('full', *PROFILES).index(stage['profile'])
    return f"""import json
import runpy
from pathlib import Path
package = runpy.run_path('tests/elastic/test_dispatch.py')['_import_installed_deep_ep']()
from deep_ep._native import require_native
native = require_native('staged diagnosis')
def require(condition, message):
    if not condition:
        raise RuntimeError(message)
require(native.low_latency_implementation == {implementation!r}, 'wrong implementation')
require(bool(native.low_latency_stage_timer_enabled) is {stage['timer']!r}, 'wrong timer ON/OFF wheel')
require(native.low_latency_stage_timer_profile == {stage['profile']!r}, 'wrong timer profile')
require(native.low_latency_stage_timer_profile_id == {profile_id!r}, 'wrong timer profile id')
layout = native.low_latency_stage_timer_layout()
require(layout.get('timer_profile') == {stage['profile']!r} and layout.get('timer_profile_id') == {profile_id!r}, 'native profile layout mismatch')
info_path = Path(package.__file__).resolve().parent / '_build_info.json'
info = json.loads(info_path.read_text(encoding='utf-8'))
fingerprint = runpy.run_path('build_support.py')['source_fingerprint']()
require(info.get('source_sha256') == fingerprint, 'installed wheel source fingerprint differs from this checkout; rebuild matching wheels')
print('installed wheel identity and source fingerprint verified:', info_path)
"""


def _performance_rows(suite):
    off_file = suite / 'runs/off/summary.json'
    if not off_file.is_file():
        return []
    off = json.loads(off_file.read_text(encoding='utf-8'))
    def key(config, route):
        return json.dumps([config, route], sort_keys=True)
    baselines = {key(row['config'], row['route']): row for row in off.get('results', []) if 'dispatch' in row}
    identity_fields = ('implementation', 'transport', 'logical_group', 'active_aiv',
                       'send_cores', 'recv_cores', 'core_protocol', 'source_revision', 'source_sha256')
    rows = []
    for path in sorted((suite / 'runs').glob('*/case_*/*/active_capture_dispatch.json')):
        data = json.loads(path.read_text(encoding='utf-8'))
        profile = path.relative_to(suite / 'runs').parts[0]
        row = dict(profile=profile, B=data.get('config', {}).get('B'), status=data.get('status'),
                   samples_per_rank=data.get('diagnostic_iters'), source=path.relative_to(suite).as_posix())
        baseline = baselines.get(key(data.get('config'), data.get('route')))
        left = (baseline or {}).get('comparison') or {}
        right = data.get('comparison') or {}
        comparable = (baseline is not None and all(name in left and name in right for name in identity_fields)
                      and all(left[name] == right[name] for name in identity_fields))
        if not comparable:
            row['status'] = 'unavailable: missing/mismatched OFF configuration or identity'
        elif data.get('status') == 'available':
            reference = baseline['dispatch']['max_rank_mean_kernel_us']
            measured = data['summary']['max_rank_mean_kernel_us']
            row.update(off_max_rank_mean_us=reference, active_max_rank_mean_us=measured,
                       delta_us=measured-reference, ratio=measured/reference)
        rows.append(row)
    return rows


def _write_performance_compare(suite):
    rows = _performance_rows(suite)
    if not rows:
        return None
    lines = ['Active instrumentation perturbation versus OFF formal baseline', '',
             'Both columns use profiler full device-kernel task durations, including INIT/final synchronization/Flush when enabled.',
             'Statistic: maximum of per-rank means. ON captures are separate runs with fewer samples; this ratio diagnoses perturbation, not an algorithm speedup.',
             'Unavailable task identity, count or ordering stays unavailable. Internal KERNEL ticks are not substituted.', '',
             '| profile | B | ON samples/rank | OFF us | active ON us | delta us | ON/OFF | status |',
             '|---|---:|---:|---:|---:|---:|---:|---|']
    def fmt(value):
        return 'missing' if value is None else f'{value:.6f}'
    for row in rows:
        lines.append(f"| {row['profile']} | {row['B']} | {row['samples_per_rank']} | {fmt(row.get('off_max_rank_mean_us'))} | {fmt(row.get('active_max_rank_mean_us'))} | {fmt(row.get('delta_us'))} | {fmt(row.get('ratio'))} | {row['status']} |")
    text = '\n'.join(lines) + '\n'
    if len(text) > LIMIT or len(text.encode('utf-8')) > LIMIT:
        raise ValueError('Performance comparison exceeds transfer limit; reduce the matrix')
    path = suite / 'PERFORMANCE_COMPARE.txt'
    path.write_bytes(text.encode('utf-8'))
    return path


def _transfer_module(name):
    spec = importlib.util.spec_from_file_location('ll_' + name, Path(__file__).with_name(name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_transfer_mode(transfer_mode):
    if transfer_mode != 'summary':
        raise ValueError('Only summary transfer is allowed: the 10-transfer budget '
                         'includes SEND_FIRST.txt and all necessary follow-ups; '
                         'full archive sending is disabled')


def write_send_first(suite, transfer_mode='summary', *, receipt=None):
    """Write the initial report within the shared 10-transfer budget."""
    _validate_transfer_mode(transfer_mode)
    suite = Path(suite)
    _write_performance_compare(suite)
    return _transfer_module('triage_report').export_report(suite, receipt=receipt)


def analyze_existing(suite, root=ROOT, python=sys.executable, transfer_mode='summary', *, receipt=None):
    _validate_transfer_mode(transfer_mode)
    suite = _within_project(suite, Path(root).resolve())
    plan_file = suite / 'node_000/plan.json'
    plan = json.loads(plan_file.read_text(encoding='utf-8'))
    # Analysis is CPU-only and can run with the current interpreter after collection.
    for stage in plan['stages']:
        if not stage['timer']:
            continue
        destination = suite / 'analysis' / stage['name']
        if destination.exists():
            # Preserve prior parsed results. The suite coverage report marks
            # missing B/rank/iterations explicitly rather than claiming success.
            continue
        if not (suite / 'runs' / stage['name']).exists():
            continue
        command = [python, str(Path(root) / 'tests/ll_stage_timer/analyze.py'),
                   str(suite / 'runs' / stage['name']), '--op', 'dispatch', '--output-dir', str(destination)]
        subprocess.run(command, cwd=root, check=True)
    # The current CLI choice controls transfer even for plans written before
    # summary mode existed; their old archive is never decoded or deleted.
    return write_send_first(suite, transfer_mode=transfer_mode, receipt=receipt)


def _validate_launcher_before_build(plan):
    """Reuse the launcher's pure argument/endpoint checks without mutating env."""
    st = runpy.run_path(str(Path(plan['root']) / 'tests/elastic/test_ep_low_latency.py'))
    launcher = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    launcher.add_argument('--devices')
    launcher.add_argument('--validation', action='store_true')
    local, remaining = launcher.parse_known_args(plan['stages'][0]['run'][2:])
    if local.validation:
        raise ValueError('diagnose executes real target runs; --validation is only for the separate CPU-plan launcher')
    if _forward_value(remaining, '--num-processes') is None:
        remaining += ['--num-processes', '16']
    options = st['parse_args'](remaining)
    environment = dict(os.environ)
    world, _, _ = st['launch_config'](options, environment)
    st['validate_args'](options, world)
    st['configure_endpoints'](options, environment)


def execute(plan, *, receipt=None):
    # Older archive plans must fail before any build, device work or directory
    # creation. analyze_existing still defaults their saved results to summary.
    _validate_transfer_mode(plan.get('transfer_mode', 'summary'))
    root = Path(plan['root'])
    suite, node = Path(plan['suite']), Path(plan['node_dir'])
    if node.exists() or (suite.exists() and plan['nodes'] == 1):
        raise ValueError('Run directory already exists; choose a new --run-name')
    sdk = root / 'third-party/shmem/install/shmem'
    if not (sdk / 'deepep-sdk.json').is_file() or not (sdk / 'include/host/shmem_host_def.h').is_file():
        raise ValueError('Prepare the locked project-local SHMEM SDK first; diagnose never downloads or rebuilds it')
    _validate_launcher_before_build(plan)
    wheels = _wheel_map(plan)
    required = ['torch', 'torch_npu'] + ([] if plan['skip_build'] else ['build', 'setuptools', 'wheel', 'pybind11'])
    missing = [module for module in required if importlib.util.find_spec(module) is None]
    if missing:
        raise ValueError('Missing existing environment dependencies: ' + ', '.join(missing) + '; install local wheels before running')
    node.mkdir(parents=True, exist_ok=False)
    (node / 'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    env = _environment(plan)
    logs = node / 'logs'
    logs.mkdir(exist_ok=True)
    _command(plan['create_venv'], plan, env, logs / 'venv.log')
    implementation = json.loads((root / 'tests/ll_stage_timer/branch.json').read_text())['implementation']
    for stage in plan['stages']:
        env = _environment(plan, stage)
        profile = stage['name']
        if not plan['skip_build']:
            wheel_dir = Path(stage['wheel_dir'])
            wheel_dir.mkdir(parents=True, exist_ok=False)
            _command(stage['build'], plan, env, logs / f'{profile}-build.log')
            candidates = list(wheel_dir.glob('*.whl'))
            if len(candidates) != 1:
                raise ValueError(f'Expected exactly one newly built wheel in {wheel_dir}')
            wheels[profile] = str(candidates[0])
        command = [wheels[profile] if value == '{wheel}' else value for value in stage['install']]
        _command(command, plan, env, logs / f'{profile}-install.log')
        _command([plan['venv_python'], '-c', _verify_code(stage, implementation)],
                 plan, env, logs / f'{profile}-identity.log')
        _command(stage['run'], plan, env, logs / f'{profile}-run.log')
        if stage['timer'] and plan['node'] == 0 and not plan['defer_analysis']:
            _command(stage['analyze'], plan, env, logs / f'{profile}-analyze.log')
    if plan['node'] == 0:
        write_send_first(suite, transfer_mode=plan.get('transfer_mode', 'summary'), receipt=receipt)
    return suite / 'SEND_FIRST.txt'


def _print_summary_receipt(path, receipt):
    if not receipt:
        print('No summary issued on this node; use the node_000 analysis result.')
        return
    print(f'Suite send budget: {receipt["issued"]}/10 issued; {receipt["remaining"]} remaining.')
    print(path if receipt['new_count'] else 'Identical summary already issued; do not resend it.')


def main(argv=None):
    p = parser()
    args, forwarded = p.parse_known_args(argv)
    receipt = {}
    try:
        if args.analyze_existing:
            if args.plan:
                print(json.dumps(dict(analyze_existing=str(args.analyze_existing),
                                      transfer_mode=args.transfer_mode, mutation=False)))
                return 0
            path = analyze_existing(args.analyze_existing, transfer_mode=args.transfer_mode, receipt=receipt)
            _print_summary_receipt(path, receipt)
            return 0
        plan = make_plan(args, forwarded)
        if args.plan:
            print(json.dumps(plan, indent=2))
            return 0
        path = execute(plan, receipt=receipt)
        _print_summary_receipt(path, receipt)
    except (ValueError, RuntimeError) as exc:
        p.exit(1, f'{exc}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
