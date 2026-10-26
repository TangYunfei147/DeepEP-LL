"""Clean/build the SDK once, then every OFF/sparse wheel for a complete A/B.

No NPU work is launched. Existing run directories, wheels and logs are preserved.
Use --plan to print the complete build selection without creating any files.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
_source = Path(__file__).with_name('diagnose.py')
_spec = importlib.util.spec_from_file_location('ll_build_ab_diagnose', _source)
diagnose = importlib.util.module_from_spec(_spec)
# Compile directly instead of SourceFileLoader.exec_module, which may create a
# __pycache__ even for a --plan invocation. The module still has normal globals
# and its own __file__, so planning and injected test commands share one state.
exec(compile(_source.read_text(encoding='utf-8'), str(_source), 'exec'), diagnose.__dict__)
EXPERIMENTS = {
    'small': ('cb59', 'cb59-small'),
    'pair': ('cb59-stream', 'cb59-pair'),
    'all': ('cb59', 'cb59-small', 'cb59-stream', 'cb59-pair'),
}


def parser():
    value = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    value.add_argument('--experiment', choices=tuple(EXPERIMENTS), required=True)
    value.add_argument('--run-name', required=True, help='Fresh group directory name; never reuse an existing build group')
    value.add_argument('--output-root', default='diagnostic-runs')
    value.add_argument('--plan', action='store_true', help='Print commands only, with no writes/imports of NPU libraries/builds')
    return value


def make_plan(args, *, root=ROOT, python=sys.executable, environment=None):
    root = Path(root).resolve()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', args.run_name):
        raise ValueError('--run-name must be a fresh ASCII directory name (1..96 characters)')
    group = diagnose._within_project(Path(args.output_root) / args.run_name, root)
    env = dict(os.environ if environment is None else environment)
    variants = []
    for implementation in EXPERIMENTS[args.experiment]:
        options, forwarded = diagnose.parser().parse_known_args([
            '--build-only', '--run-name', implementation, '--output-root', str(group)])
        selected = dict(env, DEEPEP_LL_DISPATCH_IMPL=implementation)
        # Shared configuration remains validated by diagnose/dispatch_configuration;
        # inherited mismatched S/R/P is reported instead of silently overridden.
        variants.append(diagnose.make_plan(options, forwarded, root=root, python=python, environment=selected))
    sdk = dict(root=str(root), python=python, node_dir=str(group / 'sdk'), build_only=True,
               dispatch_impl=variants[0]['dispatch_impl'], core_split=variants[0]['core_split'],
               small_b_prefix_cores=variants[0]['small_b_prefix_cores'])
    return dict(format='ll-complete-ab-build-v1', root=str(root), group=str(group), experiment=args.experiment,
                python=python, sdk_build=[python, str(root / 'scripts/build_shmem.py')], sdk_environment=sdk,
                variants=variants, wheel_maps_output=str(group / 'wheel-maps.json'),
                note='SDK is cleaned and built once by build_shmem.py before any wheel. Each implementation '
                     'cleans project wheel build products before OFF and sparse. All wheel maps are published '
                     'only after their Host identity checks; this command never launches devices.')


def execute(plan):
    group = Path(plan['group'])
    if group.exists():
        raise ValueError('Build group already exists; choose a new --run-name')
    required = ('torch', 'torch_npu', 'build', 'setuptools', 'wheel', 'pybind11')
    missing = [module for module in required if importlib.util.find_spec(module) is None]
    if missing:
        raise ValueError('Missing existing environment dependencies: ' + ', '.join(missing) + '; install local wheels before running')
    group.mkdir(parents=True, exist_ok=False)
    (group / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n', encoding='utf-8')
    logs = group / 'logs'
    logs.mkdir()
    sdk_plan = plan['sdk_environment']
    env = diagnose._environment(sdk_plan)
    diagnose._command(plan['sdk_build'], sdk_plan, env, logs / 'sdk-build.log')
    maps = {}
    for variant in plan['variants']:
        # This internal trust applies only after the successful SDK command in
        # this same process. It is not a CLI flag or a persisted bypass option.
        mapping = diagnose.execute(variant, sdk_prepared=True)
        maps[variant['dispatch_impl']] = Path(os.path.relpath(mapping, group)).as_posix()
    index = dict(format='ll-complete-ab-wheel-maps-v1', experiment=plan['experiment'], wheel_maps=maps)
    destination = Path(plan['wheel_maps_output'])
    with destination.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(index, stream, indent=2)
        stream.write('\n')
    return destination


def main(argv=None):
    command = parser()
    args = command.parse_args(argv)
    try:
        plan = make_plan(args)
        if args.plan:
            print(json.dumps(plan, indent=2))
            return 0
        result = execute(plan)
        print('SDK and all requested OFF/sparse wheels built; no NPU work was launched.')
        print('Wheel map index: ' + str(result))
        for implementation in EXPERIMENTS[args.experiment]:
            print(implementation + ': ' + str(Path(plan['group']) / implementation / 'node_000/wheel-map.json'))
        return 0
    except (ValueError, RuntimeError) as error:
        command.exit(1, f'{error}\n')


if __name__ == '__main__':
    raise SystemExit(main())
