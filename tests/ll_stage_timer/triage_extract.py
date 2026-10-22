"""Export an explicitly selected timer subset in at most three text parts by default.

This is an offline follow-up to SEND_FIRST.txt. It does not sample, round,
renumber cores, rebuild code, or launch devices. Narrow the selectors when the
requested exact subset exceeds the transfer budget.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LIMIT = 60000
PROFILES = ('full', 'overview', 'prepare', 'prefix', 'send', 'recv')


def _module(name):
    spec = importlib.util.spec_from_file_location('ll_extract_' + name,
                                                 Path(__file__).with_name(name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _json(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _path(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _candidates(suite, profile, case):
    profile_root = suite / 'analysis' / profile
    if case is None:
        scope = profile_root
    else:
        value = Path(case)
        scope = (value if value.is_absolute() else suite / value).resolve()
        if scope != profile_root and profile_root not in scope.parents:
            raise ValueError('--case must be inside analysis/' + profile + ' in this suite')
    if scope.is_file():
        if scope.name != 'manifest.json' or scope.parent.name != 'transfer':
            raise ValueError('--case file must be a per-case transfer/manifest.json')
        return [scope]
    if not scope.is_dir():
        raise ValueError('No parsed case directory exists: ' + str(scope))
    if scope.name == 'transfer' and (scope / 'manifest.json').is_file():
        return [scope / 'manifest.json']
    return sorted(scope.rglob('transfer/manifest.json'))


def _sample_matches(metadata, decoded, profile, bs, rank, iteration):
    actual_profile = decoded.get('timer_profile', 'full' if decoded.get('schema', 0) <= 13 else None)
    return (actual_profile == profile and (metadata.get('config') or {}).get('B') == bs
            and (rank is None or metadata.get('rank') == rank)
            and (iteration is None or metadata.get('iteration') == iteration))


def _select_core(samples, core):
    result = copy.deepcopy(samples)
    if core is None:
        return result
    for metadata, decoded in result:
        available = {row.get('core') for key in ('records', 'intervals', 'cores')
                     for row in decoded.get(key, [])}
        if core not in available:
            raise ValueError(f'Core {core} is absent at rank={metadata.get("rank")}, '
                             f'iteration={metadata.get("iteration")}; narrow the selectors explicitly')
        for key in ('records', 'intervals', 'cores'):
            if key in decoded:
                decoded[key] = [row for row in decoded[key] if row.get('core') == core]
        # active_aiv and all other source identities remain untouched. This is
        # a partial view of the original launch, not a launch with fewer cores.
    return result


def export_selection(suite, profile, bs, output_dir, rank=None, iteration=None,
                     core=None, case=None, max_parts=3):
    suite, output = _path(suite), _path(output_dir)
    if output.exists():
        raise ValueError('Output directory already exists; choose a fresh --output-dir')
    if profile not in PROFILES:
        raise ValueError('Unknown timer profile')
    if type(bs) is not int or bs < 1:
        raise ValueError('--bs must be a positive integer')
    for name, value in (('rank', rank), ('iteration', iteration), ('core', core)):
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError('--' + name + ' must be a nonnegative integer')
    if rank is None or iteration is None:
        raise ValueError('Necessary follow-ups require explicit --rank and --iteration; --core is optional')
    if type(max_parts) is not int or not 1 <= max_parts <= 9:
        raise ValueError('--max-parts must be an integer in 1..9; the summary also consumes one of 10 sends')
    transfer = _module('transfer_report')
    matches = []
    for path in _candidates(suite, profile, case):
        context, samples = transfer.restore_samples(path.parent)
        selected = [(metadata, decoded) for metadata, decoded in samples
                    if _sample_matches(metadata, decoded, profile, bs, None, None)]
        if selected:
            matches.append((path, context, selected))
    if not matches:
        raise ValueError('No samples match the exact profile/B/rank/iteration selectors; nothing was exported')
    if len(matches) != 1:
        names = ', '.join(path.parent.parent.relative_to(suite).as_posix() for path, _, _ in matches)
        raise ValueError('Multiple cases/routes match; supply --case with one exact analysis directory: ' + names)
    source, context, samples = matches[0]
    samples = [(metadata, decoded) for metadata, decoded in samples
               if _sample_matches(metadata, decoded, profile, bs, rank, iteration)]
    if not samples:
        raise ValueError('No samples match the exact rank/iteration selectors; nothing was exported')
    source_manifest = json.loads(source.read_text(encoding='utf-8'))
    samples = _select_core(samples, core)
    selectors = dict(profile=profile, B=bs, rank=rank, iteration=iteration, core=core,
                     case=source.parent.parent.relative_to(suite).as_posix())
    selection = dict(explicit_subset=True, complete_archive=False, selectors=selectors,
                     source_manifest=source.relative_to(suite).as_posix(),
                     source_bundle_id=source_manifest.get('bundle_id'),
                     selected_samples=len(samples),
                     note='Exact selected rank and iteration; null core means all observed cores. '
                          'Original launch identities, active_aiv, rank and core numbers are preserved. '
                          'This subset does not establish full rank/core/iteration coverage.')
    context = copy.deepcopy(context)
    context['export_selection'] = selection
    raw = transfer.encode_samples(context, samples)
    transport, parts = transfer.codec.make_parts(raw, limit=LIMIT)
    if len(parts) > max_parts:
        raise ValueError(f'Exact selection requires {len(parts)} parts, exceeding --max-parts {max_parts}; '
                         'narrow --rank, --iteration and/or --core. No files were written and no data were dropped.')
    names = [f'part_{index:04d}_of_{len(parts):04d}.txt' for index in range(1, len(parts) + 1)]
    coverage, complete_cores = transfer._coverage(context, samples)
    manifest = dict(format=transfer.FORMAT, bundle_id=transport['raw_sha256'][:20], limit=LIMIT,
                    transport=transport, parts=names, send_first=names,
                    series_counts=dict(compressed=len(parts)), samples=len(samples),
                    stage_rows=sum(len(decoded.get('records', [])) for _, decoded in samples),
                    interval_rows=sum(len(decoded.get('intervals', [])) for _, decoded in samples),
                    export_complete=True, complete_archive=False, export_selection=selection,
                    source_sample_coverage=coverage, all_observed_kernel_cores_complete=complete_cores,
                    notes=['Lossless within the explicit selection only; not a complete suite/archive.',
                           'Separate profiles cannot be summed or aligned to one timeline.'])
    files = dict(zip(names, parts))
    if any(len(text) > LIMIT or len(text.encode('utf-8')) + text.count('\n') > LIMIT
           for text in files.values()):
        raise ValueError('Selection metadata exceeds the transfer budget; no files were written')
    digests = [hashlib.sha256(part.encode('utf-8')).hexdigest() for part in parts]
    with _module('transfer_budget').transaction(suite, 'detail', digests) as usage:
        manifest['send_first'] = [name for name, digest in zip(names, digests) if digest in usage['new_digests']]
        manifest['send_budget'] = usage
        # The manifest is local bookkeeping, never another transmission.
        files['manifest.json'] = _json(manifest) + '\n'
        output.mkdir(parents=True, exist_ok=False)
        for name, text in files.items():
            (output / name).write_bytes(text.encode('utf-8'))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('suite', type=Path)
    parser.add_argument('--profile', required=True, choices=PROFILES)
    parser.add_argument('--bs', required=True, type=int)
    parser.add_argument('--rank', required=True, type=int)
    parser.add_argument('--iteration', required=True, type=int)
    parser.add_argument('--core', type=int)
    parser.add_argument('--case', type=Path, help='One analysis case directory or transfer manifest, relative to suite')
    parser.add_argument('--output-dir', required=True, type=Path, help='Fresh directory; relative paths use project root')
    parser.add_argument('--max-parts', type=int, default=3, help='Batch ceiling 1..9 (default 3), also limited by the suite\'s remaining sends')
    args = parser.parse_args(argv)
    try:
        manifest = export_selection(args.suite, args.profile, args.bs, args.output_dir,
                                    rank=args.rank, iteration=args.iteration, core=args.core,
                                    case=args.case, max_parts=args.max_parts)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, str(exc) + '\n')
    usage = manifest['send_budget']
    print(f'Suite send budget: {usage["issued"]}/10 issued; {usage["remaining"]} remaining; {usage["reserved"]} summary slot reserved.')
    print(f'Explicit subset: {manifest["samples"]} samples; send only {len(manifest["send_first"])} new part files:')
    if usage['already_issued']:
        print('Identical content already issued: do not resend it. No manifest or index is needed.')
    for name in manifest['send_first']:
        print(_path(args.output_dir) / name)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
