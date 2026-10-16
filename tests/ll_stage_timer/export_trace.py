"""Export LL timer records for Chrome tracing / Perfetto, without importing Torch."""
import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('ll_stage_timer', ROOT / 'deep_ep/utils/ll_stage_timer.py')
timer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(timer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='Raw sample JSON or directory; partial ranks supported')
    parser.add_argument('--frequency-hz', type=float, help='Explicit GetSystemCycle frequency; otherwise use saved value')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rank', type=int, help='Select one rank')
    parser.add_argument('--iteration', type=int, help='Select one diagnostic iteration')
    parser.add_argument('--operation', choices=('dispatch', 'combine'))
    args = parser.parse_args()
    source = args.source if args.source.is_absolute() else ROOT / args.source
    output = args.output if args.output.is_absolute() else ROOT / args.output
    paths = sorted(source.rglob('sample_*.json')) if source.is_dir() else [source]
    saved = []
    for path in paths:
        record = json.loads(path.read_text(encoding='utf-8'))
        meta = record.get('metadata', {})
        if args.rank is not None and meta.get('rank') != args.rank:
            continue
        if args.iteration is not None and meta.get('iteration') != args.iteration:
            continue
        if args.operation is not None and record['operation'] != args.operation:
            continue
        saved.append(record)
    trace = timer.chrome_trace(saved, args.frequency_hz)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(trace, stream)
    print(f'Chrome trace: {output} ({len(saved)} samples; independent rank/sample origins)')


if __name__ == '__main__':
    main()
