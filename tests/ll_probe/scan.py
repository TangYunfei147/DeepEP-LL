"""Plan or run a unidirectional two-device event-timed transport scan."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

ROOT = Path('/home/tyf/Q4_remote_test')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bytes', type=int, nargs='+', default=[32, 128, 512, 2048, 8192, 32768, 131072, 524288, 2097152, 8388608, 33554432, 67108864])
    p.add_argument('--cores', type=int, nargs='+', default=[1, 8, 32])
    p.add_argument('--lanes', type=int, nargs='+', default=[1, 4])
    p.add_argument('--chunk', type=int, default=4096)
    p.add_argument('--modes', type=int, nargs='+', default=[0, 1, 2, 3])
    p.add_argument('--samples', type=int, default=50)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--peak-gbps', type=float, help='Matched one-way reference peak in decimal GB/s')
    p.add_argument('--execute', action='store_true', help='User-run target-machine measurement')
    p.add_argument('--validation-only', action='store_true', help='Execute receiver checks, collect zero timing samples')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    if any(b <= 0 or b % 32 or b > 64 << 20 for b in args.bytes): p.error('bytes must be aligned and <=64MiB')
    if any(c < 1 or c > 32 for c in args.cores): p.error('cores must be 1..32')
    if any(l < 1 or l > 4 for l in args.lanes): p.error('lanes must be 1..4')
    if any(m not in range(4) for m in args.modes): p.error('modes must be 0..3')
    if args.chunk <= 0 or args.chunk % 32 or args.chunk > 32768: p.error('invalid chunk')
    if not 1 <= args.samples <= 10000 or not 0 <= args.warmup <= 10000: p.error('invalid repetition counts')
    if args.peak_gbps is not None and (not math.isfinite(args.peak_gbps) or args.peak_gbps <= 0): p.error('invalid peak')
    cases = [dict(bytes=b, cores=c, lanes=l, chunk=args.chunk, mode=m)
             for b in args.bytes for c in args.cores for l in args.lanes for m in args.modes]
    plan = {'direction': '0->1', 'timing': 'ACL event interval, includes event/stream overhead; not profiler kernel task',
            'cases': cases, 'samples': 0 if args.validation_only else args.samples,
            'reference_peak_gbps': args.peak_gbps, 'scope': 'transport probe, not native LL'}
    plan.update(measurement='legacy_acl_event_diagnostic', reference_protocol=False, profiler_entry='profile_probe.py; see PROFILING.md')
    if not args.execute:
        print(json.dumps(plan, indent=2)); return
    if args.output is None: p.error('--execute requires fresh --output under authorized root')
    output = args.output.resolve()
    if not output.is_relative_to(ROOT) or output == ROOT: p.error('output outside authorized root')
    output.mkdir(parents=True, exist_ok=False)
    binary = Path(__file__).resolve().parent / 'build/peer_probe'
    plan['binary_sha256'] = hashlib.sha256(binary.read_bytes()).hexdigest()
    (output/'plan.json').write_text(json.dumps(plan, indent=2))
    reports = []
    for index, case in enumerate(cases):
        command = [str(binary), *(str(case[k]) for k in ['bytes','cores','lanes','chunk','mode']),
                   str(0 if args.validation_only else args.warmup), str(plan['samples'])]
        result = subprocess.run(command, capture_output=True, text=True, timeout=90)
        (output/f'{index:04d}.csv').write_text(result.stdout)
        (output/f'{index:04d}.log').write_text(result.stderr)
        if result.returncode or 'normal cleanup' not in result.stderr:
            raise RuntimeError(f'case {index} failed: returncode={result.returncode}; see raw logs')
        rows = list(csv.DictReader(result.stdout.splitlines()))
        if len(rows) != plan['samples']: raise ValueError('Incomplete sample set')
        event = [float(row['event_us']) for row in rows]
        host = [float(row['host_us']) for row in rows]
        if any(not math.isfinite(t) or t <= 0 for t in event+host): raise ValueError('Invalid sample')
        report = {**case, 'correctness': 'passed', 'samples':len(rows)}
        if event:
            mean = statistics.mean(event)
            report.update(event_mean_us=mean, event_median_us=statistics.median(event), host_mean_us=statistics.mean(host))
            if case['mode'] >= 2:
                gbps = case['bytes']/(mean*1000)
                report['wire_gbps'] = gbps
                if args.peak_gbps is not None: report['attainment'] = gbps/args.peak_gbps
        reports.append(report)
        print(f'case {index+1}/{len(cases)} passed', flush=True)
    (output/'summary.json').write_text(json.dumps({'plan':plan,'results':reports,'status':'passed'},indent=2))

if __name__ == '__main__':
    main()
