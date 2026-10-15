"""Run the single-AIV diagnostic sequentially on physical devices 0 and 1."""
import argparse
import datetime
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--devices', default='0,1', choices=['0', '1', '0,1'])
    parser.add_argument('--requests', default='0,1,4,8,12,13,14,15,16,17,18,19,20,24,32')
    parser.add_argument('--chunk', default='128,512,2048,4096')
    parser.add_argument('--stride', default='65536')
    parser.add_argument('--mode', default='batch,serial')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--tail', type=int, default=50)
    parser.add_argument('--output-dir')
    parser.add_argument('--timeout', type=int, default=1800)
    args = parser.parse_args()
    if args.warmup < 0 or not 1 <= args.tail <= args.iters or args.timeout <= 0:
        parser.error('Require warmup>=0, 1<=tail<=iters and timeout>0')
    for key in ('PROBE_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES',
                'ASCEND_RT_VISIBLE_DEVICES', 'ASCEND_VISIBLE_DEVICES'):
        if os.environ.get(key) != '0,1':
            parser.error(f'{key} must be 0,1; use run_a5.sh')
    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    output = Path(args.output_dir or f'results/mte3-depth-{stamp}')
    output = (ROOT / output).resolve() if not output.is_absolute() else output.resolve()
    if not output.is_relative_to(ROOT):
        parser.error('Output must remain inside the repository')
    binary = HERE / 'build/mte3_depth'
    if not binary.is_file():
        parser.error('Missing executable; run build.sh first')
    identity_path = HERE / 'build/identity.json'
    if not identity_path.exists():
        parser.error('Missing build identity; rebuild with build.sh')
    identity = json.loads(identity_path.read_text())
    for name, expected in identity['files'].items():
        if hashlib.sha256((HERE / name).read_bytes()).hexdigest() != expected:
            parser.error(f'Build input/artifact changed: {name}; rebuild with build.sh')
    output.mkdir(parents=True, exist_ok=False)
    evidence = output / 'build-evidence'
    evidence.mkdir()
    for name in ('kernel.cpp','host.cpp','build.sh','build/kernel.dis',
                 'build/identity.json','build/asm-windows.json','build/asm-audit.log'):
        shutil.copy2(HERE / name, evidence / Path(name).name)
    manifest = dict(vars(args), block_dim=1, core_ratio='0:1',
                    purpose='single-AIV diagnostic, not LL performance baseline',
                    clock_unit='raw GetSystemCycle ticks; no frequency conversion',
                    cache_condition=identity['l2_cache'] if identity else 'sdk_default_unrecorded_build',
                    build_identity=identity,
                    commands=[], completed=False)
    manifest_path = output / 'manifest.json'
    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    save()
    try:
        for device in args.devices.split(','):
            command = [str(binary), '--device', device, '--requests', args.requests,
                       '--chunk', args.chunk, '--stride', args.stride,
                       '--mode', args.mode, '--warmup', str(args.warmup),
                       '--iters', str(args.iters), '--output', str(output / f'device-{device}.csv')]
            manifest['commands'].append(command)
            save()
            print(f'Running device {device}, one AIV, sequential validation', flush=True)
            with (output / f'device-{device}.log').open('w', encoding='utf-8') as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=args.timeout)
        with (output / 'analysis.log').open('w', encoding='utf-8') as log:
            subprocess.run([sys.executable, str(HERE / 'analyze.py'), '--input-dir', str(output),
                            '--output-dir', str(output / 'analysis'), '--tail', str(args.tail)],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        manifest['completed'] = True
    except Exception as exc:
        manifest['error'] = str(exc)
        raise
    finally:
        save()
        print(f'Results: {output}', flush=True)


if __name__ == '__main__':
    main()
