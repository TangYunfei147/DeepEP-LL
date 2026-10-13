"""A5-only smoke checks: physical devices 0/1, tiny Exp scratch, three captures."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/tyf/Q4_remote_test')
CASES = {
    'metadata': ['--kind', 'metadata'],
    'empty': ['--kind', 'comm', '--layer', '0'],
    'tiling': ['--kind', 'comm', '--layer', '1'],
    'comm': ['--kind', 'comm', '--layer', '3', '--scenario', '4'],
    'peer': ['--kind', 'peer'],
    'shmem': ['--kind', 'shmem'],
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases', choices=tuple(CASES), nargs='+', default=list(CASES))
    args = p.parse_args()
    out = Path(tempfile.mkdtemp(prefix='reference-profile-', dir=ROOT/'project_control/results'))
    env = dict(os.environ)
    env.update(PROBE_VISIBLE_DEVICES='0,1', CUDA_VISIBLE_DEVICES='0,1',
               ASCEND_RT_VISIBLE_DEVICES='0,1', ASCEND_VISIBLE_DEVICES='0,1')
    print(out, flush=True)
    for i, case in enumerate(args.cases):
        distributed = case in ('empty', 'tiling', 'comm', 'shmem')
        cmd = ['bash', str(HERE/'four_layers/run.sh'), 'python3']
        if distributed:
            cmd += ['-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2']
        cmd += [str(HERE/'profile_probe.py'), *CASES[case], '--output-dir', str(out/case),
                '--warmup', '2', '--samples', '3', '--tail', '2', '--delay-rows', '64', '--delay-cols', '64']
        if distributed:
            cmd += ['--endpoint', f'tcp://127.0.0.1:{19770+i}']
        with (out/f'{case}.log').open('w') as stream:
            subprocess.run(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=240)
        reports = sorted((out/case).glob('rank_*/report.json'))
        subprocess.run(['python3', str(HERE/'analyze_profile.py'), *map(str, reports),
                        '--output', str(out/f'{case}-summary.json')], env=env, check=True)
        print(f'PASS {case}: native correctness + profiler task count + all-rank summary', flush=True)


if __name__ == '__main__':
    main()
