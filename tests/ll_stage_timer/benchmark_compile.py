"""Cold LL dispatch compilation, all modes, no NPU execution."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import time
ROOT=Path(__file__).resolve().parents[2]
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--jobs',type=int,default=4)
    parser.add_argument('--timer',choices=('ON','OFF'),default='ON')
    parser.add_argument('--source-headers',action='store_true',help='Compile-only with pinned SHMEM source headers; does not validate SDK linking')
    args=parser.parse_args()
    if not 1<=args.jobs<=64: parser.error('jobs must be 1..64')
    out=Path(args.output);out=out if out.is_absolute() else ROOT/out
    out.mkdir(parents=True,exist_ok=False)
    cann=Path(os.environ['ASCEND_HOME_PATH']);sdk=ROOT/('third-party/shmem' if args.source_headers else 'third-party/shmem/install/shmem')
    flags=['-O2','-std=c++17','-fPIC','-xasc','--npu-arch=dav-3510','--cce-auto-infer-kernel-type=false',
           '-DCATLASS_ARCH=3510','-DUSE_SIMT','-DL2_CACHE_HINT',
           '-DDEEPEP_LL_SEND_CORES='+os.environ.get('DEEPEP_LL_SEND_CORES','28'),
           '-DDEEPEP_LL_RECV_CORES='+os.environ.get('DEEPEP_LL_RECV_CORES','28')]
    if args.timer=='ON':flags.append('-DDEEPEP_LL_STAGE_TIMER=1')
    for path in [cann/'include',cann/'asc/include',sdk/'include',sdk/'src',sdk/'src/device',sdk/'src/device_simt']:
        flags+=['-I'+str(path)]
    sources=[ROOT/'csrc/kernels/dispatch/low_latency.cpp']
    sources+=sorted((ROOT/'csrc/kernels/dispatch/low_latency/variants').glob('*.cpp'))
    def compile_one(source):
        start=time.perf_counter()
        command=['bisheng',*flags,'-c',str(source),'-o',str(out/(source.stem+'.o'))]
        with (out/(source.stem+'.log')).open('w') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        row=dict(source=str(source.relative_to(ROOT)),seconds=time.perf_counter()-start,exit_code=result.returncode)
        print(json.dumps(row),flush=True)
        return row
    start=time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool: rows=list(pool.map(compile_one,sources))
    report=dict(timer=args.timer,source_headers=args.source_headers,jobs=args.jobs,seconds=time.perf_counter()-start,flags=flags,
        revision=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),units=rows)
    (out/'timing.json').write_text(json.dumps(report,indent=2))
    print('COMPILE_RESULT '+json.dumps(report),flush=True)
    if any(row['exit_code'] for row in rows):raise SystemExit(1)
if __name__=='__main__':main()
