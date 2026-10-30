"""Cold LL dispatch compilation, all modes, no NPU execution."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import time
ROOT=Path(__file__).resolve().parents[2]
_DISPATCH=runpy.run_path(str(ROOT/'scripts/ll_dispatch_implementation.py'))
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--jobs',type=int,default=4)
    parser.add_argument('--reference-profile',choices=('ON','OFF'),default='ON')
    parser.add_argument('--source-headers',action='store_true',help='Compile-only with pinned SHMEM source headers; does not validate SDK linking')
    args=parser.parse_args()
    if not 1<=args.jobs<=64: parser.error('jobs must be 1..64')
    send=int(os.environ.get('DEEPEP_LL_SEND_CORES','28'))
    recv=int(os.environ.get('DEEPEP_LL_RECV_CORES','28'))
    if send<=0 or recv<=0 or send+recv!=56:parser.error('SEND_CORES and RECV_CORES must be positive and sum to 56')
    prefix=os.environ.get('DEEPEP_LL_SMALL_B_PREFIX_CORES','0')
    if not re.fullmatch(r'0|[1-9][0-9]?',prefix) or int(prefix)>recv:
        parser.error('DEEPEP_LL_SMALL_B_PREFIX_CORES must be 0 or an integer in [1, RECV_CORES]')
    try:
        dispatch_impl=_DISPATCH['dispatch_implementation']()
        identity=_DISPATCH['dispatch_configuration'](dispatch_impl,send,recv,int(prefix))
    except ValueError as error:
        parser.error(str(error))
    out=Path(args.output);out=out if out.is_absolute() else ROOT/out
    out.mkdir(parents=True,exist_ok=False)
    cann=Path(os.environ['ASCEND_HOME_PATH']);sdk=ROOT/('third-party/shmem' if args.source_headers else 'third-party/shmem/install/shmem')
    flags=['-O2','-std=c++17','-fPIC','-xasc','--npu-arch=dav-3510','--cce-auto-infer-kernel-type=false',
           '-DCATLASS_ARCH=3510','-DUSE_SIMT','-DL2_CACHE_HINT',
           '-DDEEPEP_LL_SEND_CORES='+str(send),
           '-DDEEPEP_LL_RECV_CORES='+str(recv),
           '-DDEEPEP_LL_SMALL_B_PREFIX_CORES='+prefix,
           '-DDEEPEP_LL_DISPATCH_CB59='+str(identity['cb59'])]
    if args.reference_profile=='ON':flags.append('-DDEEPEP_LL_REFERENCE_PROFILE=1')
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
    report=dict(reference_profile=args.reference_profile,schema=7,dispatch_impl=dispatch_impl,source_headers=args.source_headers,jobs=args.jobs,seconds=time.perf_counter()-start,flags=flags,
        revision=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),units=rows)
    (out/'timing.json').write_text(json.dumps(report,indent=2))
    print('COMPILE_RESULT '+json.dumps(report),flush=True)
    if any(row['exit_code'] for row in rows):raise SystemExit(1)
if __name__=='__main__':main()
