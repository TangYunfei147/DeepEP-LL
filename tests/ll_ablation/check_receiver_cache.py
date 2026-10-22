"""CANN object-cache and Host syntax check; no link, SDK build, install or NPU run."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sysconfig
import time
from common import ROOT,write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output if a.output.is_absolute() else ROOT/a.output
    out.mkdir(parents=True,exist_ok=False)
    header=ROOT/'csrc/kernels/dispatch/low_latency/common/ll_ablation_config.h'
    original=header.read_bytes()
    cann=Path(os.environ['ASCEND_HOME_PATH']);compiler=shutil.which('bisheng')
    if not compiler:raise RuntimeError('bisheng unavailable')
    cmake=shutil.which('cmake')
    if not cmake:raise RuntimeError('cmake unavailable')
    shmem=ROOT/'third-party/shmem';rows=[]
    def run(command,name):
        start=time.perf_counter()
        with (out/(name+'.log')).open('w') as log:
            rc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
        rows.append(dict(stage=name,seconds=time.perf_counter()-start,exit_code=rc))
        write_json(out/'timing.json',rows)
        if rc:raise RuntimeError(f'{name} failed; see {out/(name+".log")}')
    try:
        header.write_text(original.decode().replace('kLlAblationMode = 0;','kLlAblationMode = 5;'),newline='\n')
        src=out/'project';src.mkdir()
        def q(path):return '[=['+str(path).replace('\\','/')+']=]'
        includes=[ROOT/'csrc',cann/'include',cann/'asc/include',shmem/'include',shmem/'src',shmem/'src/device',shmem/'src/device_simt']
        cmakelists='cmake_minimum_required(VERSION 3.20)\nproject(receiver_cache LANGUAGES CXX)\n'
        cmakelists+='add_library(receiver OBJECT '+q(ROOT/'csrc/kernels/dispatch/low_latency/variants/prequant_e4m3.cpp')+')\n'
        cmakelists+='target_compile_features(receiver PRIVATE cxx_std_17)\n'
        cmakelists+='target_compile_options(receiver PRIVATE -O2 -fPIC -xasc --npu-arch=dav-3510 --cce-auto-infer-kernel-type=false)\n'
        cmakelists+='target_compile_definitions(receiver PRIVATE CATLASS_ARCH=3510 USE_SIMT L2_CACHE_HINT DEEPEP_LL_STAGE_TIMER=1 DEEPEP_LL_SEND_CORES=30 DEEPEP_LL_RECV_CORES=26)\n'
        cmakelists+='target_include_directories(receiver PRIVATE '+' '.join(map(q,includes))+')\n'
        (src/'CMakeLists.txt').write_text(cmakelists)
        build=out/'build'
        run([cmake,'-S',str(src),'-B',str(build),'-G','Unix Makefiles','-DCMAKE_CXX_COMPILER='+compiler],'configure')
        command=[cmake,'--build',str(build),'--parallel','1']
        run(command,'cold')
        objects=list(build.rglob('prequant_e4m3.cpp.o'))
        if len(objects)!=1:raise RuntimeError('Cannot identify compiled object')
        stamp=objects[0].stat().st_mtime_ns
        run(command,'unchanged')
        if objects[0].stat().st_mtime_ns!=stamp:raise RuntimeError('Unchanged build unexpectedly recompiled')
        header.write_text(header.read_text()+'\n// depfile validation\n',newline='\n')
        run(command,'changed_header')
        if objects[0].stat().st_mtime_ns==stamp:raise RuntimeError('Header dependency did not rebuild object')
        try:
            import pybind11
            pybind=Path(pybind11.get_include())
        except ImportError:
            import torch
            pybind=Path(torch.__file__).parent/'include'
        host=[compiler,'-std=c++17','-fsyntax-only','-DDEEPEP_LL_STAGE_TIMER=1',
              '-DDEEPEP_LL_SEND_CORES=30','-DDEEPEP_LL_RECV_CORES=26']
        for path in [*includes,Path(sysconfig.get_path('include')),pybind]:host+=['-I',str(path)]
        host+=[str(ROOT/'csrc/ops/low_latency.cpp')]
        run(host,'host_syntax')
        print(json.dumps(rows,indent=2));print('CACHE_CHECK_PASS',out)
    finally:header.write_bytes(original)

if __name__=='__main__':main()
