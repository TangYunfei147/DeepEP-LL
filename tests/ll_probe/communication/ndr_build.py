"""Build NDR from the same communication source using a matching SDK backend."""
import argparse
import json
import os
import platform
from pathlib import Path
import subprocess
from ndr_support import ROOT,HERE,sdk_backend,identity,digest

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compile-check',choices=('XSCALE','HNS_1825'),help='Compile/link only; does not install a runnable NDR manifest')
    args=p.parse_args()
    backend=args.compile_check or sdk_backend()
    if backend is None:
        print('NDR skipped: SHMEM RDMA disabled. UB Memory build remains available.');return
    cann=Path(os.environ['ASCEND_HOME_PATH']);source=ROOT/'third-party/shmem'
    lib=source/'install/shmem/lib';compiler=cann/'tools/bisheng_compiler/bin/bisheng'
    output=HERE/'build'
    if args.compile_check:output=output/'ndr-compile-check'/backend
    output.mkdir(parents=True,exist_ok=True)
    flags=['-O2','-std=c++17','-fPIC','-DLL_PROBE_NDR','-DACLSHMEM_RDMA_V2_SUPPORT',
           f'-DACLSHMEMI_RDMA_K_BACKEND_{backend}=1',f'-ffile-prefix-map={ROOT}=.',f'-ffile-prefix-map={cann}=CANN_SDK']
    inc=[f'-I{x}' for x in (cann/'asc/include',cann/'include',cann/f'{platform.machine()}-linux/include/hcomm',cann/'include/hcomm',source/'include',source/'src',source/'src/device')]
    link=['--cce-fatobj-link',f'-L{lib}',f'-Wl,-rpath,{lib}',f'-L{cann}/lib64',
          '-lshmem','-lascendcl','-lruntime','-lplatform','-ltiling_api','-lunified_dlog']
    obj=output/'comm_ndr.o'
    subprocess.run([str(compiler),*flags,*inc,'-xasc','--npu-arch=dav-3510','--cce-auto-infer-kernel-type=false',
                    '-c',str(HERE/'kernel/comm_kernel.cpp'),'-o',str(obj)],check=True)
    library=output/'libcomm_ndr_profile.so'
    subprocess.run([str(compiler),*flags,*inc,'-shared','-DLL_PROFILE_BRIDGE',str(HERE/'host/comm_host.cpp'),
                    str(HERE.parent/'profiling/host/profile_bridge.cpp'),str(obj),*link,'-o',str(library)],check=True)
    subprocess.run([str(compiler),*flags,*inc,str(HERE/'host/comm_host.cpp'),str(obj),*link,'-o',str(output/'comm_ndr_probe')],check=True)
    if args.compile_check:
        print(f'PASS compile/link only: {backend}; SDK/NIC execution not validated');return
    (output/'ndr-build.json').write_text(json.dumps(dict(identity=identity(backend),library_sha256=digest(library)),indent=2)+'\n')
    print(f'NDR ready build: {backend}; actual NIC execution still requires validation')
if __name__=='__main__':main()
