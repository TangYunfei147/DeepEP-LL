"""Compile the real Host selector for both algorithms and verify exported IDs."""
from pathlib import Path
import os
import shlex
import shutil
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[2]

@pytest.mark.parametrize('replay',[0,1,2,3,4])
@pytest.mark.parametrize('timer',[0,1])
def test_actual_bridge_selected_identity_and_all_modes(tmp_path,replay,timer):
    cc=shlex.split(os.environ.get('CXX',''))
    if not cc:
        found=shutil.which('g++') or shutil.which('clang++')
        if not found:pytest.skip('Host C++ compiler required')
        cc=[found]
    source=tmp_path/'selector.cpp'
    source.write_text(r'''#include <cassert>
#include <cstring>
#include "csrc/kernels/dispatch/low_latency/launch.hpp"
#include "csrc/kernels/dispatch/low_latency/common/ll_dispatch_implementation.h"
static unsigned calls=0,selected=99;
#define STUB(name,n) extern "C" void dispatch_server_dedup_launch_reference7_##name(void*, const deepep::EpServerDedupKernelArgs*) {++calls;selected=n;}
STUB(prequant_e4m3,0) STUB(fp16,1) STUB(fp16_e4m3,2) STUB(fp16_e5m2,3)
STUB(bf16,4) STUB(bf16_e4m3,5) STUB(bf16_e5m2,6)
int main(){
 deepep::EpServerDedupKernelArgs args{};
 for(unsigned m=0;m<7;++m){calls=0;dispatch_server_dedup_kernel_do_reference7(nullptr,&args,m);assert(calls==1&&selected==m);}
 assert(deepep_ll_dispatch_implementation()==EXPECTED_REPLAY);
 assert(DispatchDedup::kLlCb59Replay==(EXPECTED_REPLAY!=0));
 assert(DispatchDedup::kLlCb59Stream==(EXPECTED_REPLAY==2));
 assert(DispatchDedup::kLlCb59Pair==(EXPECTED_REPLAY==3));
 assert(DispatchDedup::kLlCb59Small==(EXPECTED_REPLAY==4));
 const char* selections[]={"current","cb59","cb59-stream","cb59-pair","cb59-small"};
 const char* names[]={"g4-small-b-parallel-prefix","cb59-replay","cb59-stream-recv","cb59-pair-recv","cb59-small-direct"};
 const unsigned protocols[]={19028U,21028U,22028U,23028U,24028U};
 assert(std::strcmp(DispatchDedup::kLlDispatchImplementation,selections[EXPECTED_REPLAY])==0);
 assert(std::strcmp(DispatchDedup::kLlImplementationName,names[EXPECTED_REPLAY])==0);
 assert(deepep_ll_core_protocol_version()==protocols[EXPECTED_REPLAY]);
 assert(deepep_ll_ranks_per_group()==(EXPECTED_REPLAY?16U:4U));
 assert(deepep_ll_overlap_stage()==(EXPECTED_REPLAY?0U:3U));
 assert(deepep_ll_reference_profile_enabled()==EXPECTED_TIMER);
 assert(deepep_ll_reference_profile_schema()==7U);
 assert(deepep_ll_small_b_prefix_cores()==0U);
}
''',encoding='utf-8')
    executable=tmp_path/('selector.exe' if os.name=='nt' else 'selector')
    flags=['-DDEEPEP_LL_REFERENCE_PROFILE=1'] if timer else []
    subprocess.run([*cc,'-std=c++17','-O2','-UNDEBUG','-DCATLASS_ARCH=3510',
                    f'-DDEEPEP_LL_DISPATCH_CB59={replay}',f'-DEXPECTED_REPLAY={replay}',
                    f'-DEXPECTED_TIMER={timer}',*flags,'-iquote',str(ROOT),str(source),
                    str(ROOT/'csrc/kernels/dispatch/low_latency.cpp'),'-o',str(executable)],check=True)
    subprocess.run([str(executable)],check=True)


def test_reference7_bridge_cannot_link_legacy_sparse16_device_symbols(tmp_path):
    cc=shlex.split(os.environ.get('CXX',''))
    if not cc:
        found=shutil.which('g++') or shutil.which('clang++')
        if not found:pytest.skip('Host C++ compiler required for incompatible link check')
        cc=[found]
    source=tmp_path/'legacy_device.cpp'
    source.write_text(r'''#include "csrc/kernels/dispatch/low_latency/launch.hpp"
#define LEGACY(name) extern "C" void dispatch_server_dedup_launch_sparse16_##name(void*,const deepep::EpServerDedupKernelArgs*){}
LEGACY(prequant_e4m3) LEGACY(fp16) LEGACY(fp16_e4m3) LEGACY(fp16_e5m2)
LEGACY(bf16) LEGACY(bf16_e4m3) LEGACY(bf16_e5m2)
int main(){deepep::EpServerDedupKernelArgs args{};dispatch_server_dedup_kernel_do_reference7(nullptr,&args,0);}
''')
    executable=tmp_path/('mixed.exe' if os.name=='nt' else 'mixed')
    result=subprocess.run([*cc,'-std=c++17','-O2','-DCATLASS_ARCH=3510','-iquote',str(ROOT),str(source),
        str(ROOT/'csrc/kernels/dispatch/low_latency.cpp'),'-o',str(executable)],capture_output=True,text=True,timeout=180)
    assert result.returncode!=0 and 'dispatch_server_dedup_launch_reference7' in result.stderr
