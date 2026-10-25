"""Compile the real Host selector for both algorithms and verify exported IDs."""
from pathlib import Path
import os
import shlex
import shutil
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[2]

@pytest.mark.parametrize('replay',[0,1])
@pytest.mark.parametrize('timer',[0,1])
def test_actual_bridge_selected_identity_and_all_modes(tmp_path,replay,timer):
    cc=shlex.split(os.environ.get('CXX',''))
    if not cc:
        found=shutil.which('g++') or shutil.which('clang++')
        if not found:pytest.skip('Host C++ compiler required')
        cc=[found]
    source=tmp_path/'selector.cpp'
    source.write_text(r'''#include <cassert>
#include "csrc/kernels/dispatch/low_latency/launch.hpp"
static unsigned calls=0,selected=99;
#define STUB(name,n) extern "C" void dispatch_server_dedup_launch_sparse15_##name(void*, const deepep::EpServerDedupKernelArgs*) {++calls;selected=n;}
STUB(prequant_e4m3,0) STUB(fp16,1) STUB(fp16_e4m3,2) STUB(fp16_e5m2,3)
STUB(bf16,4) STUB(bf16_e4m3,5) STUB(bf16_e5m2,6)
int main(){
 deepep::EpServerDedupKernelArgs args{};
 for(unsigned m=0;m<7;++m){calls=0;dispatch_server_dedup_kernel_do_sparse15(nullptr,&args,m);assert(calls==1&&selected==m);}
 assert(deepep_ll_dispatch_implementation()==EXPECTED_REPLAY);
 assert(deepep_ll_core_protocol_version()==(EXPECTED_REPLAY?21028U:19028U));
 assert(deepep_ll_ranks_per_group()==(EXPECTED_REPLAY?16U:4U));
 assert(deepep_ll_overlap_stage()==(EXPECTED_REPLAY?0U:3U));
 assert(deepep_ll_stage_timer_enabled()==EXPECTED_TIMER);
 assert(deepep_ll_stage_timer_schema()==15U&&deepep_ll_stage_timer_profile_id()==6U);
 assert(deepep_ll_small_b_prefix_cores()==0U);
}
''',encoding='utf-8')
    executable=tmp_path/('selector.exe' if os.name=='nt' else 'selector')
    flags=['-DDEEPEP_LL_STAGE_TIMER=1'] if timer else []
    subprocess.run([*cc,'-std=c++17','-O2','-UNDEBUG','-DCATLASS_ARCH=3510',
                    f'-DDEEPEP_LL_DISPATCH_CB59={replay}',f'-DEXPECTED_REPLAY={replay}',
                    f'-DEXPECTED_TIMER={timer}',*flags,'-iquote',str(ROOT),str(source),
                    str(ROOT/'csrc/kernels/dispatch/low_latency.cpp'),'-o',str(executable)],check=True)
    subprocess.run([str(executable)],check=True)

@pytest.mark.parametrize('replay',[0,1])
def test_actual_recorder_exports_selected_sender_role(tmp_path,replay):
    cc=shlex.split(os.environ.get('CXX',''))
    if not cc:
        found=shutil.which('g++') or shutil.which('clang++')
        if not found:pytest.skip('Host C++ compiler required')
        cc=[found]
    (tmp_path/'kernel_operator.h').write_text(r'''#pragma once
#include <cstdint>
#define __aicore__
#define __gm__
using GM_ADDR=uint8_t*;
namespace AscendC {
inline uint32_t core=0; inline uint64_t clock=1;
inline uint32_t GetBlockIdx(){return core;} inline uint32_t GetBlockNum(){return 56;}
inline uint64_t GetSystemCycle(){return clock++;} inline void Trap(){}
enum class CacheLine{SINGLE_CACHE_LINE}; enum class DcciDst{CACHELINE_OUT};
template<class T>struct GlobalTensor{void SetGlobalBuffer(T*,unsigned){}};
template<class T,CacheLine C,DcciDst D>void DataCacheCleanAndInvalid(GlobalTensor<T>&){}
}
''')
    source=tmp_path/'roles.cpp'
    source.write_text(r'''#include <cassert>
#include "csrc/kernels/dispatch/low_latency/common/ll_stage_timer_device.h"
int main(){
 using namespace LlStageTimer;
 uint64_t words[56*kWordsPerCore]{};
 for(unsigned op=1;op<=2;++op)for(unsigned core=0;core<56;++core){
   AscendC::core=core;
   DeviceTimer timer;timer.Init(reinterpret_cast<uint8_t*>(words),op,1);
   timer.Mark<ENTRY>();timer.Mark<INIT_DONE>();timer.Mark<WORK_DONE>();timer.Flush();
   unsigned expected=op==1?(core<28?(EXPECTED_REPLAY?1:3):2):(core<28?1:4);
   assert(((words[core*kWordsPerCore+5]>>16)&255)==expected);
   assert(words[core*kWordsPerCore+3]==56);
 }
}
''')
    executable=tmp_path/('roles.exe' if os.name=='nt' else 'roles')
    subprocess.run([*cc,'-std=c++17','-O2','-UNDEBUG','-DDEEPEP_LL_STAGE_TIMER=1',
                    f'-DDEEPEP_LL_DISPATCH_CB59={replay}',f'-DEXPECTED_REPLAY={replay}',
                    '-I',str(tmp_path),'-iquote',str(ROOT),str(source),'-o',str(executable)],check=True)
    subprocess.run([str(executable)],check=True)
