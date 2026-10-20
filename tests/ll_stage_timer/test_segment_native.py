"""Compile the actual segment recorder with a minimal Ascend API mock when C++ is available."""
from pathlib import Path
import shutil
import subprocess
import pytest
ROOT=Path(__file__).resolve().parents[2]

@pytest.mark.parametrize('enabled',[False,True])
def test_actual_recorder_delays_gm_export_and_preserves_kernel(tmp_path,enabled):
    compiler=shutil.which('g++')
    if compiler is None:pytest.skip('C++ compiler unavailable; target native mock remains required')
    (tmp_path/'kernel_operator.h').write_text(r'''
#pragma once
#include <cstdint>
#include <stdexcept>
#define __aicore__
#define __gm__
using GM_ADDR=uint8_t*;
namespace AscendC {
inline uint64_t clock=1; inline uint32_t core=0; inline unsigned flushes=0;
inline uint64_t GetSystemCycle(){return clock++;}
inline uint32_t GetBlockIdx(){return core;}
inline uint32_t GetBlockNum(){return 56;}
inline void Trap(){throw std::runtime_error("trap");}
enum class CacheLine{SINGLE_CACHE_LINE};enum class DcciDst{CACHELINE_OUT};
template<class T>struct GlobalTensor{void SetGlobalBuffer(T*,unsigned){}};
template<class T,CacheLine C,DcciDst D>void DataCacheCleanAndInvalid(GlobalTensor<T>&){++flushes;}
}
''')
    source=tmp_path/'check.cpp';source.write_text(r'''
#include <algorithm>
#include <cassert>
#include <vector>
#include "csrc/kernels/dispatch/low_latency/common/ll_stage_timer_device.h"
int main(){
 using namespace LlStageTimer;
 std::vector<uint64_t> words(56*kWordsPerCore+64,0);
 std::fill(words.end()-64,words.end(),0x12345678ULL);
 AscendC::core=55;
 DeviceTimer timer;timer.Init(reinterpret_cast<uint8_t*>(words.data()),1,1);
 auto kernel=timer.Begin(KERNEL);
 for(uint32_t stage=1;stage<64;++stage){
  if(stage>=51 && stage<=55){timer.Count(static_cast<Stage>(stage),17);continue;}
  for(unsigned i=0;i<5;++i){auto begin=timer.Begin(static_cast<Stage>(stage));timer.End(static_cast<Stage>(stage),begin);}
 }
 timer.End(KERNEL,kernel);
 assert(std::all_of(words.begin(),words.end()-64,[](auto value){return value==0;}));
 assert(AscendC::flushes==0);
 timer.Flush();
#ifdef DEEPEP_LL_STAGE_TIMER
 assert(AscendC::flushes>0);
 auto* data=words.data()+55*kWordsPerCore;
 assert(data[0]==kMagic && data[1]==7 && data[6]<=256 && data[7]>0);
 unsigned kernelBegins=0,kernelEnds=0,kernelMetrics=0;
 for(unsigned i=0;i<data[6];++i){
  auto event=uint32_t(data[8+i*2]);
  kernelBegins+=event==0;kernelEnds+=event==1;kernelMetrics+=event==0x8000;
 }
 assert(kernelBegins==1 && kernelEnds==1 && kernelMetrics==1);
 assert(std::all_of(words.begin(),words.begin()+55*kWordsPerCore,[](auto value){return value==0;}));
#else
 assert(AscendC::flushes==0);
 assert(std::all_of(words.begin(),words.end()-64,[](auto value){return value==0;}));
#endif
 assert(std::all_of(words.end()-64,words.end(),[](auto value){return value==0x12345678ULL;}));
 DeviceTimer disabled;disabled.Init(nullptr,1,1);auto begin=disabled.Begin(KERNEL);disabled.End(KERNEL,begin);disabled.Flush();
}
''')
    executable=tmp_path/'check'
    subprocess.run([compiler,'-std=c++17','-Wall','-Wextra','-Werror',
                    *(['-DDEEPEP_LL_STAGE_TIMER=1'] if enabled else []),f'-I{tmp_path}',f'-I{ROOT}',str(source),'-o',str(executable)],check=True)
    subprocess.run([str(executable)],check=True,timeout=30)
