"""Compile the actual segment recorder with a minimal Ascend API mock when C++ is available."""
from pathlib import Path
import shutil
import subprocess
import pytest
ROOT=Path(__file__).resolve().parents[2]

@pytest.mark.parametrize('profile',range(6))
@pytest.mark.parametrize('enabled',[False,True])
def test_actual_recorder_delays_gm_export_and_preserves_kernel(tmp_path,enabled,profile):
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
enum class HardEvent { MTE3_S };
struct TPipe { int FetchEventID(HardEvent){return 0;} };
template<HardEvent E> void SetFlag(int){}
template<HardEvent E> void WaitFlag(int){}
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
 for(uint32_t stage=1;stage<kMetrics;++stage){
  if(stage==BEFORE_FIRST_SEND || stage==BEFORE_FIRST_RECV_POLL || stage==SEND_SUBMIT_END)continue;
  if((stage>=51 && stage<=55) || stage==69 || (stage>=74 && stage<=77) || (stage>=103 && stage<=107)){timer.Count(static_cast<Stage>(stage),17);continue;}
  for(unsigned i=0;i<5;++i){auto begin=timer.Begin(static_cast<Stage>(stage));timer.End(static_cast<Stage>(stage),begin);}
 }
 timer.First(BEFORE_FIRST_SEND);timer.First(BEFORE_FIRST_SEND);
 timer.First(BEFORE_FIRST_RECV_POLL);timer.First(BEFORE_FIRST_RECV_POLL);
 auto submitted=timer.Begin(SEND_SUBMIT_END);timer.End(SEND_SUBMIT_END,submitted);
 timer.End(KERNEL,kernel);
 assert(std::all_of(words.begin(),words.end()-64,[](auto value){return value==0;}));
 assert(AscendC::flushes==0);
 timer.Flush();
#ifdef DEEPEP_LL_STAGE_TIMER
 assert(AscendC::flushes>0);
 auto* data=words.data()+55*kWordsPerCore;
 assert(((data[5]>>24)&255)==kProfileId);
 assert(data[0]==kMagic && data[1]==kSchema && data[6]<=kRecords && data[7]>0);
 unsigned kernelBegins=0,kernelEnds=0,kernelMetrics=0,firstSend=0,firstPoll=0;
 static_assert(kWordsPerCore*8==5632);
 for(unsigned i=0;i<data[6];++i){
  auto event=uint32_t(data[8+i*2]);
  assert(ProfileEnabled(Stage(event>=0x8000 ? event-0x8000 : event/2)));
  if(event==BEFORE_FIRST_SEND*2){assert(data[9+i*2]==kernel);++firstSend;}
  if(event==BEFORE_FIRST_RECV_POLL*2){assert(data[9+i*2]==kernel);++firstPoll;}
  if(event==0x8000+BEFORE_FIRST_SEND || event==0x8000+BEFORE_FIRST_RECV_POLL)
      assert((data[8+i*2]>>32)==1);
  kernelBegins+=event==0;kernelEnds+=event==1;kernelMetrics+=event==0x8000;
 }
 assert(kernelBegins==1 && kernelEnds==1 && kernelMetrics==1 && firstSend==1 && firstPoll==1);
 assert(std::all_of(words.begin(),words.begin()+55*kWordsPerCore,[](auto value){return value==0;}));
#else
 assert(AscendC::flushes==0);
 assert(std::all_of(words.begin(),words.end()-64,[](auto value){return value==0;}));
#endif
 assert(std::all_of(words.end()-64,words.end(),[](auto value){return value==0x12345678ULL;}));
 // Reuse enabled->null->enabled without reading stale/uninitialized arrays.
 auto oldClock=AscendC::clock;
 timer.Init(nullptr,1,1);timer.End(KERNEL,timer.Begin(KERNEL));timer.Count(TX_PACKETS,4);timer.Flush();
 assert(AscendC::clock==oldClock);
 std::fill(words.begin(),words.end()-64,0);
 timer.Init(reinterpret_cast<uint8_t*>(words.data()),1,1);
 timer.End(KERNEL,timer.Begin(KERNEL));timer.Flush();
#ifdef DEEPEP_LL_STAGE_TIMER
 auto* again=words.data()+55*kWordsPerCore;
 assert(again[6]==3 && again[7]==0 && (again[12]>>32)==1);
#endif
 DeviceTimer disabled;disabled.Init(nullptr,1,1);auto begin=disabled.Begin(KERNEL);disabled.End(KERNEL,begin);disabled.Flush();
}
''')
    executable=tmp_path/'check'
    subprocess.run([compiler,'-std=c++17','-Wall','-Wextra','-Werror',
                    *(['-DDEEPEP_LL_STAGE_TIMER=1'] if enabled else []),f'-DDEEPEP_LL_TIMER_PROFILE_ID={profile}',f'-I{tmp_path}',f'-I{ROOT}',str(source),'-o',str(executable)],check=True)
    subprocess.run([str(executable)],check=True,timeout=30)
