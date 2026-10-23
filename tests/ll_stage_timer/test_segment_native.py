"""Compile the actual sparse recorder with a minimal Ascend API mock when C++ is available."""
from pathlib import Path
import shutil
import subprocess
import pytest
ROOT=Path(__file__).resolve().parents[2]

@pytest.mark.parametrize('enabled',[False,True])
def test_actual_recorder_is_sparse_bounded_and_delays_gm_export(tmp_path,enabled):
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
 static_assert(kSchema==15 && kWordsPerCore*8==512 && kPointCount==14 && kExportWords==24);
 std::vector<uint64_t> words(56*kWordsPerCore+64,0);
 std::fill(words.end()-64,words.end(),0x12345678ULL);
 AscendC::core=55;
 DeviceTimer timer;timer.Init(reinterpret_cast<uint8_t*>(words.data()),1,1);
 auto* timer_=&timer;
 // Removed probes must not evaluate arguments or read the clock at all.
 unsigned sideEffects=0; auto clock=AscendC::clock;
 LL_SEG_SCOPE(INIT); LL_SEG_BEGIN(old,KERNEL); LL_SEG_END(old,KERNEL);
 LL_SEG_COUNT(TX_PACKETS,++sideEffects); LL_SEG_RELAY_COMPLETION(++sideEffects);
 LL_SEG_FIRST(BEFORE_FIRST_SEND);
 assert(sideEffects==0 && AscendC::clock==clock);
 timer.MarkAt<ENTRY>(clock);
 timer.Mark<INIT_DONE>();
 timer.MarkIfSeen<FIRST_SEND,SEND_DONE>();
 LL_SPARSE_MARK(FIRST_SEND);
 auto afterFirst=AscendC::clock;
 for(unsigned i=0;i<10000;++i){LL_SPARSE_MARK(FIRST_SEND);}
 assert(AscendC::clock==afterFirst);
 LL_SPARSE_AFTER(FIRST_SEND,SEND_DONE);
 timer.Mark<WORK_DONE>();
 assert(std::all_of(words.begin(),words.end()-64,[](auto value){return value==0;}));
 assert(AscendC::flushes==0);
 timer.Flush();
#ifdef DEEPEP_LL_STAGE_TIMER
 assert(AscendC::clock==clock+4);
 assert(AscendC::flushes==3);
 auto* data=words.data()+55*kWordsPerCore;
 assert(((data[5]>>24)&255)==6);
 assert(data[0]==kMagic && data[1]==15 && data[7]==0);
 assert(data[6]==((1ULL<<ENTRY)|(1ULL<<INIT_DONE)|(1ULL<<FIRST_SEND)|(1ULL<<SEND_DONE)|(1ULL<<WORK_DONE)));
 assert(data[8+ENTRY]<=data[8+INIT_DONE] && data[8+INIT_DONE]<data[8+FIRST_SEND]);
 assert(data[8+FIRST_SEND]<data[8+WORK_DONE]);
 assert(data[8+RECV_DONE]==0 && data[22]==0 && data[23]==0);
 assert(std::all_of(words.begin(),words.begin()+55*kWordsPerCore,[](auto value){return value==0;}));
 assert(std::all_of(data+24,data+kWordsPerCore,[](auto value){return value==0;}));
#else
 (void)timer_;
 assert(AscendC::flushes==0 && AscendC::clock==clock);
 assert(std::all_of(words.begin(),words.end()-64,[](auto value){return value==0;}));
#endif
 assert(std::all_of(words.end()-64,words.end(),[](auto value){return value==0x12345678ULL;}));
 // Reuse enabled -> null -> enabled without stale timestamps or a clear loop.
 auto oldClock=AscendC::clock;
 timer.Init(nullptr,1,1);timer.Mark<ENTRY>();timer.Mark<WORK_DONE>();timer.Flush();
 assert(AscendC::clock==oldClock);
 std::fill(words.begin(),words.end()-64,0);
 timer.Init(reinterpret_cast<uint8_t*>(words.data()),2,1);
 timer.Mark<ENTRY>();timer.Mark<INIT_DONE>();timer.Mark<WORK_DONE>();timer.Flush();
#ifdef DEEPEP_LL_STAGE_TIMER
 auto* again=words.data()+55*kWordsPerCore;
 assert(again[2]==2 && again[6]==((1ULL<<ENTRY)|(1ULL<<INIT_DONE)|(1ULL<<WORK_DONE)));
 assert(again[8+FIRST_SEND]==0);
 bool rejected=false;
 try{timer.Init(reinterpret_cast<uint8_t*>(words.data()),1,3);}catch(const std::runtime_error&){rejected=true;}
 assert(rejected);
#endif
}
''')
    executable=tmp_path/'check'
    subprocess.run([compiler,'-std=c++17','-Wall','-Wextra','-Werror',
                    *(['-DDEEPEP_LL_STAGE_TIMER=1'] if enabled else []),'-DDEEPEP_LL_TIMER_PROFILE_ID=6',f'-I{tmp_path}',f'-I{ROOT}',str(source),'-o',str(executable)],check=True)
    subprocess.run([str(executable)],check=True,timeout=30)
