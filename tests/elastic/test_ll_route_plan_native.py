"""Execute the actual planner in a threaded CPU mock when g++ is available.

Validates phases/strides/packing; does not validate CANN or physical MTE ordering.
"""
from pathlib import Path
import shutil
import subprocess
import pytest
ROOT=Path(__file__).resolve().parents[2]

@pytest.mark.parametrize('group',[4,16])
def test_actual_planner_with_56_cpu_workers(tmp_path,group):
    compiler=shutil.which('g++')
    if not compiler:pytest.skip('g++ required for actual shared-planner threaded mock')
    (tmp_path/'kernel_operator.h').write_text(MOCK)
    source=tmp_path/'plan.cpp';source.write_text(MAIN)
    exe=tmp_path/'plan'
    subprocess.run([compiler,'-std=c++17','-pthread','-O2',f'-DDEEPEP_LL_RANKS_PER_GROUP={group}',
                    '-I',str(tmp_path),'-I',str(ROOT),str(source),'-o',str(exe)],check=True,capture_output=True)
    subprocess.run([str(exe)],check=True,capture_output=True,timeout=60)

MOCK=r'''
#pragma once
#include <vector>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <cstdint>
#include <algorithm>
#define __aicore__
#define __gm__
using GM_ADDR=uint8_t*;
namespace AscendC {
inline thread_local unsigned core=0;
inline unsigned GetBlockIdx(){return core;}
enum class HardEvent{MTE2_S,V_S,S_MTE3}; using TEventID=unsigned;
template<HardEvent> void SetFlag(unsigned){}
template<HardEvent> void WaitFlag(unsigned){}
constexpr int PIPE_ALL=0;
template<int> void PipeBarrier(){}
class Barrier {
 std::mutex mutex;std::condition_variable cv;unsigned count=0,generation=0;
 public:void wait(){std::unique_lock<std::mutex> lock(mutex);auto old=generation;
  if(++count==56){count=0;++generation;cv.notify_all();}else cv.wait(lock,[&]{return generation!=old;});}
};
inline Barrier barrier;
template<bool> void SyncAll(){barrier.wait();}
template<class T> struct LocalTensor {
 T* p=nullptr;LocalTensor operator[](unsigned n)const{return {p+n};}
 T GetValue(unsigned n)const{return p[n];}void SetValue(unsigned n,T v){p[n]=v;}
};
template<class T> struct GlobalTensor {
 T* p=nullptr;void SetGlobalBuffer(T* q){p=q;}
 GlobalTensor operator[](unsigned n)const{return {p+n};}
};
struct TBufBase {std::shared_ptr<std::vector<uint32_t>> data;
 template<class T>LocalTensor<T> Get(){return {reinterpret_cast<T*>(data->data())};}};
template<class...>struct TBuf:TBufBase{};
struct TPipe {
 void Reset(){} template<class T>void InitBuffer(T& b,unsigned bytes){b.data=std::make_shared<std::vector<uint32_t>>((bytes+3)/4);}
 template<HardEvent>unsigned FetchEventID(){return 0;}
};
struct DataCopyParams{uint16_t blockCount,blockLen,srcStride,dstStride;};
struct DataCopyExtParams{uint16_t blockCount;uint32_t blockLen,srcStride,dstStride;uint8_t reserved;};
template<class T>struct DataCopyPadExtParams{bool pad;uint8_t left,right;T value;};
template<class T>void Duplicate(LocalTensor<T> d,T v,unsigned n){std::fill(d.p,d.p+n,v);}
template<class T>void DataCopy(GlobalTensor<T> d,LocalTensor<T> s,unsigned n){std::copy(s.p,s.p+n,d.p);}
template<class T>void DataCopy(LocalTensor<T> d,GlobalTensor<T> s,unsigned n){std::copy(s.p,s.p+n,d.p);}
template<class D,class S>void copyStrided(D d,S s,DataCopyParams p){
 const unsigned block=32/sizeof(*d.p),len=p.blockLen*block;
 for(unsigned b=0;b<p.blockCount;++b)std::copy(s.p+b*(len+p.srcStride*block),s.p+b*(len+p.srcStride*block)+len,d.p+b*(len+p.dstStride*block));
}
template<class T>void DataCopy(GlobalTensor<T> d,LocalTensor<T> s,DataCopyParams p){copyStrided(d,s,p);}
template<class T>void DataCopy(LocalTensor<T> d,GlobalTensor<T> s,DataCopyParams p){copyStrided(d,s,p);}
template<class T>void DataCopyPad(LocalTensor<T>d,GlobalTensor<T>s,DataCopyExtParams p,DataCopyPadExtParams<T>){std::copy(s.p,s.p+p.blockLen/sizeof(T),d.p);}
template<class T>void DataCopyPad(GlobalTensor<T>d,LocalTensor<T>s,DataCopyExtParams p){std::copy(s.p,s.p+p.blockLen/sizeof(T),d.p);}
}
'''

MAIN=r'''
#include <cassert>
#include <thread>
#include <array>
#include "csrc/kernels/dispatch/low_latency/device/dispatch_shared_route_plan.h"
using namespace DispatchDedup;
struct Ids {const uint32_t* p;uint32_t GetValue(unsigned i)const{return p[i];}};
int main(){
 const std::vector<std::array<unsigned,4>> shapes={{1,1,16,16},{3,8,384,16},{57,32,1024,64},{256,8,384,16}};
 for(auto shape:shapes){
  auto b=shape[0],k=shape[1],e=shape[2],r=shape[3];auto layout=LlRoutePlanMemory(b*k,e);
  std::vector<uint32_t> memory(layout.bytes/4+128,0xdeadbeefU),ids(b*k),counts(e),running(e);
  for(unsigned i=0;i<b*k;++i){ids[i]=(i*37+i/7)%e;++counts[ids[i]];}
  std::vector<std::thread> threads;
  for(unsigned c=0;c<56;++c)threads.emplace_back([&,c]{AscendC::core=c;AscendC::TPipe pipe;
   BuildSharedRoutePlan(reinterpret_cast<uint8_t*>(memory.data()),reinterpret_cast<uint8_t*>(ids.data()),b,k,e,r,&pipe);});
  for(auto& t:threads)t.join();
  for(unsigned x=0;x<e;++x){assert(memory[layout.totalsOffset/4+x*8]==0x3f800000U);assert(memory[layout.totalsOffset/4+x*8+1]==counts[x]);}
  const unsigned groupExperts=e/r*kLlRanksPerGroup;
  for(unsigned i=0;i<b*k;++i){
   assert(memory[layout.idsOffset/4+i]==ids[i]);auto packed=memory[layout.qOffset/4+i];
   assert((packed&0xffffU)==(i<k?0:running[ids[i]]));++running[ids[i]];
   unsigned mask=0,t=i/k;
   for(unsigned j=0;j<k;++j)if(ids[t*k+j]/groupExperts==ids[i]/groupExperts)mask|=1U<<j;
   assert(memory[layout.masksOffset/4+i]==mask);
   auto anchor=LlPlanFirst(mask),rest=mask&~(1U<<anchor),event=rest?LlPlanFirst(rest):anchor;
   assert((packed>>24)==event);
   assert(((packed>>16)&255U)==LlChooseLeader(Ids{ids.data()+t*k},k,mask,e/r));
  }
  for(unsigned i=layout.bytes/4;i<memory.size();++i)assert(memory[i]==0xdeadbeefU);
 }
}
'''
