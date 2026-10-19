"""Compile the real device recorder against a CPU memory/clock double."""
from pathlib import Path
import shutil
import subprocess
import pytest
from event_fixtures import timer
from tests.utils.cpp import cpp_functions

ROOT=Path(__file__).resolve().parents[2]


def test_event_recorder_cpp(tmp_path):
    compiler=shutil.which('g++') or shutil.which('clang++')
    if compiler is None:
        pytest.skip('C++ compiler required for the real recorder host model')
    (tmp_path/'kernel_operator.h').write_text(r'''
#pragma once
#include <cstdint>
#define __aicore__
#define __gm__
using GM_ADDR=uint8_t*;
namespace AscendC {
enum Pipe { PIPE_ALL };
inline unsigned barriers=0;
inline uint64_t ticks=0;
inline unsigned GetBlockIdx(){return 0;}
inline unsigned GetBlockNum(){return 16;}
inline uint64_t GetSystemCycle(){return ++ticks;}
template<Pipe> void PipeBarrier(){++barriers;}
template<class T> struct GlobalTensor {
 T* p=nullptr;
 void SetGlobalBuffer(T* x){p=x;}
 T& operator()(unsigned i){return p[i];}
};
}
using AscendC::PIPE_ALL;
''')
    sender = (ROOT/'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h').read_text(encoding='utf-8')
    fill_tail = cpp_functions(sender, 'FillTail')
    code=r'''
#include "kernels/dispatch/low_latency/common/ll_stage_timer_device.h"
#include <cassert>
#include <vector>
using namespace LlStageTimer;
template<class T> struct LocalTensor {
 T* p;
 LocalTensor operator[](unsigned i){return {p+i};}
 template<class U> LocalTensor<U> ReinterpretCast(){return {reinterpret_cast<U*>(p)};}
 void SetValue(unsigned i,T value){p[i]=value;}
 T GetValue(unsigned i){return p[i];}
};
enum class EpServerDedupTokenType {Direct,FirstHop};
struct TailHarness {
 DeviceTimer* timer_;
 struct {unsigned headerPayloadOffset=0,readIndexPayloadOffset=32;} data_;
 struct {unsigned rank=3;} argsValue;
 decltype(argsValue)* args_=&argsValue;
 bool enableDedup_=true;
 unsigned start_=0,k_=8,routeCalls=0;
 int32_t ids[32]={};float weights[32]={};
 LocalTensor<int32_t> routeIds_{ids};LocalTensor<float> routeWeights_{weights};
 unsigned CalTokenSendExpertCnt(unsigned expert,unsigned index){
  timer_->Start(SendRoute);++routeCalls;timer_->End(SendRoute);return index;
 }
 FILL_TAIL
};
int main(){
 static_assert(kWordsPerCore==EXPECTED_WORDS);
 static_assert(kWordsPerCore*8%512==0);
 for(unsigned mode=1;mode<=3;++mode){
  std::vector<uint64_t> words(kWordsPerCore+64,0);
  words[kWordsPerCore]=12345;
  DeviceTimer t;
  t.Init(reinterpret_cast<GM_ADDR>(words.data()),1,mode);
  t.Start(Kernel);
  t.Start(RecvMetadata);t.End(RecvMetadata);
  t.Start(RecvLoadWait);t.End(RecvLoadWait);
  t.End(Kernel);t.Finish();
  assert(words[ControlOffset(RecvMetadata)]==1);
  assert(words[EventOffset(RecvMetadata,0)+1]>words[EventOffset(RecvMetadata,0)]);
  assert(AscendC::barriers==(mode==1?0U:mode==2?6U:4U));
  assert(words[kEventErrorOffset]==0 && words[6]==1 && words[kWordsPerCore]==12345);
  AscendC::barriers=0;
 }
 // Execute the production FillTail, including the helper call that used to
 // enter SendRoute while SendTail was still active (device flags=6).
 for(unsigned mode=1;mode<=3;++mode){
  for(bool dedup:{false,true}) for(unsigned mask:{0U,3U,255U}){
   std::vector<uint64_t> memory(kWordsPerCore+64,0);
   DeviceTimer recorder;recorder.Init(reinterpret_cast<GM_ADDR>(memory.data()),1,mode);
   TailHarness h;h.timer_=&recorder;h.enableDedup_=dedup;
   alignas(8) uint8_t packet[512]={};
   recorder.Start(Kernel);recorder.Start(SendTail);
   h.FillTail({packet},1,0,mask);
   recorder.End(SendTail);recorder.End(Kernel);recorder.Finish();
   assert(memory[kEventErrorOffset]==0 && memory[6]==1);
   unsigned expected=dedup?(mask==255?7U:mask==3?1U:0U):0U;
   assert(h.routeCalls==expected);
   assert(memory[ControlOffset(SendRoute)]==expected);
   assert(memory[ControlOffset(SendTail)]==expected+1);
  }
  // Combine direct return, partial reduction and source output, as well as
  // dispatch relay, pause only for dependency waits. Resume the same task ID.
  for(Stage task:{CombineReturnTask,CombineReduceTask,CombineOutputTask,RecvForward}){
   std::vector<uint64_t> memory(kWordsPerCore+64,0);
   DeviceTimer recorder;recorder.Init(reinterpret_cast<GM_ADDR>(memory.data()),2,mode);
   recorder.Start(Kernel);recorder.Start(task);
   for(Stage wait:{CombineDataWait,SendRdmaBodyWait,SendRdmaFlagWait}){
    recorder.Start(wait);recorder.End(wait);
   }
   recorder.End(task);recorder.End(Kernel);recorder.Finish();
   assert(memory[kEventErrorOffset]==0 && memory[ControlOffset(task)]==4);
   for(unsigned i=1;i<4;++i)
    assert(memory[EventOffset(task,i)+2]==memory[EventOffset(task,0)+2]);
  }
 }
 std::vector<uint64_t> words(kWordsPerCore+64,0);
 words[kWordsPerCore]=987;
 DeviceTimer t;t.Init(reinterpret_cast<GM_ADDR>(words.data()),1,1);
 t.Start(Magic);t.End(Magic);t.Start(Magic);t.End(Magic);t.Finish();
 assert(words[kEventErrorOffset]&1);
 assert(words[ControlOffset(Magic)]==2 && words[kWordsPerCore]==987);
 t.Start(RecvMetadata);t.Start(RecvMetadata);t.End(RecvMetadata);
 assert(words[kEventErrorOffset]&2);
 t.End(RecvMetadata);assert(words[kEventErrorOffset]&4);
 // The reported flags=6 is an ordinary task nested inside another task.
 // Keep rejecting it rather than making Start/End permissive.
 std::fill(words.begin(),words.end(),0);
 t.Init(reinterpret_cast<GM_ADDR>(words.data()),1,3);
 t.Start(Kernel);t.Start(SendTail);t.Start(SendRoute);
 t.End(SendRoute);t.End(SendTail);t.End(Kernel);t.Finish();
 assert(words[kEventErrorOffset]==6);
 std::fill(words.begin(),words.end(),0);
 t.Init(reinterpret_cast<GM_ADDR>(words.data()),1,3);
 t.Start(Kernel);
 t.Start(CombineReduceTask);
 t.Start(CombineDataWait);t.End(CombineDataWait);
 t.End(CombineReduceTask);t.End(Kernel);t.Finish();
 assert(words[kEventErrorOffset]==0);
 assert(words[ControlOffset(CombineReduceTask)]==2);
 assert(words[EventOffset(CombineReduceTask,0)+2]==words[EventOffset(CombineReduceTask,1)+2]);
 assert(words[EventOffset(CombineReduceTask,0)+2]==words[EventOffset(CombineDataWait,0)+2]);
 assert(words[EventOffset(CombineReduceTask,0)+1]==words[EventOffset(CombineDataWait,0)]);
 assert(words[EventOffset(CombineDataWait,0)+1]==words[EventOffset(CombineReduceTask,1)]);
 t.Init(nullptr,1,3);t.Start(Kernel);t.End(Kernel);
}
'''.replace('EXPECTED_WORDS',str(timer.EVENT_WORDS)).replace('FILL_TAIL',fill_tail)
    (tmp_path/'events.cpp').write_text(code)
    binary=tmp_path/'events.exe'
    subprocess.run([compiler,'-std=c++17','-DDEEPEP_LL_STAGE_TIMER=1','-I'+str(tmp_path),
                    '-I'+str(ROOT/'csrc'),str(tmp_path/'events.cpp'),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True,timeout=30)
