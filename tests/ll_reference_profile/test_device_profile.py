"""Actual reference recorder and Host selector compiled with CPU-only stubs."""
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT=Path(__file__).resolve().parents[2]


def compiler():
    result=shutil.which('g++') or shutil.which('clang++')
    if result is None:pytest.skip('CPU C++ compiler unavailable')
    return result


def run_reference_recorder_mock(tmp_path,enabled):
    (tmp_path/'kernel_operator.h').write_text(r'''
#pragma once
#include <cstdint>
#define __aicore__
#define __gm__
using GM_ADDR=uint8_t*;
namespace AscendC {
inline uint64_t clock=(1ULL<<63),reads=0,barriers=0;
inline uint32_t core=0;
inline uint64_t GetSystemCycle(){++reads;return clock++;}
inline uint32_t GetBlockIdx(){return core;}
inline uint32_t GetBlockNum(){return 56;}
enum class MemDsbT { DDR };
template<MemDsbT> void DataSyncBarrier(){++barriers;}
}
''',encoding='utf-8')
    (tmp_path/'shmem.h').write_text(r'''
#pragma once
#include <cstdint>
inline uint64_t cache_flushes=0,last_flush_bytes=0;
inline void dcci_cachelines(uint8_t*,uint64_t bytes){++cache_flushes;last_flush_bytes=bytes;}
''',encoding='utf-8')
    source=tmp_path/'recorder.cpp'
    source.write_text(r'''
#include <cassert>
#include <vector>
#include <algorithm>
#include "csrc/kernels/dispatch/low_latency/common/ll_reference_profile_device.h"
using namespace LlReferenceProfile;
int main(){
 static_assert(kSchema==6 && kSlots==80 && kFields==8 && kWordsPerCore*8==5120);
 static_assert(kPacketTraceCapacity==128 && kPacketTraceFields*4==32);
 std::vector<uint64_t> words(56*kWordsPerCore+64,0);
 std::fill(words.end()-64,words.end(),0x12345678ULL);
 DeviceProfile p;
 auto start=AscendC::reads;
 p.Init(nullptr);p.Mark<0>();p.Value<37>(0);p.Finish(7,16,256);
 assert(AscendC::reads==start && !p.Enabled());
 for(uint32_t core: {0U,29U,30U,55U}) {
  AscendC::core=core;
  auto entry=DeviceProfile::EntryClock(reinterpret_cast<uint8_t*>(words.data()));
  p.Init(reinterpret_cast<uint8_t*>(words.data()));
  p.Value<0>(entry);p.Mark<1>();p.Value<37>(0);p.Mark<20>();p.Mark<21>();
#ifdef DEEPEP_LL_REFERENCE_PROFILE
  assert(p.Enabled());
  auto* row=words.data()+core*kWordsPerCore;
  assert(row[0]==entry && row[37]==0 && row[61]==0);
  p.SendLanes(6);p.SendBegin(false);p.GroupSubmitted();
  assert(row[64]==0); // Sender detail stays in private Scalar state until Finish.
  p.SendBegin(true);p.GroupSubmitted();p.SendBegin(true);p.GroupSubmitted();
  p.Detail(96,1);p.Detail(97,1);p.Detail(98,2);p.Detail(107,1);
  auto* packet=reinterpret_cast<uint32_t*>(p.PacketLogAddress());
  for(unsigned f=0;f<8;++f)packet[f]=f+1;
  p.Finish(7,16,256);
  assert(row[22]==30 && row[23]==26 && row[50]==7 && row[51]==core && row[52]==16 && row[53]==56);
  assert(row[54]==256 && row[55]==6 && row[56]==1 && row[59]==kMagic && row[60]==1 && row[61]==1);
  assert(row[62]==0 && row[63]==(core<30?0:1));
  assert(row[58]&1 && row[58]&(1ULL<<37) && row[58]&(1ULL<<57));
  assert(!(row[58]&(1ULL<<22)) && !(row[58]&(1ULL<<23)));
  assert(row[64]==3 && row[65]==0 && row[66]==2 && row[88]==3 && row[89]==6);
  assert(row[69]<=row[70] && row[86]==2 && row[87]==3);
  assert(row[67]==0 && row[68]==0);
  for(unsigned i=71;i<=85;++i)assert(row[i]==0);
  for(unsigned i=90;i<=95;++i)assert(row[i]==0);
  assert(row[96]==1 && row[107]==1 && packet[7]==8);
  assert(row[57]>=row[21] && cache_flushes==AscendC::barriers && last_flush_bytes==1023);
#else
  (void)entry;p.Finish(7,16,256);
  assert(!p.Enabled() && AscendC::reads==start && AscendC::barriers==0);
  assert(std::all_of(words.begin(),words.end()-64,[](auto x){return x==0;}));
#endif
 }
 assert(std::all_of(words.end()-64,words.end(),[](auto x){return x==0x12345678ULL;}));
}
''',encoding='utf-8')
    executable=tmp_path/'recorder'
    subprocess.run([compiler(),'-std=c++17','-Wall','-Wextra','-Werror',
        *(['-DDEEPEP_LL_REFERENCE_PROFILE=1'] if enabled else []),
        '-DDEEPEP_LL_SEND_CORES=30','-DDEEPEP_LL_RECV_CORES=26','-I',str(tmp_path),'-I',str(ROOT),
        str(source),'-o',str(executable)],check=True)
    subprocess.run([str(executable)],check=True,timeout=30)


def run_reference_host_selector_mock(tmp_path,enabled):
    source=tmp_path/'selector.cpp'
    source.write_text(r'''
#include <cassert>
#include <stdexcept>
#include <initializer_list>
#include "csrc/kernels/dispatch/low_latency/common/ll_dispatch_variant.h"
#include "csrc/kernels/dispatch/low_latency/launch.hpp"
#include "csrc/kernels/combine/low_latency/launch.hpp"
static unsigned selected=99,calls=0;
#define STUB(name,n) extern "C" void dispatch_server_dedup_launch_ref6_##name(void*,const deepep::EpServerDedupKernelArgs*){selected=n;++calls;}
STUB(prequant_e4m3,0) STUB(fp16,1) STUB(fp16_e4m3,2) STUB(fp16_e5m2,3)
STUB(bf16,4) STUB(bf16_e4m3,5) STUB(bf16_e5m2,6)
int main(){
 static_assert(sizeof(deepep::EpServerDedupKernelArgs)==144);
 static_assert(sizeof(deepep::EpServerDedupCombineKernelArgs)==88);
 static_assert(offsetof(deepep::EpServerDedupKernelArgs,referenceProfile)==128);
 static_assert(offsetof(deepep::EpServerDedupKernelArgs,coreNum)==136);
 static_assert(offsetof(deepep::EpServerDedupCombineKernelArgs,coreNum)==80);
 deepep::EpServerDedupKernelArgs args{};
 for(unsigned i=0;i<7;++i){calls=0;dispatch_server_dedup_kernel_do_ref6(nullptr,&args,i);assert(calls==1&&selected==i);}
 for(unsigned i:{7U,99U,0xffffffffU}){calls=0;bool bad=false;try{dispatch_server_dedup_kernel_do_ref6(nullptr,&args,i);}catch(const std::invalid_argument&){bad=true;}assert(bad&&!calls);}
 bool bad=false;try{dispatch_server_dedup_kernel_do_ref6(nullptr,nullptr,0);}catch(const std::invalid_argument&){bad=true;}assert(bad);
 assert(deepep_ll_reference_profile_schema()==6 && deepep_ll_core_protocol_version()==19030);
#ifdef DEEPEP_LL_REFERENCE_PROFILE
 assert(deepep_ll_reference_profile_enabled()==1);
#else
 assert(deepep_ll_reference_profile_enabled()==0);
#endif
}
''',encoding='utf-8')
    executable=tmp_path/'selector'
    subprocess.run([compiler(),'-std=c++17','-DCATLASS_ARCH=3510',
        *(['-DDEEPEP_LL_REFERENCE_PROFILE=1'] if enabled else []),
        '-DDEEPEP_LL_SEND_CORES=30','-DDEEPEP_LL_RECV_CORES=26','-I',str(ROOT),str(source),
        str(ROOT/'csrc/kernels/dispatch/low_latency.cpp'),'-o',str(executable)],check=True)
    subprocess.run([str(executable)],check=True,timeout=30)


@pytest.mark.parametrize('enabled',[False,True])
def test_actual_recorder(tmp_path,enabled):run_reference_recorder_mock(tmp_path,enabled)


@pytest.mark.parametrize('enabled',[False,True])
def test_actual_selector_and_abi(tmp_path,enabled):run_reference_host_selector_mock(tmp_path,enabled)


def method(source,name):
    start=source.index('__aicore__ inline ',source.index(name)-40)
    opening=source.index('{',start);end=opening+1;depth=1
    while depth:
        depth+=(source[end]=='{')-(source[end]=='}');end+=1
    return source[start:end]


def test_actual_receive_log_helpers_and_ready_batch_clock_budget(tmp_path):
    text=(ROOT/'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_recv.h').read_text(encoding='utf-8')
    methods='\n'.join(method(text,name) for name in ('void StartPoll()','void ReadyPoll()',
        'void FinishBatch(','void RecordReady(','void RecordProcessed(','void ExportPacketTrace('))
    source=tmp_path/'receive_detail.cpp'
    source.write_text(r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>
#define __aicore__
using GM_ADDR=uint8_t*;
enum class CacheMode { CACHE_MODE_DISABLE };
enum class HardEvent { S_MTE3,MTE3_S };
unsigned syncs=0,copies=0;bool businessDone=false;
template<HardEvent> void ServerDedupReceiveSync(){assert(businessDone);++syncs;}
template<class T> struct LocalTensor {
 T* ptr;
 T GetValue(uint32_t i)const{return ptr[i];}
 void SetValue(uint32_t i,T value){ptr[i]=value;}
 LocalTensor operator[](uint32_t i)const{return {ptr+i};}
 template<class U> LocalTensor<U> ReinterpretCast()const{return {reinterpret_cast<U*>(ptr)};}
};
struct TBuf {alignas(32) uint8_t bytes[4096]{};template<class T>LocalTensor<T> Get(){return {reinterpret_cast<T*>(bytes)};}};
template<class T> struct GlobalTensor {
 T* ptr=nullptr;
 void SetGlobalBuffer(GM_ADDR p){ptr=reinterpret_cast<T*>(p);}
 void SetL2CacheHint(CacheMode){}
};
void DataCopy(GlobalTensor<uint8_t> dst,LocalTensor<uint8_t> src,uint32_t n){assert(businessDone);++copies;std::memcpy(dst.ptr,src.ptr,n);}
enum class EpServerDedupTokenType:uint32_t { Direct=1,FirstHop=2,Terminal=3 };
namespace LlReferenceProfile {
struct DeviceProfile {
 std::array<uint64_t,640> words{};uint64_t clock=100,reads=0;
 bool Enabled()const{return true;}
 uint64_t Clock(){++reads;return clock++;}
 template<unsigned N>void Value(uint64_t v){words[N]=v;}
 void Detail(uint32_t w,uint64_t v){words[w]=v;}
 GM_ADDR PacketLogAddress(){return reinterpret_cast<uint8_t*>(words.data()+128);}
};
}
struct Tasks {
 uint32_t count=0,begin=13;
 uint32_t Count()const{return count;}
 uint32_t Begin()const{return begin;}
 uint32_t Position(uint32_t j)const{return 4096+j;}
};
struct Receiver {
 LlReferenceProfile::DeviceProfile* profile_;
 bool traceEnabled_=true,pollStarted_=false;
 uint32_t traceCapacity_=0,traceReady_=0,traceProcessed_=0,batches_=0;
 uint64_t pollStart_=0,lastReady_=0,pollTicks_=0,consumeTicks_=0;
 uint32_t lanes_=6,bs_=256,localExperts_=24;
 TBuf packetTraceBuf_;Tasks tasks_;
 struct {uint32_t relayReadIndexWireOffset=64,readIndexPayloadOffset=64;}data_;
''' + methods + r'''
};
int main(){
 for(unsigned capacity:{0U,1U,128U}) {
  LlReferenceProfile::DeviceProfile profile;
  Receiver recv{};recv.profile_=&profile;recv.traceCapacity_=capacity;recv.tasks_.count=257;
  alignas(32) uint32_t ids[16]{};alignas(32) uint32_t packet[32]{};
  std::vector<uint32_t> found(257),batch(257),processed(257);
  unsigned seen=0,done=0;
  for(unsigned begin=0;begin<257;begin+=16) {
   unsigned count=std::min(16U,257-begin);
   auto reads=profile.reads;
   recv.StartPoll();for(unsigned retry=0;retry<100;++retry)recv.StartPoll();
   assert(profile.reads==reads+1); // Empty scans never add clocks.
   profile.clock+=10;recv.ReadyPoll();assert(profile.reads==reads+2);
   for(unsigned i=0;i<count;++i){ids[i]=begin+i;found[begin+i]=++seen;batch[begin+i]=recv.batches_;}
   recv.RecordReady({ids},(1U<<count)-1,count);
   // Process order differs from readiness order and j remains stable.
   for(unsigned i=count;i>0;--i) {
    unsigned j=begin+i-1;packet[0]=j%16;packet[1]=j;packet[2]=j%8;
    packet[3]=j%2?2U:1U;packet[16]=j%2?3U:0U;
    recv.RecordProcessed(j,{reinterpret_cast<uint8_t*>(packet)},0,true);
    processed[j]=++done;
    packet[3]=3U;packet[16]=0; // Later wire rewrite cannot change recorded identity.
   }
   profile.clock+=20;recv.FinishBatch(recv.lastReady_);
   assert(profile.reads==reads+3);
  }
  assert(recv.traceReady_==257 && recv.traceProcessed_==257 && recv.batches_==17);
  assert(recv.pollTicks_==17*11 && recv.consumeTicks_==17*21);
  auto before=profile.reads;businessDone=true;unsigned oldCopies=copies,oldSyncs=syncs;
  recv.ExportPacketTrace(&profile);
  assert(profile.reads==before+2 && copies-oldCopies==(capacity?1U:0U));
  assert(syncs-oldSyncs==(capacity?2U:0U));
  assert(profile.words[96]==capacity && profile.words[97]==capacity && profile.words[98]==257);
  assert(profile.words[99]==257 && profile.words[100]==257 && profile.words[107]==1 && profile.words[108]==17);
  assert(profile.words[104]==13 && profile.words[105]==256 && profile.words[106]==24);
  auto* rows=reinterpret_cast<uint32_t*>(profile.words.data()+128);
  for(unsigned j=0;j<capacity;++j){auto* r=rows+j*8;
   assert(r[0]==found[j] && r[1]==batch[j] && r[2]==processed[j]);
   assert(r[3]==j%16 && r[4]==j && r[5]==j%8);
   assert(r[6]==(j%2?(2U|(3U<<8)):1U) && r[7]==4096+j);
  }
  for(unsigned i=128+capacity*4;i<640;++i)assert(profile.words[i]==0);
  businessDone=false;
 }
 // A truly empty receiver exports complete shape metadata but no records/DMA.
 LlReferenceProfile::DeviceProfile profile;Receiver empty{};empty.profile_=&profile;empty.lanes_=0;
 businessDone=true;auto old=copies;empty.ExportPacketTrace(&profile);
 assert(copies==old && profile.words[98]==0 && profile.words[107]==0 && profile.words[108]==0);
}
''',encoding='utf-8')
    executable=tmp_path/'receive_detail'
    subprocess.run([compiler(),'-std=c++17','-Wall','-Wextra','-Werror',str(source),'-o',str(executable)],check=True)
    subprocess.run([str(executable)],check=True,timeout=30)
