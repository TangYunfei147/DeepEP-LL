"""Compile and execute recorder/Host ABI on CPU; not CANN or NPU validation."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest

ROOT=Path(__file__).resolve().parents[2]
COMMON=ROOT/'csrc/kernels/dispatch/low_latency/common'


def compiler():
    configured=shlex.split(os.environ.get('CXX',''))
    if configured:return configured
    found=shutil.which('g++') or shutil.which('clang++')
    if not found:pytest.skip('Local C++17 compiler required; no device toolchain is used')
    return [found]


def environment(path):
    result=dict(os.environ)
    for name,folder in (('ZIG_GLOBAL_CACHE_DIR',ROOT/'.tmp/reference-cxx-cache'),
                        ('ZIG_LOCAL_CACHE_DIR',path/'local-cache'),('TMP',path/'tmp'),('TEMP',path/'tmp'),('TMPDIR',path/'tmp')):
        folder.mkdir(parents=True,exist_ok=True);result[name]=str(folder)
    return result


def build_run(path,source,*,flags=(),extra=()):
    unit=path/'unit.cpp';unit.write_text(source,encoding='utf-8')
    executable=path/('unit.exe' if os.name=='nt' else 'unit')
    built=subprocess.run([*compiler(),'-std=c++17','-O2','-UNDEBUG',*flags,'-I'+str(path),'-iquote',str(ROOT),
        '-I'+str(COMMON),str(unit),*map(str,extra),'-o',str(executable)],capture_output=True,text=True,env=environment(path),timeout=180)
    assert built.returncode==0,built.stderr[-8000:]
    result=subprocess.run([str(executable)],capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr


def headers(path):
    (path/'kernel_operator.h').write_text(r'''
#pragma once
#include <cstdint>
#include <cstring>
#define __aicore__
#define __gm__
using GM_ADDR=uint8_t*;
namespace AscendC {
inline uint32_t core=0;
inline uint32_t GetBlockIdx(){return core;}
inline uint32_t GetBlockNum(){return 56;}
#ifdef DEEPEP_LL_REFERENCE_PROFILE
inline uint64_t ticks=(uint64_t(1)<<60),clocks=0,fences=0,events=0,copies=0;
inline uint64_t GetSystemCycle(){++clocks;return ticks+=17;}
enum class MemDsbT{DDR};
template<MemDsbT> void DataSyncBarrier(){++fences;}
enum class HardEvent{S_MTE3,MTE3_S};
template<HardEvent>void SetFlag(unsigned){++events;}
template<HardEvent>void WaitFlag(unsigned){++events;}
enum class CacheMode{CACHE_MODE_DISABLE};
template<class T>struct LocalTensor {
 T* p=nullptr; T GetValue(unsigned n)const{return p[n];} void SetValue(unsigned n,T v){p[n]=v;}
 LocalTensor operator[](unsigned n)const{return {p+n};}
 template<class U>LocalTensor<U> ReinterpretCast()const{return {reinterpret_cast<U*>(p)};}
};
template<class T>struct GlobalTensor {
 T* p=nullptr; void SetGlobalBuffer(uint8_t* q){p=reinterpret_cast<T*>(q);} void SetL2CacheHint(CacheMode){}
};
template<class T>void DataCopy(GlobalTensor<T> a,LocalTensor<T>b,unsigned n){++copies;std::memcpy(a.p,b.p,n*sizeof(T));}
#endif
}
''',encoding='utf-8')
    (path/'shmem.h').write_text('''#pragma once
#include <cstdint>
inline uint64_t flushes=0,length=0;
inline uint8_t* address=nullptr;
inline void dcci_cachelines(uint8_t* a,uint64_t n){++flushes;address=a;length=n;}
''',encoding='utf-8')


@pytest.mark.parametrize('enabled',[False,True])
def test_clock_store_validity_cache_scope_and_empty_poll_cost(tmp_path,enabled):
    headers(tmp_path)
    build_run(tmp_path,r'''
#include "ll_reference_profile_device.h"
#include <cassert>
int main(){
 using namespace LlReferenceProfile;
 static_assert(kSchema==7&&kWordsPerCore==640);
 alignas(512) uint64_t storage[58*kWordsPerCore]{};
 for(unsigned i=0;i<kWordsPerCore;++i)storage[i]=storage[57*kWordsPerCore+i]=0x7777;
 auto* base=storage+kWordsPerCore;
 for(unsigned core=0;core<56;++core){
  AscendC::core=core;DeviceProfile p;
  auto ptr=reinterpret_cast<uint8_t*>(base);
  const auto entry=DeviceProfile::EntryClock(ptr);p.Init(ptr);p.Value<0>(entry);p.Mark<1>();
  p.Mark<6>();p.Mark<8>();p.Mark<18>();p.Value<33>(0);
#ifdef DEEPEP_LL_REFERENCE_PROFILE
  const auto clocks=AscendC::clocks;
#endif
  for(unsigned n=0;n<10000;++n)p.PollBegin();
  p.BatchReady();p.BatchDone();p.ReceiveTotals(1);
  p.Mark<21>();p.Finish(0,16,2);
  auto* row=base+core*kWordsPerCore;
#ifdef DEEPEP_LL_REFERENCE_PROFILE
  assert(AscendC::clocks==clocks+5); // begin, ready, done, w21, w57
  assert(row[33]==17&&row[34]==17&&row[41]==1&&row[37]==1);
  assert(row[0]>(uint64_t(1)<<60)&&row[1]-row[0]==17);
  assert(row[55]==7&&row[62]==0&&row[63]==7&&row[64]==2);
  assert(row[44]==28&&row[45]==28&&row[46]==56&&row[47]==0);
  assert(row[51]==core&&row[53]==56&&row[50]==0&&row[52]==16&&row[54]==2);
  assert((row[58]&(uint64_t(1)<<33))!=0&&row[32]==0);
  assert(address==reinterpret_cast<uint8_t*>(row)&&length==1023);
#else
  static_assert(sizeof(DeviceProfile)==1);
  assert(!p.Enabled()&&entry==0&&p.Clock()==0);
  for(unsigned n=0;n<kWordsPerCore;++n)assert(row[n]==0);
#endif
 }
 for(unsigned i=0;i<kWordsPerCore;++i)assert(storage[i]==0x7777&&storage[57*kWordsPerCore+i]==0x7777);
 DeviceProfile disabled;disabled.Init(nullptr);
#ifdef DEEPEP_LL_REFERENCE_PROFILE
 const auto clocks=AscendC::clocks,flushed=flushes;
#endif
 disabled.Mark<0>();disabled.PollBegin();disabled.BatchReady();disabled.BatchDone();disabled.Finish(0,16,2);
 assert(!disabled.Enabled());
#ifdef DEEPEP_LL_REFERENCE_PROFILE
 assert(AscendC::clocks==clocks&&flushes==flushed);
#endif
}
''',flags=['-DDEEPEP_LL_REFERENCE_PROFILE=1'] if enabled else [])


def test_packet_log_uses_only_available_tail_and_exports_after_business(tmp_path):
    headers(tmp_path)
    build_run(tmp_path,r'''
#include "ll_reference_receiver_log.h"
#include <cassert>
int main(){
 using namespace LlReferenceProfile;
 alignas(512) uint64_t output[kWordsPerCore]{};
 uint32_t log[16]{},snapshot[2]{0,1},wire[32]{};
 DeviceProfile p;p.Init(reinterpret_cast<uint8_t*>(output));
 ReceiverLog trace;assert(trace.Bytes(&p,63,2)==32);
 trace.Init({log});trace.Ready({snapshot},3,2);
 wire[0]=1;wire[1]=0;wire[2]=3;wire[3]=1;
 trace.Processed(1,{reinterpret_cast<uint8_t*>(wire)},0,16,9);
 trace.Processed(0,{reinterpret_cast<uint8_t*>(wire)},0,16,8);
 assert(AscendC::copies==0&&AscendC::events==0);
 p.Mark<21>();trace.Export(&p,2,4,2,24,8);p.Finish(0,16,2);
 assert(output[96]==1&&output[97]==1&&output[98]==2&&output[99]==2&&output[100]==2&&output[107]==1);
 assert(output[102]>=output[21]&&output[103]>=output[102]&&output[57]>=output[103]);
 assert(output[128]==(uint64_t(1)<<32|1));assert((output[129]&0xffffffff)==2);
 assert(AscendC::copies==1&&AscendC::events==4);
 ReceiverLog none;assert(none.Bytes(&p,0,10)==0);
 ReceiverLog bounded;assert(bounded.Bytes(&p,100000,500)==128*32);
}
''',flags=['-DDEEPEP_LL_REFERENCE_PROFILE=1'])


@pytest.mark.parametrize('selector',range(5))
@pytest.mark.parametrize('enabled',[False,True])
def test_real_host_bridge_seven_modes_identity_and_launch_layout(tmp_path,selector,enabled):
    code=r'''
#include <cassert>
#include <cstddef>
#include <stdexcept>
#include "csrc/kernels/dispatch/low_latency/launch.hpp"
#include "csrc/kernels/combine/low_latency/launch.hpp"
unsigned observed=99;
#define STUB(name,n) extern "C" void dispatch_server_dedup_launch_reference7_##name(void*,const deepep::EpServerDedupKernelArgs*){observed=n;}
STUB(prequant_e4m3,0) STUB(fp16,1) STUB(fp16_e4m3,2) STUB(fp16_e5m2,3)
STUB(bf16,4) STUB(bf16_e4m3,5) STUB(bf16_e5m2,6)
int main(){
 static_assert(sizeof(deepep::EpServerDedupKernelArgs)==144);
 static_assert(sizeof(deepep::EpServerDedupCombineKernelArgs)==88);
 static_assert(offsetof(deepep::EpServerDedupKernelArgs,referenceProfile)==128);
 static_assert(offsetof(deepep::EpServerDedupKernelArgs,coreNum)==136);
 static_assert(offsetof(deepep::EpServerDedupCombineKernelArgs,coreNum)==80);
 assert(deepep_ll_instrumentation_source_id()==7&&deepep_ll_reference_profile_schema()==7);
 assert(deepep_ll_dispatch_launch_bytes()==144&&deepep_ll_combine_launch_bytes()==88);
 assert(deepep_ll_dispatch_implementation()==SELECTOR);
 assert(deepep_ll_reference_profile_enabled()==ENABLED);
 const unsigned protocols[]={19028,21028,22028,23028,24028};
 assert(deepep_ll_core_protocol_version()==protocols[SELECTOR]);
 deepep::EpServerDedupKernelArgs args{};
 for(unsigned i=0;i<7;++i){dispatch_server_dedup_kernel_do_reference7(nullptr,&args,i);assert(observed==i);}
 bool rejected=false;try{dispatch_server_dedup_kernel_do_reference7(nullptr,&args,7);}catch(const std::invalid_argument&){rejected=true;}assert(rejected);
}
'''
    build_run(tmp_path,code,flags=['-DCATLASS_ARCH=3510',f'-DDEEPEP_LL_DISPATCH_CB59={selector}',
        f'-DSELECTOR={selector}',f'-DENABLED={int(enabled)}',*(['-DDEEPEP_LL_REFERENCE_PROFILE=1'] if enabled else [])],
        extra=[ROOT/'csrc/kernels/dispatch/low_latency.cpp'])
