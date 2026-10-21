"""Actual private-q builder versus independent per-route prefix counts."""
import shutil
import subprocess
from pathlib import Path
import importlib.util
import pytest
ROOT=Path(__file__).resolve().parents[2]

def test_actual_private_q_builder(tmp_path):
    cc=shutil.which('g++')
    if cc is None:pytest.skip('C++ compiler required on A3')
    (tmp_path/'kernel_operator.h').write_text(r'''
#pragma once
#include <cstdint>
#include <cassert>
#include <vector>
#define __aicore__
namespace AscendC {
template<class T>struct LocalTensor{
 std::vector<T>* data;
 T GetValue(unsigned i)const{assert(i<data->size());return (*data)[i];}
 void SetValue(unsigned i,T value){assert(i<data->size());(*data)[i]=value;}
};
}
''')
    source=tmp_path/'q.cpp'
    source.write_text(r'''
#include <algorithm>
#include <numeric>
#include <random>
#include "csrc/kernels/dispatch/low_latency/common/ll_private_route_q.h"
using namespace DispatchDedup;using namespace AscendC;
int main(){
 std::mt19937 rng(123);
 for(unsigned b:{1U,7U,28U,29U,65U,256U})for(unsigned k:{1U,8U})
 for(unsigned e:{16U,384U,1024U})for(unsigned s:{1U,10U,28U,55U})for(unsigned kind=0;kind<3;++kind){
  std::vector<int32_t> ids(b*k),pool(e);std::iota(pool.begin(),pool.end(),0);
  for(unsigned t=0;t<b;++t){
   std::shuffle(pool.begin(),pool.end(),rng);
   for(unsigned j=0;j<k;++j) ids[t*k+j]=kind==0?pool[j]:kind==1?j:0;
  }
  // kind2 also checks exact legacy first-token behavior for duplicate IDs.
  std::vector<unsigned> ref(b*k);
  for(unsigned i=0;i<b*k;++i)ref[i]=i<k?0:std::count(ids.begin(),ids.begin()+i,ids[i]);
  unsigned covered=0;
  for(unsigned c=0;c<s;++c){
   unsigned n=b/s+(c<b%s),start=c*(b/s)+std::min(c,b%s),lo=start*k;
   if(!n)continue;
   std::vector<uint32_t> hist(e),table(n*k+8,0xdeadbeef),quant(100,0);
   LlBuildPrivateRouteQ({&ids},{&hist},{&table},lo,n*k,k);
   // Delayed FillTail / reordered DIRECT lookups must not change after unrelated quant work.
   std::fill(quant.begin(),quant.end(),0xffffffff);
   for(unsigned j=n*k;j>0;--j)assert(table[j-1]==ref[lo+j-1]);
   for(unsigned j=n*k;j<table.size();++j)assert(table[j]==0xdeadbeef);
   for(unsigned expert=0;expert<e;++expert)
    assert(hist[expert]==unsigned(std::count(ids.begin(),ids.begin()+lo+n*k,int(expert))));
   covered+=n*k;
  }
  assert(covered==b*k);
 }
}
''')
    exe=tmp_path/'q'
    subprocess.run([cc,'-std=c++17','-O2',f'-I{tmp_path}',f'-I{ROOT}',str(source),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True,timeout=60)

def test_sender_paths_and_quant_lifetime():
    s=(ROOT/'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h').read_text(encoding='utf-8')
    dedup=s.split('inline void SendWithDedup(',1)[1].split('inline void SendWithoutDedup(',1)[0]
    plain=s.split('inline void SendWithoutDedup(',1)[1].split('private:',1)[0]
    tail=s.split('inline void FillTail(',1)[1].split('inline void SendPacket(',1)[0]
    quant=s.split('inline void QuantMxfp8(',1)[1].split('inline void FillTail(',1)[0]
    local=s.split('inline void ProcessGroupLocal(',1)[1].split('inline void SendWithDedup(',1)[0]
    assert 'CalTokenSendExpertCnt(' not in dedup+tail
    assert plain.count('CalTokenSendExpertCnt(')==2
    assert 'routeQ_' not in plain+quant+local
    assert 'qDstBuf_.Get<float>()' in quant
    assert 'routeQ_=qWorkBuf_.Get<uint32_t>()' in s
    assert s.index('LL_SEG_SCOPE(BUILD_Q_TABLE)')<s.index('LL_SEG_SCOPE(SEND_TOKENS)')
    assert s.index('Duplicate<int32_t>(histogram')<s.index('LlBuildPrivateRouteQ(')

def test_new_stage_and_legacy_limits():
    spec=importlib.util.spec_from_file_location('q_timer',ROOT/'deep_ep/utils/ll_stage_timer.py')
    timer=importlib.util.module_from_spec(spec);spec.loader.exec_module(timer)
    assert timer.SCHEMA==15 and timer.STAGES[91]=='BUILD_Q_TABLE'
    assert 91 not in timer.SAMPLED_DETAILS and 91 not in timer.COUNTERS
    assert 8+2*timer.RECORDS==timer.SLOTS*timer.FIELDS
