"""Execute the production post-dedup ownership planner on CPU."""
import os
from pathlib import Path
import subprocess
import pytest
from tests.ll_rdma.test_combine_overhead import block, run_cpp

ROOT = Path(__file__).resolve().parents[2]


def test_post_dedup_packets_rotate_by_actual_peer(tmp_path):
    zig = os.environ.get('DEEPEP_TEST_ZIG')
    if not zig:
        pytest.skip('Set DEEPEP_TEST_ZIG for CPU C++ validation')
    exe = tmp_path / 'packet-owners.exe'
    subprocess.run([zig, 'c++', '-std=c++17', '-O2', '-UNDEBUG', '-I'+str(ROOT/'csrc'),
        str(Path(__file__).with_name('packet_owners_test.cpp')), '-o', str(exe)], check=True, timeout=180)
    result = subprocess.run([str(exe)], check=True, capture_output=True, text=True, timeout=60)
    assert 'unique physical-core QP owners' in result.stdout


def test_remote_owner_preserves_original_q_and_weight_tail(tmp_path):
    source = (ROOT/'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h').read_text(encoding='utf-8')
    methods = '\n'.join(block(source, name) for name in (
        '__aicore__ inline float RouteWeight(', '__aicore__ inline uint32_t CalTokenSendExpertCnt(',
        '__aicore__ inline void FillTail('))
    run_cpp(tmp_path, r'''
#include <cassert>
#include <vector>
#include <cstring>
#include "kernels/dispatch/low_latency/common/ll_dispatch_packet_plan.h"
#include "kernels/dispatch/low_latency/common/ll_sender_input_reuse.h"
#include "kernels/dispatch/low_latency/common/ep_memory_server_dedup_protocol.h"
#define __aicore__
using namespace DispatchDedup;
template<class T> struct LocalTensor {
 T* p;size_t n;
 T GetValue(uint32_t i) const {assert(i<n);return p[i];}
 void SetValue(uint32_t i,T v){assert(i<n);p[i]=v;}
 LocalTensor operator[](uint32_t i){assert(i<=n);return {p+i,n-i};}
 template<class U> LocalTensor<U> ReinterpretCast(){return {reinterpret_cast<U*>(p),n*sizeof(T)/sizeof(U)};}
};
struct Buffer {uint32_t* p;template<class T> LocalTensor<T> Get(){return {reinterpret_cast<T*>(p),160};}};
struct Args {uint32_t rank=7;};
template<bool Prequantized> struct Test {
 Args a;Args* args_=&a;uint32_t start_=3,k_=8,cachedTailEntries_=0;
 bool remoteIssue_=true,enableDedup_=true;
 struct Layout {uint32_t headerPayloadOffset=64,readIndexPayloadOffset=96;} data_;
 LocalTensor<float> weights_,routeWeights_;
 LocalTensor<int32_t> routeIds_;
 Buffer qPlanBuf_;
''' + methods + r'''
};
template<bool P> void Check(){
 std::vector<float> weights(160);std::vector<int32_t> ids(160);std::vector<uint32_t> q(160),raw(128);
 for(unsigned i=0;i<160;++i){weights[i]=float(i+1)/8;ids[i]=i*17%384;q[i]=0x80070000U|(i/8+7);}
 Test<P> t;t.weights_={weights.data(),160};t.routeWeights_={weights.data()+24,8};
 t.routeIds_={ids.data(),160};t.qPlanBuf_={q.data()};
 LocalTensor<uint8_t> payload{reinterpret_cast<uint8_t*>(raw.data()),raw.size()*4};
 for(bool remote:{true,false})for(unsigned token:{0U,3U,19U})for(uint32_t mask:{255U,60U,0U,8U}){
  if(!remote && token!=3)continue;
  t.remoteIssue_=remote;unsigned first=mask?0:3;while(mask && !(mask&(1U<<first)))++first;
  t.routeWeights_=remote?LocalTensor<float>{nullptr,0}:LocalTensor<float>{weights.data()+24,8};
  t.FillTail(payload,token,first,mask);
  assert(raw[16]==7 && raw[17]==token && raw[18]==first && raw[19]==(mask?2U:1U));
  auto* index=raw.data()+24;float weight;std::memcpy(&weight,index+1,4);
  assert(weight==weights[token*8+first]);
  unsigned entries=0;
  for(unsigned k=0;k<8;++k)if(k!=first && (mask&(1U<<k))){
   ++entries;const auto route=token*8+k;
   assert(index[entries*2]==(uint32_t(ids[route])|((q[route]&0xffffU)<<16)));
   std::memcpy(&weight,index+entries*2+1,4);assert(weight==weights[route]);
  }
  assert(index[0]==entries);
 }
}
int main(){Check<false>();Check<true>();}
''', 'remote-owner-tail')


def test_count_ownership_and_host_ub_admission_match(tmp_path):
    run_cpp(tmp_path, r'''
#include <cassert>
#include <initializer_list>
#include "kernels/dispatch/low_latency/common/ep_memory_server_dedup_layout.h"
using namespace DispatchDedup;
int main(){

 for(unsigned cores=16;cores<=72;cores+=2)for(unsigned world:{16U,32U,64U,128U,256U})
 for(unsigned b:{1U,8U,256U})for(bool rdma:{false,true}){
  const auto workers=LlCountProducerCores(cores,b,world,b*8U,rdma);
  assert(workers==cores);
  for(auto mode:{EpServerDedupInputMode::Plain16,EpServerDedupInputMode::PrequantizedFp8Packs}){
   EpMemoryServerDedupLayout layout;
   assert(BuildDispatchServerDedupLayout(world,b,3584,8,512,&layout,mode,cores,rdma)==DISPATCH_DEDUP_SUCCESS);
   assert(DispatchDedupPacketUbFits(layout,b,8,512,world,cores,rdma));
   assert(DispatchServerDedupCountSendUbBytes(b*8U,512,workers)<=kEpServerDedupCountUbBytes);
  }
 }
 assert(LlCountProducerCores(56,512,32,16384,true)==28);
 assert(LlCountProducerCores(56,512,32,16384,false)==28);
}
''', 'count-owners-ub')
