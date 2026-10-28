from pathlib import Path
from tests.ll_rdma.test_combine_overhead import block, run_cpp
ROOT=Path(__file__).resolve().parents[2]

def test_original_partition_vector_q_and_weight_tail(tmp_path):
    source = (ROOT/'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h').read_text(encoding='utf-8')
    methods = '\n'.join(block(source, name) for name in (
        '__aicore__ inline uint32_t CalTokenSendExpertCnt(',
        '__aicore__ inline void FillTail('))
    run_cpp(tmp_path, r'''
#include <cassert>
#include <vector>
#include <cstring>
#include "kernels/dispatch/low_latency/common/ll_fullmesh_first_hop.h"
constexpr unsigned kLlRouteQMask=0xffffU;
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
constexpr int PIPE_V=1;
enum class HardEvent{V_S};
template<int>void PipeBarrier(){}
template<class T>void Duplicate(LocalTensor<T> t,T x,uint32_t n){for(unsigned i=0;i<n;++i)t.SetValue(i,x);}
void Sub(LocalTensor<int32_t> d,LocalTensor<int32_t> a,LocalTensor<int32_t> b,uint32_t n){for(unsigned i=0;i<n;++i)d.SetValue(i,a.GetValue(i)-b.GetValue(i));}
void Abs(LocalTensor<float> d,LocalTensor<float> a,uint32_t n){for(unsigned i=0;i<n;++i)d.ReinterpretCast<uint32_t>().SetValue(i,a.ReinterpretCast<uint32_t>().GetValue(i)&0x7fffffffU);}
void Mins(LocalTensor<int32_t> d,LocalTensor<int32_t> a,int x,uint32_t n){for(unsigned i=0;i<n;++i)d.SetValue(i,a.GetValue(i)<x?a.GetValue(i):x);}
template<class T>void ReduceSum(LocalTensor<T> d,LocalTensor<T> a,LocalTensor<T> work,uint32_t n){
 unsigned sum=0;for(unsigned i=0;i<n;++i)sum+=a.template ReinterpretCast<uint32_t>().GetValue(i);
 // Model bit-pattern FP32 reduction under the kernel's preserved CTRL mode.
 work.template ReinterpretCast<uint32_t>().SetValue(0,0xdeadbeef);
 d.template ReinterpretCast<uint32_t>().SetValue(0,sum);
}
struct Args {uint32_t rank=7;};
template<bool Prequantized> struct Test {
 Args a;Args* args_=&a;uint32_t start_=3,k_=8,cachedTailEntries_=0;
 bool remoteIssue_=true,enableDedup_=true;
 struct Layout {uint32_t headerPayloadOffset=64,readIndexPayloadOffset=96;} data_;
 LocalTensor<float> weights_,routeWeights_;
 LocalTensor<int32_t> routeIds_;
 Buffer qPlanBuf_,qDstBuf_,qSubBuf_,qWorkBuf_;
 template<HardEvent> void Sync(){}
''' + methods + r'''
};
template<bool P> void Check(){
 std::vector<float> weights(160);std::vector<int32_t> ids(160);std::vector<uint32_t> q(160),raw(128),dst(160),sub(160),work(160),hist(384);
 for(unsigned i=0;i<160;++i){weights[i]=float(i+1)/8;ids[i]=i*17%37;q[i]=0x80070000U|(i<8?0U:hist[ids[i]]);++hist[ids[i]];}
 Test<P> t;t.weights_={weights.data(),160};t.routeWeights_={weights.data()+24,8};
 t.routeIds_={ids.data(),160};t.qPlanBuf_={q.data()};t.qDstBuf_={dst.data()};t.qSubBuf_={sub.data()};t.qWorkBuf_={work.data()};
 LocalTensor<uint8_t> payload{reinterpret_cast<uint8_t*>(raw.data()),raw.size()*4};
 for(bool remote:{false})for(unsigned token:{3U})for(uint32_t mask:{255U,60U,0U,8U}){
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




def test_no_eight_qp_planning_remains():
    code=(ROOT/'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h').read_text(encoding='utf-8')
    for token in ['qPlanBuf_', 'SendRemotePackets', 'LlBuildRemotePacketOwners', 'networkRoundRobin_', 'RouteWeight(', 'LlBuildRouteQ(']:
        assert token not in code
    assert 'LlSenderHasWork(enableDedup_, count_' in code
    assert 'LlStageTimer' not in code
