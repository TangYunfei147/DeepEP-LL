"""Execute the production gateway phase with delayed flags and repeated inbox reuse."""
from pathlib import Path
from tests.ll_rdma.test_combine_overhead import block, run_cpp
ROOT = Path(__file__).resolve().parents[2]

def test_gateway_fanout_and_reuse(tmp_path):
    code = (ROOT/'csrc/kernels/dispatch/low_latency/device/ll_paired_gateway.h').read_text(encoding='utf-8')
    phase = block(code, '__aicore__ inline void ForwardPairedGateway(')
    run_cpp(tmp_path, r"""
#include <cassert>
#include <vector>
#include <map>
#include <cstring>
#include <cstdint>
#include "kernels/dispatch/low_latency/common/ll_dispatch_packet_plan.h"
#define __aicore__
#define __gm__
constexpr int PIPE_ALL=0;
namespace AscendC {
enum class HardEvent{MTE2_S,S_MTE3,MTE3_S,S_MTE2};
template<class T> struct LocalTensor {
 T* p; T GetValue(uint32_t i){return p[i];} void SetValue(uint32_t i,T x){p[i]=x;}
 template<class U> LocalTensor<U> ReinterpretCast(){return {reinterpret_cast<U*>(p)};}
 LocalTensor operator[](uint32_t i){return {p+i};}
};
template<class Dummy=void> struct TBuf {
 std::vector<uint64_t> v;
 template<class T> LocalTensor<T> Get(){return {reinterpret_cast<T*>(v.data())};}
};
struct TPipe {void Reset(){} template<class T>void InitBuffer(TBuf<T>& b,uint32_t n){b.v.resize((n+7)/8);}};
template<int>void PipeBarrier(){}
struct Copy {uint16_t count;uint32_t bytes,src,dst,reserved;};
void DataCopy(LocalTensor<uint8_t> dst,uint8_t* src,uint32_t bytes){std::memcpy(dst.p,src,bytes);}
void DataCopyPad(uint8_t* dst,LocalTensor<uint8_t> src,Copy c){
 for(unsigned i=0;i<c.count;++i)std::memcpy(dst+i*(c.bytes+c.dst),src.p+i*(c.bytes+c.src*32),c.bytes);
}
}
using namespace AscendC;
using namespace DispatchDedup;

struct EpServerDedupDataLayout {
 uint64_t blockCount=2,tokenStride=1024,headerWireOffset=64,relayReadIndexWireOffset=96;
 uint64_t gatewayOffset=0,dispatchOffset=4096;
};
uint64_t DispatchDedupAlign(uint64_t x,uint64_t a){return (x+a-1)/a*a;}
uint64_t DispatchServerDedupDispatchOffset(EpServerDedupDataLayout d,uint32_t src,uint32_t dst,
 uint32_t expert,uint32_t q,uint32_t ranks,uint32_t local,uint32_t bs){
 return d.dispatchOffset+(((src+dst)%ranks*local+expert)*bs+q)*d.tokenStride;
}
struct PairedRouteExchange {std::vector<int32_t> ids;void ReadPeerAsync(LocalTensor<int32_t> dst){std::memcpy(dst.p,ids.data(),ids.size()*4);}};
template<HardEvent>void ServerDedupReceiveSync(){}
namespace PacketReceiveLayout {constexpr uint32_t kMarkerOffset=480;constexpr float kCompact=1.0f;}
struct ServerDedupPacketTail {
 LocalTensor<uint8_t> wire;uint32_t head,index;
 ServerDedupPacketTail(LocalTensor<uint8_t> w,uint32_t h,uint32_t i):wire(w),head(h),index(i){}
 LocalTensor<uint32_t> Header(){return wire[head].ReinterpretCast<uint32_t>();}
 LocalTensor<uint32_t> ReadIndex(){return wire[index].ReinterpretCast<uint32_t>();}
 LocalTensor<uint32_t> Entry(uint32_t i){return ReadIndex()[(i+1)*2];}
};
void PrepareServerDedupForward(ServerDedupPacketTail t){t.Header().SetValue(3,3);}
std::map<std::pair<uint32_t,uint64_t>,unsigned> deliveries;
std::map<uint8_t*,unsigned> probes;
namespace LlTransport {
uint8_t* Tensor(uint8_t* p,bool){return p;}
void Packet(const CommArgs* args,uint32_t target,uint64_t offset,LocalTensor<uint8_t> wire,uint32_t bytes){
 assert(target<16); // no second RoCE hop
 assert((++deliveries[{target,offset}]==1));
 std::memcpy(args->peerMems[target]+offset,wire.p,bytes);
}
}
void ReadServerDedupFlags(uint8_t* slot,uint32_t blocks,LocalTensor<float> flags,bool){
 for(unsigned b=0;b<blocks;++b)std::memcpy(flags.p+b*8,slot+b*512+480,32);
 // Model a delayed last fragment: first poll must not publish any output.
 if(probes[slot]++==0)flags.p[(blocks-1)*8+7]=0;
}
""" + phase + r"""
int main(){
 for(unsigned bs:{1U,8U,32U})for(unsigned k:{1U,8U})for(unsigned round=0;round<3;++round){
  const unsigned local=2,source=16;
  EpServerDedupDataLayout d;d.gatewayOffset=d.dispatchOffset+64ULL*bs*d.tokenStride;
  const unsigned bytes=d.gatewayOffset+bs*k*d.tokenStride+1024;
  std::vector<std::vector<uint8_t>> mem(16,std::vector<uint8_t>(bytes));
  CommArgs a{};a.rank=0;a.rankSize=32;for(unsigned rank=0;rank<16;++rank)a.peerMems[rank]=mem[rank].data();
  PairedRouteExchange exchange;
  // Alternate local-only, mixed, full-remote and singleton remote tokens.
  for(unsigned t=0;t<bs;++t)for(unsigned j=0;j<k;++j){
   unsigned expert=(t*3+j)%32;
   if((t+round)%4==0 || ((t+round)%4==1 && j>0))expert+=32;
   exchange.ids.push_back(expert);
  }
  LocalTensor<int32_t> ids{exchange.ids.data()};
  std::vector<unsigned> q(bs*k),hist(64);
  for(unsigned i=0;i<bs*k;++i)q[i]=hist[exchange.ids[i]]++;
  for(unsigned reuse=0;reuse<2;++reuse){
   deliveries.clear();probes.clear();std::vector<unsigned> inboxes;
   std::map<std::pair<uint32_t,uint64_t>,unsigned> expectedRole;
   for(unsigned i=0;i<bs*k;++i){
    const unsigned expert=exchange.ids[i],target=expert/local;
    if(target>=16)continue;
    const auto plan=LlPacketForRoute(ids,i,k,local,source,16,true,true);
    if(!plan.present)continue;
    inboxes.push_back(i);
    auto* slot=mem[0].data()+d.gatewayOffset+i*d.tokenStride;
    std::memset(slot,0,d.tokenStride);slot[0]=uint8_t(i+1);
    auto* h=reinterpret_cast<uint32_t*>(slot+d.headerWireOffset);
    h[0]=source;h[1]=i/k;h[2]=i%k;h[3]=plan.mask?2:1;h[4]=expert;h[5]=q[i];h[6]=0;
    auto* tail=reinterpret_cast<uint32_t*>(slot+d.relayReadIndexWireOffset);tail[0]=0;
    auto headOffset=DispatchServerDedupDispatchOffset(d,source,target,expert%local,q[i],32,local,bs);
    expectedRole[{target,headOffset}]=h[3];
    for(unsigned j=0;j<k;++j)if(j!=i%k && (plan.mask&(1U<<j))){
     unsigned m=i/k*k+j,e=exchange.ids[m],n=++tail[0];tail[n*2]=e|(q[m]<<16);tail[n*2+1]=0x3f000000;
     expectedRole[{e/local,DispatchServerDedupDispatchOffset(d,source,e/local,e%local,q[m],32,local,bs)}]=3;
    }
    for(unsigned b=0;b<2;++b)for(unsigned w=0;w<8;++w)
     reinterpret_cast<uint32_t*>(slot+b*512+480)[w]=b==0&&w==0&&plan.mask?0x3fc00000:0x3f800000;
   }
   TPipe pipe;
   // Every receiver participates, including cores with no final expert output.
   for(unsigned core=0;core<28;++core)ForwardPairedGateway(&a,d,0,bs,k,local,core,28,true,&exchange,&pipe);
   assert(deliveries.size()==expectedRole.size());
   for(auto [key,role]:expectedRole){
    assert(deliveries[key]==1);
    auto* h=reinterpret_cast<uint32_t*>(a.peerMems[key.first]+key.second+d.headerWireOffset);
    assert(h[3]==role && h[6]==1);
   }
   for(auto i:inboxes){
    auto* slot=mem[0].data()+d.gatewayOffset+i*d.tokenStride;
    assert(probes[slot]>=2);
    for(unsigned b=0;b<2;++b)for(unsigned w=0;w<8;++w)assert(reinterpret_cast<uint32_t*>(slot+b*512+480)[w]==0);
   }
  }
 }
}
""", 'paired-gateway')


def test_gateway_memory_excludes_combine_and_dispatch(tmp_path):
    run_cpp(tmp_path, r"""
#include <cassert>
#include <algorithm>
#include <initializer_list>
#include "kernels/dispatch/low_latency/common/ep_memory_server_dedup_layout.h"
using namespace DispatchDedup;
int main(){
 for(unsigned b:{1U,8U,32U,256U})for(unsigned k:{1U,8U,32U})
 for(unsigned h:{32U,3584U,7168U})for(unsigned e:{32U,384U,1024U})
 for(auto mode:{EpServerDedupInputMode::Plain16,EpServerDedupInputMode::PrequantizedFp8Packs}){
  auto d=DispatchServerDedupDataLayout(b,h,k,e,mode);
  auto combineEnd=d.sourceOffset+uint64_t(b)*k*d.returnStride+
      uint64_t(32)*b*std::min(k,e/32)*DispatchDedupAlign(uint64_t(h)*2,512);
  assert(d.gatewayOffset>=combineEnd);
  assert(d.gatewayOffset>=d.dispatchOffset+d.dispatchBytes);
  assert(d.gatewayBytes==uint64_t(b)*k*d.tokenStride);
  EpMemoryServerDedupLayout l;
  if(BuildDispatchServerDedupLayout(32,b,h,k,e,&l,mode,56,true)==DISPATCH_DEDUP_SUCCESS)
      assert(l.data.requiredBytes>=d.gatewayOffset+d.gatewayBytes);
 }
}
""", 'gateway-memory')


def test_gateway_cleanup_is_published_before_peer_sends():
    root=ROOT/'csrc/kernels/dispatch/low_latency'
    counts=(root/'device/dispatch_server_dedup_counts.h').read_text(encoding='utf-8')
    send=block(counts,'void ServerDedupCounts::SendCnt()')
    clean=send.index('block < gatewayBlocks_')
    publish=send.index('routeExchange_.PostFrame')
    assert 'SyncFunc<HardEvent::MTE3_S>();' in send[clean:publish]
    kernel=(root/'dispatch_server_dedup_kernel.h').read_text(encoding='utf-8')
    assert kernel.index('counts_.WaitPeerGateway();') < kernel.index('sender_.Process(')
    receiver=(root/'device/dispatch_server_dedup_recv.h').read_text(encoding='utf-8')
    assert receiver.index('ForwardPairedGateway(') < receiver.index('if (tasks_.Count() == 0U)')
    assert 'if (firstHop && !gatewayForwarded)' in receiver
    assert '!gatewayForwarded && i < members' in receiver
    assert 'if (firstHop) {' in receiver # metadata retained for combine
