#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>
#define DEEPEP_LL_RDMA 1
#define DEEPEP_TEST_DEFERRED_MTE3 1
#include "kernels/dispatch/low_latency/device/ll_dispatch_rdma_batch.h"
#include "kernels/dispatch/low_latency/device/ll_count_frame_transport.h"
#include "kernels/dispatch/low_latency/common/ll_count_assignment.h"
using namespace AscendC;
int main(){
 std::vector<uint8_t> local(2<<20),remote(2<<20),stage(72*LlRdma::kCoreBytes),record(2048),scratch(256);
 Model::local=local.data();Model::remote=remote.data();
 DispatchDedup::CommArgs a{};a.reserved0=16;a.peerMemBytes=2<<20;a.reservedConfig[0]=LlRdma::kBackend;
 a.reservedAddresses[0]=local.data();a.reservedAddresses[1]=stage.data();
 unsigned cases=0;
 for(unsigned world:{32U,64U,128U,256U})for(unsigned c:{16U,28U,56U,72U})for(unsigned rank:{0U,world/2,world-1})
 for(unsigned worker=0;worker<c;++worker){
  core=worker;cores=c;a.rank=rank;a.rankSize=world;a.reservedConfig[1]=c;a.peerMems[rank]=local.data();
  LlTransport::CountRetirement pending;pending.Init(&a);
  std::vector<unsigned> peers;
  const unsigned experts=1024/world,bytes=(experts+1)*32;
  auto before=Model::quiets;
  for(unsigned task=worker;task<world-16;task+=c){
   const auto peer=DispatchDedup::CountRemotePeer(rank,world,task);peers.push_back(peer);
   std::fill(record.begin(),record.end(),uint8_t(peer));
   LlTransport::DirectCountFrame(&a,peer,0,{record.data()},experts,{scratch.data()});pending.Posted(peer);
   assert(Model::quiets==before);
  }
  // Every UB frame and ALL shared payload staging may change before NIC reads
  // the count sources. Dedicated per-destination sources must remain intact.
  std::fill(record.begin(),record.end(),0xcc);std::fill(stage.begin(),stage.end(),0xdd);
  for(auto& request:Model::pending){
   assert(request.src>=local.data()+LlTransport::kCountTxBase);
   for(unsigned i=0;i<bytes;++i)assert(request.src[i]==uint8_t(request.peer));
  }
  // Metadata cores retire their own QPs; payload cores have disjoint QPs.
  pending.Finish({scratch.data()});
  assert(!pending.Pending() && Model::pending.empty());
  assert(Model::quiets-before==peers.size()); // no duplicate count quiet
  // Zero-payload/zero-work core remains covered by final retirement too.
  pending.Init(&a);before=Model::quiets;
  for(unsigned task=worker;task<world-16;task+=c){
   auto peer=DispatchDedup::CountRemotePeer(rank,world,task);
   LlTransport::DirectCountFrame(&a,peer,0,{record.data()},experts,{scratch.data()});pending.Posted(peer);
  }
  pending.Finish({scratch.data()});assert(Model::quiets-before==peers.size() && Model::pending.empty());
  ++cases;
 }
 std::cout<<"PASS isolated count sources and metadata CQ retirement "<<cases<<" cases\n";
}
