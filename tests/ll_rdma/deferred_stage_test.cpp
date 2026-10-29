
#include <cassert>
#include <iostream>
#include <vector>
#define DEEPEP_LL_RDMA 1
#define DEEPEP_TEST_DEFERRED_MTE3 1
#include "kernels/dispatch/low_latency/device/ll_dispatch_rdma_batch.h"
using namespace AscendC;
int main(){
 cores=16;core=0;
 unsigned cases=0;
 for(unsigned bytes:{512U,4096U,65536U,131072U})for(unsigned lanes:{1U,2U,8U})
 for(unsigned pattern:{0U,1U,2U})for(unsigned n:{0U,1U,2U,3U,9U,19U}){
  std::vector<uint8_t> local(8U<<20),remote(local.size()),stage(16U*LlRdma::kCoreBytes);
  std::vector<std::vector<uint8_t>> ub(lanes,std::vector<uint8_t>(bytes));
  Model::local=local.data();Model::remote=remote.data();Model::pending.clear();
  Model::onPost={};Model::onQuiet={};Model::before={};Model::quiets=Model::puts=0;
  deferredMte3.clear();issuedMte3=0;
  DispatchDedup::CommArgs a{};a.rank=0;a.rankSize=32;a.reserved0=16;a.peerMemBytes=local.size();
  a.reservedConfig[0]=LlRdma::kBackend;a.reservedConfig[1]=cores;
  a.reservedAddresses[0]=local.data();a.reservedAddresses[1]=stage.data();
  LlTransport::DispatchRdmaBatch batch;batch.Init(&a,bytes);
  const uint32_t stride=bytes+4096U;
  std::vector<unsigned> order,expected;std::vector<bool> isRemote(n);
  unsigned capacity=(LlRdma::kCoreBytes/bytes),remoteCount=0;
  Model::onPost=[&](Model::Request& req){
   unsigned index=unsigned((req.dst-Model::remote-uint64_t(req.peer)*1024)/stride);
   assert(req.qp==core && isRemote[index]);
   for(unsigned j=0;j<bytes;++j)assert(req.src[j]==uint8_t(index+1));
   order.push_back(index);
  };
  Model::before=[&](const Model::Request& req){
   unsigned index=unsigned((req.dst-Model::remote-uint64_t(req.peer)*1024)/stride);
   // The model translated Request::dst at enqueue; all immutable sources survive until quiet.
   for(unsigned j=0;j<bytes;++j)assert(req.src[j]==uint8_t(index+1));
  };
  for(unsigned lane=0;lane<lanes;++lane)SetFlag<HardEvent::MTE3_V>(lane);
  for(unsigned i=0;i<n;++i){
   const bool network=pattern==0 || (pattern==1 ? i%2==0 : i%3!=1);
   isRemote[i]=network;expected.push_back(i);
   unsigned lane=i%lanes;
   WaitFlag<HardEvent::MTE3_V>(lane);
   std::fill(ub[lane].begin(),ub[lane].end(),uint8_t(i+1)); // common preparation B
   auto before=ub[lane];
   batch.SubmitPending(); // A uses its old, distinct lane; never clobbers B
   assert(ub[lane]==before);
   if(network){
    ++remoteCount;auto puts=Model::puts;
    batch.Enqueue(16+i%2,uint64_t(i)*stride,{ub[lane].data()},lanes>1);
    assert(batch.HasStaged()==(lanes>1));
    assert(Model::puts==puts+unsigned(lanes==1));
   }else{
    order.push_back(i);
    DataCopy(LlTransport::Tensor(local.data()+uint64_t(i)*stride,true),LocalTensor<uint8_t>{ub[lane].data()},bytes);
   }
   SetFlag<HardEvent::MTE3_V>(lane);
  }
  batch.SubmitPending(); // same position before sender w39
  assert(!batch.HasStaged() && order==expected);
  for(unsigned lane=0;lane<lanes;++lane)WaitFlag<HardEvent::MTE3_V>(lane);
  batch.Flush({ub[0].data()});
  assert(!batch.Pending() && Model::pending.empty() && deferredMte3.empty());
  for(unsigned i=0;i<n;++i){
   auto* out=isRemote[i]?remote.data()+uint64_t(16+i%2)*1024+uint64_t(i)*stride:local.data()+uint64_t(i)*stride;
   for(unsigned j=0;j<bytes;++j)assert(out[j]==uint8_t(i+1));
  }
  for(auto& row:events)for(bool live:row)assert(!live);
  ++cases;
 }
 std::cout<<"PASS "<<cases<<" deferred staging cases: delayed DMA, lane reuse, local/remote order, batch source lifetime\n";
}
