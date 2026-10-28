#include <cassert>
#include <iostream>
#include <vector>
#define DEEPEP_LL_RDMA 1
#include "kernels/dispatch/low_latency/device/ll_dispatch_rdma_batch.h"
#include "kernels/dispatch/low_latency/common/ll_dispatch_peer_owners.h"
#include "roce_so_model.h"
using namespace AscendC;
namespace Receive = DispatchDedup::PacketReceiveLayout;
static Receive::State probe(const uint8_t* packet, uint32_t bytes) {
    float sum=0;
    for(uint32_t block=0;block<bytes;block+=512)
        for(uint32_t word=0;word<8;++word) {
            float value;std::memcpy(&value,packet+block+480+word*4,4);sum+=value;
        }
    return Receive::Decode(sum,bytes/512);
}
static void flags(uint8_t* packet,uint32_t bytes,bool firstHop) {
    for(uint32_t block=0;block<bytes;block+=512)
        for(uint32_t word=0;word<8;++word) {
            float value=block==0 && word==0 && firstHop ? 1.5f : 1.0f;
            std::memcpy(packet+block+480+word*4,&value,4);
        }
}

int main() {
    std::vector<uint8_t> local(4U<<20), remote(local.size()), stages(16U*LlRdma::kCoreBytes), ub(128U<<10);
    Model::local=local.data(); Model::remote=remote.data();
    DispatchDedup::CommArgs args;
    args.rank=0; args.rankSize=32; args.reserved0=16; args.peerMemBytes=3U<<20;
    args.reservedAddresses[0]=local.data(); args.reservedAddresses[1]=stages.data();
    args.reservedConfig[0]=LlRdma::kBackend; args.reservedConfig[1]=8;
    args.reservedAddresses[2]=stages.data()+16U*LlRdma::kCoreBytes;
    cores=16; core=3;
    unsigned cases=0;
    for (uint32_t bytes : {512U,4096U,15872U,65536U,131072U}) {
        for (uint32_t peerCount : {1U,2U,8U}) {
            for (uint32_t total : {0U,1U,3U,8U,19U}) {
                LlTransport::DispatchRdmaBatch batch;
                batch.Init(&args,bytes);
                std::fill(remote.begin(),remote.end(),0);
                std::vector<std::vector<uint8_t>> saved;
                uint32_t capacity=std::min(8U,LlRdma::kCoreBytes/bytes);
                auto quiets=Model::quiets, puts=Model::puts;
                Model::before=[&](const Model::Request& req) {
                    assert(req.qp==core && req.peer>=16);
                    assert(req.bytes==bytes);
                    SoModel::Deliver(req);
                };
                for(uint32_t i=0;i<total;++i) {
                    for(uint32_t b=0;b<bytes;++b) ub[b]=uint8_t((i*13+b)%251+1);
                    flags(ub.data(),bytes,i%2!=0);
                    saved.emplace_back(ub.begin(),ub.begin()+bytes);
                    auto q=Model::quiets;
                    batch.Enqueue(16+i%peerCount,uint64_t(i)*(bytes+8192U),{ub.data()},core);
                    assert(std::memcmp(ub.data()+480,saved.back().data()+480,32)==0);
                    // No CQ wait until full; destroy UB immediately to test source lifetime.
                    if((i+1)%capacity) assert(Model::quiets==q);
                    std::fill(ub.begin(),ub.end(),0xcc);
                }
                batch.Flush({ub.data()});
                auto done=Model::quiets;batch.Flush({ub.data()});assert(done==Model::quiets);
                assert(!batch.Pending() && Model::pending.empty() && Model::gets==0);
                for(uint32_t i=0;i<total;++i) {
                    auto dst=remote.data()+uint64_t(16+i%peerCount)*1024+uint64_t(i)*(bytes+8192U);
                    assert(std::memcmp(dst,saved[i].data(),bytes)==0);
                }
                uint32_t expectedQuiets=0;
                for(uint32_t i=0;i<total;i+=capacity)
                    expectedQuiets+=std::min(peerCount,std::min(capacity,total-i));
                assert(Model::quiets-quiets==expectedQuiets);
                assert(Model::puts-puts==total);
                ++cases;
            }
        }
    }
    // CQ failure must not recycle sources. Readiness may already be visible.
    LlTransport::DispatchRdmaBatch failed;failed.Init(&args,4096);
    Model::before={}; std::fill(ub.begin(),ub.end(),9);
    failed.Enqueue(16,0,{ub.data()},core);
    auto puts=Model::puts;
    Model::before=[](const Model::Request&) { throw std::runtime_error("CQ failed"); };
    bool caught=false;
    try { failed.Flush({ub.data()}); } catch(const std::runtime_error&) { caught=true; }
    assert(caught && failed.Pending() && Model::puts==puts);
    Model::pending.clear();Model::before={};
    // Receiver may observe a complete message before the sender's CQ retires.
    // Its staging is nevertheless retained until quiet, not reused on arrival.
    LlTransport::DispatchRdmaBatch eager;eager.Init(&args,4096);
    std::fill(remote.begin(),remote.end(),0);
    flags(ub.data(),4096,true); const auto expected=ub;
    Model::onPost=[](Model::Request& r) { SoModel::Deliver(r); };
    auto q=Model::quiets;
    eager.Enqueue(16,0,{ub.data()},core);
    assert(eager.Pending() && Model::quiets==q && !Model::pending.empty());
    assert(probe(remote.data()+16384,4096)==Receive::State::KeepFlags);
    std::fill(ub.begin(),ub.end(),0xdc);
    assert(std::memcmp(stages.data()+uint64_t(core)*LlRdma::kCoreBytes,expected.data(),4096)==0);
    Model::onPost={}; eager.Flush({ub.data()});
    assert(!eager.Pending() && Model::pending.empty());
    // Without the target SO contract, sending all flag bytes first is unsafe.
    std::vector<uint8_t> unsafe(4096,0);
    for(uint32_t b=0;b<4096;b+=512) std::memcpy(unsafe.data()+b+480,expected.data()+b+480,32);
    assert(probe(unsafe.data(),4096)==Receive::State::KeepFlags);
    assert(std::memcmp(unsafe.data(),expected.data(),4096)!=0);
    // The sender chooses MTE versus RDMA once. The internal batch accepts only
    // that remote branch, retaining checks for route-dependent peer/range bounds.
    LlTransport::DispatchRdmaBatch batch;batch.Init(&args,4096);
    assert(!LlTransport::RemoteWrite(&args,1) && LlTransport::RemoteWrite(&args,16));
    caught=false;
    try { batch.Enqueue(uint32_t(args.rankSize),0,{ub.data()},core); } catch(const std::runtime_error&) { caught=true; }
    assert(caught);
    // Reconfiguration cannot silently forget pending data.
    batch.Init(&args,4096);
    flags(ub.data(),4096,true);batch.Enqueue(16,0,{ub.data()},core);
    caught=false;
    try { batch.Init(&args,LlRdma::kCoreBytes); } catch(const std::runtime_error&) { caught=true; }
    assert(caught && batch.Pending());
    batch.Flush({ub.data()});
    assert(Model::pending.empty());
    auto guard=stages.data()+uint64_t(core+1)*LlRdma::kCoreBytes;
    std::memset(guard,0xda,32);
    batch.Init(&args,LlRdma::kCoreBytes);
    flags(ub.data(),LlRdma::kCoreBytes,false);
    batch.Enqueue(16,0,{ub.data()},core);
    assert(!batch.Pending() && Model::pending.empty());
    for(unsigned i=0;i<32;++i) assert(guard[i]==0xda);
    caught=false;
    try { batch.Init(&args,131584); } catch(const std::runtime_error&) { caught=true; }
    assert(caught);
    batch.Init(&args,4096);
    Model::pending.push_back({remote.data()+3500000,local.data()+3500000,1U,16U,core+1});
    batch.Enqueue(16,0,{ub.data()},core);batch.Flush({ub.data()});
    assert(Model::pending.size()==1 && Model::pending[0].qp==core+1);
    Model::pending.clear();
    // One physical S24 owns different QP indices for different peers. Flush
    // must retire the exact peer/QP pair, not the hardware core number.
    cores=56;core=24;
    stages.resize(uint64_t(cores)*LlRdma::kCoreBytes);
    args.reservedAddresses[1]=stages.data();args.reservedAddresses[2]=stages.data()+stages.size();
    assert(DispatchDedup::LlDispatchQpOwner(0,32,28,17,0,8)==core);
    assert(DispatchDedup::LlDispatchQpOwner(0,32,28,20,4,8)==core);
    Model::onPost=[&](Model::Request& req){
        assert(req.qp==(req.peer==17?0U:4U));
        assert(req.src>=stages.data()+uint64_t(core)*LlRdma::kCoreBytes);
        assert(req.src<stages.data()+uint64_t(core+1)*LlRdma::kCoreBytes);
    };
    auto previousQuiets=Model::quiets;
    batch.Init(&args,4096);
    for(uint32_t i=0;i<3;++i){
        std::fill(ub.begin(),ub.end(),uint8_t(i+9));flags(ub.data(),4096,true);
        batch.Enqueue(i==1?20:17,1000000U+i*16384U,{ub.data()},i==1?4:0);
    }
    Model::pending.push_back({remote.data()+3500000,local.data()+3500000,1U,17U,4U});
    batch.Flush({ub.data()});
    assert(Model::quiets-previousQuiets==2U);
    assert(Model::pending.size()==1 && Model::pending[0].peer==17 && Model::pending[0].qp==4);
    Model::pending.clear();Model::onPost={};
    std::cout<<"PASS "<<cases<<" batch cases: SO final-fragment partial visibility, Compact/KeepFlags, source lifetime, reuse, multi-peer, QP isolation, full-stage capacity and failure\n";
}
