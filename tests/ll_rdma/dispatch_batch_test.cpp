#include <cassert>
#include <iostream>
#include <vector>
#define DEEPEP_LL_RDMA 1
#include "kernels/dispatch/low_latency/device/ll_dispatch_rdma_batch.h"
#include "kernels/common/ep_packet_receive_layout.h"
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
    args.reservedConfig[0]=LlRdma::kBackend; args.reservedConfig[1]=16;
    cores=16; core=3;
    std::vector<uint64_t> timerWords(16U*LlStageTimer::kWordsPerCore);
    unsigned cases=0;
    for (uint32_t bytes : {512U,4096U,15872U,65536U,131072U}) {
        for (uint32_t peerCount : {1U,2U,8U}) {
            for (uint32_t total : {0U,1U,3U,8U,19U}) {
                LlTransport::DispatchRdmaBatch batch;
                LlStageTimer::DeviceTimer timer;
                LlStageTimer::DeviceTimer* timerPtr=nullptr;
#ifdef DEEPEP_LL_STAGE_TIMER
                std::fill(timerWords.begin(),timerWords.end(),0);
                timer.Init(reinterpret_cast<GM_ADDR>(timerWords.data()),1U,1U);
                timer.Start(LlStageTimer::Kernel);
                timerPtr=&timer;
#endif
                batch.Init(&args,bytes,timerPtr);
                std::fill(remote.begin(),remote.end(),0);
                std::vector<std::vector<uint8_t>> saved;
                const bool gated=bytes+32U<=LlRdma::kCoreBytes;
                uint32_t capacity=gated ? std::min(8U,LlRdma::kCoreBytes/(bytes+32U)) : 1U;
                auto quiets=Model::quiets, puts=Model::puts;
                uint32_t batchBegin=0;
                Model::before=[&](const Model::Request& req) {
                    assert(req.qp==core && req.peer>=16);
                    if(gated && req.bytes==bytes) {
                        // NIC may expose later fragments/flags before earlier body.
                        // The first complete flag block remains zero throughout.
                        for(uint32_t end=bytes;end;end-=32) {
                            std::memcpy(req.dst+end-32,req.src+end-32,32);
                            assert(probe(req.dst,bytes)==Receive::State::Pending);
                        }
                        return;
                    }
                    if(req.bytes!=32) return;
                    // Even adversarial flags may only observe completed bodies.
                    for(uint32_t j=batchBegin;j<saved.size();++j) {
                        auto dst=remote.data()+uint64_t(16+j%peerCount)*1024+uint64_t(j)*(bytes+8192U);
                        for(uint32_t b=0;b<bytes;b+=512)
                            assert(std::memcmp(dst+b,saved[j].data()+b,480)==0);
                    }
                    if(gated) {
                        auto packet=req.dst-480;
                        assert(probe(packet,bytes)==Receive::State::Pending);
                        // Publish in words, marker last; no atomic packet write.
                        for(uint32_t word=8;word;--word) {
                            std::memcpy(req.dst+(word-1)*4,req.src+(word-1)*4,4);
                            if(word>1) assert(probe(packet,bytes)==Receive::State::Pending);
                        }
                        float marker;std::memcpy(&marker,req.src,4);
                        assert(probe(packet,bytes)==(marker==1.5f ? Receive::State::KeepFlags : Receive::State::Compact));
                    }
                };
                for(uint32_t i=0;i<total;++i) {
                    for(uint32_t b=0;b<bytes;++b) ub[b]=uint8_t((i*13+b)%251+1);
                    flags(ub.data(),bytes,i%2!=0);
                    saved.emplace_back(ub.begin(),ub.begin()+bytes);
                    auto q=Model::quiets;
                    batch.Enqueue(16+i%peerCount,uint64_t(i)*(bytes+8192U),{ub.data()});
                    assert(std::memcmp(ub.data()+480,saved.back().data()+480,32)==0);
                    // No CQ wait until full; destroy UB immediately to test source lifetime.
                    if((i+1)%capacity) assert(Model::quiets==q);
                    else batchBegin=i+1;
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
                    expectedQuiets+=2*std::min(peerCount,std::min(capacity,total-i));
                assert(Model::quiets-quiets==expectedQuiets);
                assert(Model::puts-puts==total*(gated ? 2U : (bytes/512)*2U));
#ifdef DEEPEP_LL_STAGE_TIMER
                timer.End(LlStageTimer::Kernel);timer.Finish();
                auto counters=timerWords.data()+core*LlStageTimer::kWordsPerCore;
                assert(counters[LlStageTimer::StageOffset(LlStageTimer::SendRdmaStage)+3U]==total);
                assert(counters[LlStageTimer::StageOffset(LlStageTimer::SendRdmaBodyWait)+3U]==(total+capacity-1)/capacity);
                assert(counters[LlStageTimer::StageOffset(LlStageTimer::SendRdmaFlagPost)+3U]==(total+capacity-1)/capacity);
                assert(counters[LlStageTimer::StageOffset(LlStageTimer::SendRdmaFlagWait)+3U]==(total+capacity-1)/capacity);
                assert(counters[LlStageTimer::StageOffset(LlStageTimer::SendRdmaRestore)+3U]==(gated ? 0U : total));
#endif
                ++cases;
            }
        }
    }
    // CQ failure must prevent flags and must not recycle descriptors/staging.
    LlTransport::DispatchRdmaBatch failed;failed.Init(&args,4096,nullptr);
    Model::before={}; std::fill(ub.begin(),ub.end(),9);
    failed.Enqueue(16,0,{ub.data()});
    auto puts=Model::puts;
    Model::before=[](const Model::Request&) { throw std::runtime_error("CQ failed"); };
    bool caught=false;
    try { failed.Flush({ub.data()}); } catch(const std::runtime_error&) { caught=true; }
    assert(caught && failed.Pending() && Model::puts==puts);
    Model::pending.clear();Model::before={};
    // A commit CQ error cannot retire the GM slot either.
    LlTransport::DispatchRdmaBatch failedGate;failedGate.Init(&args,4096,nullptr);
    flags(ub.data(),4096,true);failedGate.Enqueue(16,0,{ub.data()});
    Model::before=[](const Model::Request& r) { if(r.bytes==32)throw std::runtime_error("gate CQ failed"); };
    caught=false;
    try { failedGate.Flush({ub.data()}); } catch(const std::runtime_error&) { caught=true; }
    assert(caught && failedGate.Pending());
    Model::pending.clear();Model::before={};
    // The sender chooses MTE versus RDMA once. The internal batch accepts only
    // that remote branch, retaining checks for route-dependent peer/range bounds.
    LlTransport::DispatchRdmaBatch batch;batch.Init(&args,4096,nullptr);
    assert(!LlTransport::RemoteWrite(&args,1) && LlTransport::RemoteWrite(&args,16));
    caught=false;
    try { batch.Enqueue(uint32_t(args.rankSize),0,{ub.data()}); } catch(const std::runtime_error&) { caught=true; }
    assert(caught);
    // Switching to the extreme fallback cannot silently forget pending data.
    batch.Init(&args,4096,nullptr);
    flags(ub.data(),4096,true);batch.Enqueue(16,0,{ub.data()});
    caught=false;
    try { batch.Init(&args,LlRdma::kCoreBytes,nullptr); } catch(const std::runtime_error&) { caught=true; }
    assert(caught && batch.Pending());
    batch.Flush({ub.data()});
    assert(Model::pending.empty());
    auto guard=stages.data()+uint64_t(core+1)*LlRdma::kCoreBytes;
    std::memset(guard,0xda,32);
    batch.Init(&args,LlRdma::kCoreBytes,nullptr);
    flags(ub.data(),LlRdma::kCoreBytes,false);
    batch.Enqueue(16,0,{ub.data()});
    assert(!batch.Pending() && Model::pending.empty());
    for(unsigned i=0;i<32;++i) assert(guard[i]==0xda);
    caught=false;
    try { batch.Init(&args,131584,nullptr); } catch(const std::runtime_error&) { caught=true; }
    assert(caught);
    batch.Init(&args,4096,nullptr);
    Model::pending.push_back({remote.data()+3500000,local.data()+3500000,1U,16U,core+1});
    batch.Enqueue(16,0,{ub.data()});batch.Flush({ub.data()});
    assert(Model::pending.size()==1 && Model::pending[0].qp==core+1);
    Model::pending.clear();
    std::cout<<"PASS "<<cases<<" batch cases: gated partial/reordered visibility, Compact/KeepFlags, source lifetime, reuse, multi-peer, QP isolation, fallback and failure\n";
}
