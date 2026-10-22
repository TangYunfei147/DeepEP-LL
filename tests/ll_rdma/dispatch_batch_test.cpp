#include <cassert>
#include <iostream>
#include <vector>
#define DEEPEP_LL_RDMA 1
#include "kernels/dispatch/low_latency/device/ll_dispatch_rdma_batch.h"
using namespace AscendC;

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
                uint32_t capacity=std::min(8U,LlRdma::kCoreBytes/bytes);
                auto quiets=Model::quiets, puts=Model::puts;
                uint32_t batchBegin=0;
                Model::before=[&](const Model::Request& req) {
                    assert(req.qp==core && req.peer>=16);
                    if(req.bytes!=32) return;
                    // Even adversarial flags may only observe completed bodies.
                    for(uint32_t j=batchBegin;j<saved.size();++j) {
                        auto dst=remote.data()+uint64_t(16+j%peerCount)*1024+uint64_t(j)*(bytes+8192U);
                        for(uint32_t b=0;b<bytes;b+=512)
                            assert(std::memcmp(dst+b,saved[j].data()+b,480)==0);
                    }
                };
                for(uint32_t i=0;i<total;++i) {
                    for(uint32_t b=0;b<bytes;++b) ub[b]=uint8_t((i*13+b)%251+1);
                    saved.emplace_back(ub.begin(),ub.begin()+bytes);
                    auto q=Model::quiets;
                    batch.Enqueue(16+i%peerCount,uint64_t(i)*(bytes+8192U),{ub.data()});
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
                assert(Model::puts-puts==total*(bytes/512)*2);
#ifdef DEEPEP_LL_STAGE_TIMER
                timer.End(LlStageTimer::Kernel);timer.Finish();
                auto counters=timerWords.data()+core*LlStageTimer::kWordsPerCore;
                assert(counters[LlStageTimer::StageOffset(LlStageTimer::SendRdmaStage)+3U]==total);
                assert(counters[LlStageTimer::StageOffset(LlStageTimer::SendRdmaBodyWait)+3U]==(total+capacity-1)/capacity);
                assert(counters[LlStageTimer::StageOffset(LlStageTimer::SendRdmaRestore)+3U]==0);
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
    // Invalid destinations and shapes fail rather than using a wrong staging view.
    LlTransport::DispatchRdmaBatch batch;batch.Init(&args,4096,nullptr);
    caught=false;
    try { batch.Enqueue(1,0,{ub.data()}); } catch(const std::runtime_error&) { caught=true; }
    assert(caught);
    caught=false;
    try { batch.Init(&args,131584,nullptr); } catch(const std::runtime_error&) { caught=true; }
    assert(caught);
    batch.Init(&args,4096,nullptr);
    Model::pending.push_back({remote.data()+3500000,local.data()+3500000,1U,16U,core+1});
    batch.Enqueue(16,0,{ub.data()});batch.Flush({ub.data()});
    assert(Model::pending.size()==1 && Model::pending[0].qp==core+1);
    Model::pending.clear();
    std::cout<<"PASS "<<cases<<" batch cases: deferred source reads, reverse completion, reuse, multi-peer, QP isolation and failure\n";
}
