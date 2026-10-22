#include <cassert>
#include <condition_variable>
#include <iostream>
#include <thread>
#include <vector>
#define DEEPEP_LL_RDMA 1
#define DEEPEP_TEST_ROUTE_THREADS 1
#include "kernels/dispatch/low_latency/device/ll_route_exchange.h"
using namespace AscendC;
using namespace DispatchDedup;
class KernelEnd {
    std::mutex mutex;
    std::condition_variable cv;
    unsigned arrived=0,round=0;
public:
    void Wait() {
        std::unique_lock<std::mutex> guard(mutex);
        unsigned before=round;
        if(++arrived==16) {arrived=0;++round;cv.notify_all();}
        else cv.wait(guard,[&]{return before!=round;});
    }
};
int main() {
    assert(DispatchServerDedupCountLayout(1024,72).stateBytes < 1024U*1024U);
    constexpr unsigned rounds=12;
    const unsigned routes[rounds]={8,16,64,256,8192,8,16,1024,1,24,512,8};
    const unsigned bankIds[rounds]={0,1,1,0,1,0,0,0,1,1,0,1};
    std::vector<uint8_t> memory[2]={std::vector<uint8_t>(4U<<20),std::vector<uint8_t>(4U<<20)};
    KernelEnd end[2];
    std::vector<std::thread> workers;
    auto value=[](unsigned source,unsigned iteration,unsigned index) {
        return int32_t((source*17U+iteration*31U+index)%384U);
    };
    for(unsigned side=0;side<2;++side) for(unsigned cpuCore=0;cpuCore<16;++cpuCore) {
        workers.emplace_back([&,side,cpuCore] {
            core=cpuCore;cores=16;
            unsigned rank=side*16,peer=rank^16;
            Model::local=memory[side].data();Model::remote=memory[1-side].data();
            CommArgs a;a.rank=rank;a.rankSize=32;a.reserved0=16;a.peerMemBytes=4U<<20;
            a.reservedConfig[0]=LlRdma::kBackend;
            a.reservedConfig[1]=16;
            a.reservedAddresses[0]=memory[side].data();
            a.peerMems[rank]=memory[side].data()+rank*1024;
            std::vector<int32_t> data(kRouteExchangeMaxBytes/4);
            std::vector<uint8_t> scratch(256);
            for(unsigned iteration=0;iteration<rounds;++iteration) {
                for(unsigned i=0;i<routes[iteration];++i) data[i]=value(rank,iteration,i);
                PairedRouteExchange exchange;
                exchange.Init(&a,routes[iteration],uint64_t(bankIds[iteration])*(2U<<20));
                exchange.Publish({data.data()},{scratch.data()});
                // Only cores owning this host's expert interval need the inbox.
                bool consumes=side==0?cpuCore<8:cpuCore>=8;
                if(consumes) {
                    if(cpuCore==7 || cpuCore==15) std::this_thread::sleep_for(std::chrono::milliseconds(2));
                    exchange.ReadPeer({data.data()});
                    for(unsigned i=0;i<routes[iteration];++i) assert(data[i]==value(peer,iteration,i));
                }
                exchange.Complete({scratch.data()});
                assert(Model::pending.empty());
                end[side].Wait(); // real same-stream kernel completion, not a protocol barrier
            }
            assert(Model::gets==0);
            assert(Model::puts==(cpuCore==0?rounds*3:0));
            assert(Model::quiets==Model::puts);
        });
    }
    for(auto& worker:workers) worker.join();
    // Fail body completion: no commit may be posted and the previous bank's
    // reuse watermark must not be advanced to the failed publication.
    Model::local=memory[0].data();Model::remote=memory[1].data();core=0;cores=16;
    CommArgs a;a.rank=0;a.rankSize=32;a.reserved0=16;a.peerMemBytes=4U<<20;
    a.reservedConfig[0]=LlRdma::kBackend;a.reservedConfig[1]=16;
    a.reservedAddresses[0]=memory[0].data();a.peerMems[0]=memory[0].data();
    std::vector<int32_t> data(64,1);std::vector<uint8_t> scratch(256);
    PairedRouteExchange exchange;exchange.Init(&a,8,0);
    Model::before=[](const Model::Request&) {throw std::runtime_error("CQ failed");};
    bool failed=false;
    try {exchange.Publish({data.data()},{scratch.data()});}
    catch(const std::runtime_error&) {failed=true;}
    assert(failed && Model::puts==1 && Model::pending.size()==1 && Model::pending[0].bytes==32);
    Model::pending.clear();Model::before={};
    std::cout<<"PASS paired route exchange: 12 epochs, variable size, same/alternating bank, delayed and empty cores; 3 puts per rank\n";
}
