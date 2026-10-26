#include <array>
#include <cassert>
#include <condition_variable>
#include <future>
#include <iostream>
#include <thread>
#include <vector>
#define DEEPEP_LL_RDMA 1
#define DEEPEP_TEST_ROUTE_THREADS 1
#include "kernels/dispatch/low_latency/device/ll_route_exchange.h"
using namespace AscendC;
using namespace DispatchDedup;
using namespace std::chrono_literals;

static constexpr uint64_t bankBytes=2U<<20;
static constexpr unsigned testCores=16;
struct Shape { unsigned b,k,e; };
static uint32_t ConsumerMask(unsigned side,Shape shape) {
    const bool all=shape.b<512U;
    const unsigned countCores=all?testCores:testCores/2U;
    uint32_t mask=0;
    for(unsigned c=0;c<testCores;++c) {
        if(!all && c<testCores/2U) continue;
        const unsigned worker=all?c:c-testCores/2U;
        const unsigned base=shape.e/countCores,extra=shape.e%countCores;
        const unsigned first=worker*base+std::min(worker,extra);
        const unsigned last=first+base+(worker<extra);
        if(std::max(first,side*shape.e/2U)<std::min(last,(side+1U)*shape.e/2U)) mask|=1U<<c;
    }
    assert(mask); // Valid EP32 shapes always have at least one route consumer.
    return mask;
}
static uint32_t ReadWord(uint8_t* p) {
    std::lock_guard<std::mutex> guard(memoryMutex);
    uint32_t value;std::memcpy(&value,p,sizeof(value));return value;
}

// NIC progress is independent of quiet. Body/commit delays differ by source;
// delivery uses reverse eligible order and never assumes cross-QP ordering.
// Every inbox overwrite checks that all readers of its previous body retired.
class Network {
    enum Kind { Body,Commit };
    struct Flight {
        Model::Request request;
        unsigned side,epoch;
        uint64_t bank;
        Kind kind;
        std::vector<uint8_t> source;
        std::chrono::steady_clock::time_point due;
        bool done=false;
        std::string error;
    };
    struct Inbox { unsigned epoch=0;uint32_t readers=0,retired=0; };
    std::mutex mutex;
    std::condition_variable cv;
    std::vector<Flight> flights;
    std::vector<Shape> shapes;
    Inbox inbox[2][2];
    bool stop=false,failBody;
    std::thread nic;
    void Run() {
        std::unique_lock<std::mutex> guard(mutex);
        while(!stop) {
            size_t next=flights.size();
            const auto now=std::chrono::steady_clock::now();
            for(size_t i=flights.size();i--;) {
                if(!flights[i].done && flights[i].due<=now) {next=i;break;}
            }
            if(next==flights.size()) {cv.wait_for(guard,100us);continue;}
            auto& f=flights[next];
            if(f.kind==Body) {
                auto& target=inbox[1U-f.side][f.bank/bankBytes];
                assert(!target.epoch || target.retired==target.readers);
                target={f.epoch,ConsumerMask(1U-f.side,shapes.at(f.epoch-1U)),0U};
            }
            {
                std::lock_guard<std::mutex> memoryGuard(memoryMutex);
                assert(std::memcmp(f.request.src,f.source.data(),f.source.size())==0);
                std::memcpy(f.request.dst,f.request.src,f.request.bytes);
            }
            if(failBody && f.kind==Body) f.error="injected body CQ failure";
            f.done=true;cv.notify_all();
        }
    }
public:
    explicit Network(std::vector<Shape> cases,bool failure=false)
        :shapes(std::move(cases)),failBody(failure),nic([this]{Run();}) {}
    ~Network() {
        {std::lock_guard<std::mutex> guard(mutex);stop=true;cv.notify_all();}
        nic.join();
    }
    void Post(Model::Request& request,unsigned side,uint8_t* localBase,uint8_t* remoteBase) {
        std::lock_guard<std::mutex> guard(mutex);
        const uint64_t offset=uint64_t(request.dst-remoteBase)-uint64_t(request.peer)*1024U;
        const uint64_t bank=offset/bankBytes*bankBytes;
        Kind kind;
        if(offset-bank==kRouteRx) kind=Body;
        else {assert(offset-bank==kRouteCommit);kind=Commit;}
        const unsigned epoch=ReadWord(localBase+side*16U*1024U+bank+kRouteSequence);
        assert(epoch && request.qp==0U);
        if(kind==Commit) {
            bool completed=false;
            for(const auto& f:flights) if(f.side==side && f.bank==bank && f.epoch==epoch && f.kind==Body)
                completed=f.done && f.error.empty();
            assert(completed); // Failed/unfinished body must never publish.
        }
        std::vector<uint8_t> source(request.bytes);
        {std::lock_guard<std::mutex> memoryGuard(memoryMutex);std::memcpy(source.data(),request.src,request.bytes);}
        const auto delay=kind==Body?(side?2ms:1ms):(side?4ms:1ms);
        request.id=flights.size();
        flights.push_back({request,side,epoch,bank,kind,std::move(source),std::chrono::steady_clock::now()+delay});
        cv.notify_all();
    }
    void RetireRead(unsigned side,unsigned cpuCore,unsigned epoch,uint64_t bank) {
        std::lock_guard<std::mutex> guard(mutex);
        auto& target=inbox[side][bank/bankBytes];
        assert(target.epoch==epoch && (target.readers&(1U<<cpuCore)) && !(target.retired&(1U<<cpuCore)));
        target.retired|=1U<<cpuCore;
    }
    void Wait(uint64_t id) {
        std::unique_lock<std::mutex> guard(mutex);
        assert(id<flights.size());
        if(!cv.wait_for(guard,5s,[&]{return flights[id].done;})) throw std::runtime_error("NIC progress timeout");
        const auto& f=flights[id];
        if(!f.error.empty()) throw std::runtime_error(f.error);
        std::lock_guard<std::mutex> memoryGuard(memoryMutex);
        assert(std::memcmp(f.request.src,f.source.data(),f.source.size())==0);
    }
    void Attach(unsigned side,uint8_t* localBase,uint8_t* remoteBase) {
        Model::local=localBase;Model::remote=remoteBase;
        Model::pending.clear();Model::before={};Model::puts=Model::quiets=Model::gets=0;
        Model::onPost=[=](Model::Request& request){Post(request,side,localBase,remoteBase);};
        Model::onQuiet=[this](uint32_t peer,uint32_t qp){
            for(const auto& request:Model::pending) if(request.peer==peer && request.qp==qp) Wait(request.id);
        };
    }
};

class Completion {
    std::mutex mutex;
    std::condition_variable cv;
    unsigned arrived=0,round=0,participants;
public:
    explicit Completion(unsigned count):participants(count) {}
    void Wait() {
        std::unique_lock<std::mutex> guard(mutex);
        const unsigned before=round;
        if(++arrived==participants) {arrived=0;++round;cv.notify_all();}
        else if(!cv.wait_for(guard,5s,[&]{return before!=round;})) throw std::runtime_error("completion timeout");
    }
};
static CommArgs Args(unsigned side,std::vector<uint8_t>& local) {
    CommArgs a{};a.rank=side*16U;a.rankSize=32;a.reserved0=16;a.peerMemBytes=4U<<20;
    a.reservedConfig[0]=LlRdma::kBackend;a.reservedConfig[1]=cores;
    a.reservedAddresses[0]=local.data();a.peerMems[a.rank]=local.data()+a.rank*1024U;
    return a;
}

static void TestEpochs(bool interleaveCombine) {
    const std::vector<Shape> shapes={{1,8,384},{2,8,384},{8,8,384},{32,8,384},
        {256,32,1024},{1,8,384},{2,8,384},{512,2,384},{1,1,32},{3,8,384},{1024,8,1024},{1,8,384}};
    const unsigned rounds=shapes.size();
    std::vector<uint8_t> memory[2]={std::vector<uint8_t>(5U<<20),std::vector<uint8_t>(5U<<20)};
    Network network(shapes);
    Completion localEnd[2]={Completion(testCores),Completion(testCores)};
    Completion combineReuse(2U*testCores);
    std::vector<std::thread> workers;
    auto value=[&](unsigned source,unsigned iteration,unsigned index) {
        // Includes source tables with zero routes to the opposite host.
        const unsigned limit=iteration%3U==0U?shapes[iteration].e/2U:shapes[iteration].e;
        return int32_t((source*17U+iteration*31U+index)%limit);
    };
    for(unsigned side=0;side<2;++side) for(unsigned cpuCore=0;cpuCore<testCores;++cpuCore) {
        workers.emplace_back([&,side,cpuCore] {
            core=cpuCore;cores=testCores;
            const unsigned rank=side*16U,peer=rank^16U;
            network.Attach(side,memory[side].data(),memory[1U-side].data());
            auto a=Args(side,memory[side]);
            std::vector<int32_t> data(kRouteExchangeMaxBytes/4U);
            std::array<uint8_t,256> postScratch{},commitScratch{},finishScratch{};
            unsigned magic=0;
            for(unsigned iteration=0;iteration<rounds;++iteration) {
                const unsigned routes=shapes[iteration].b*shapes[iteration].k;
                magic^=1U;
                const uint64_t bank=uint64_t(magic)*bankBytes;
                for(unsigned i=0;i<routes;++i) data[i]=value(rank,iteration,i);
                PairedRouteExchange exchange;
                exchange.Init(&a,routes,bank);
                const bool consumes=(ConsumerMask(side,shapes[iteration])&(1U<<cpuCore))!=0U;
                if(consumes || cpuCore==0U) {
                    if(cpuCore==0U && side==1U && iteration%2U) std::this_thread::sleep_for(3ms);
                    exchange.PostBody({data.data()},{postScratch.data()});
                    if(cpuCore==0U) {
                        assert(Model::puts==iteration*2U+1U && Model::quiets==iteration*2U);
                        assert(postScratch[0]==0xa5 && postScratch[127]==0xa5 && postScratch[128]==0);
                        std::fill(data.begin(),data.end(),-1);
                        postScratch.fill(0xc9); // Put retains GM, never the UB scratch.
                        network.Wait(Model::pending[0].id); // Progress without quiet.
                        assert(Model::quiets==iteration*2U);
                    }
                    exchange.PublishCommit({commitScratch.data()});
                    if(cpuCore==0U) {
                        assert(Model::puts==iteration*2U+2U && Model::quiets==iteration*2U+1U);
                        assert(Model::pending.size()==1U);
                        commitScratch.fill(0xd7);
                    }
                    if(consumes) {
                        // Delayed readers exercise the inbox overwrite check.
                        if(cpuCore==7U || cpuCore==15U) std::this_thread::sleep_for(side?8ms:2ms);
                        exchange.ReadPeer({data.data()});
                        for(unsigned i=0;i<routes;++i) assert(data[i]==value(peer,iteration,i));
                        network.RetireRead(side,cpuCore,iteration+1U,bank);
                        if(cpuCore==7U || cpuCore==15U) std::this_thread::sleep_for(2ms);
                    }
                    exchange.FinishSend({finishScratch.data()});
                }
                assert(Model::pending.empty());
                postScratch.fill(0);
                // Only local kernel completion, with no distributed join in
                // dispatch-only runs. The next peer commit supplies progress.
                localEnd[side].Wait();
                if(interleaveCombine && iteration%3U==1U) {
                    // Existing NDR Host combine_reuse retires both local
                    // dispatches before HCCL, then combine rotates the magic.
                    combineReuse.Wait();
                    magic^=1U;
                }
            }
            assert(Model::gets==0U);
            assert(Model::puts==(cpuCore==0U?rounds*2U:0U));
            assert(Model::quiets==(cpuCore==0U?rounds*2U:0U));
        });
    }
    for(auto& worker:workers) worker.join();
    std::cout<<"PASS paired route exchange: "<<(interleaveCombine?"existing combine completion boundary":"alternating dispatch banks without rank barrier")
        <<", delayed/empty readers, variable B/K/E, zero remote routes, inbox reuse checked; 2 puts / 2 quiets per rank\n";
}

static void TestFinishDoesNotJoinReaders() {
    std::vector<uint8_t> local(5U<<20),remote(5U<<20);
    Network network({{1,8,384}});
    auto finish=std::async(std::launch::async,[&] {
        core=0;cores=testCores;network.Attach(1,local.data(),remote.data());
        auto a=Args(1,local);
        std::vector<int32_t> data(8,11);std::array<uint8_t,256> scratch{};
        PairedRouteExchange exchange;exchange.Init(&a,8,0);
        exchange.PostBody({data.data()},{scratch.data()});
        exchange.PublishCommit({scratch.data()});
        exchange.FinishSend({scratch.data()});
        // No peer publication and no other local core has even initialized.
        assert(ReadWord(local.data()+16U*1024U+kRouteCommit)==0U);
        assert(Model::pending.empty() && Model::puts==2U && Model::quiets==2U);
        core=3U;exchange.FinishSend({}); // Empty nonowner needs no scratch.
        assert(Model::puts==2U && Model::quiets==2U);
    });
    assert(finish.wait_for(1s)==std::future_status::ready);finish.get();
    std::cout<<"PASS FinishSend retires outbound commit without peer or local-reader join\n";
}

static void TestBodyFailure() {
    std::vector<uint8_t> local(5U<<20),remote(5U<<20);
    Network network({{1,8,384}},true);core=0;cores=testCores;
    network.Attach(0,local.data(),remote.data());
    auto a=Args(0,local);
    std::vector<int32_t> data(64,1);std::array<uint8_t,256> scratch{};
    PairedRouteExchange exchange;exchange.Init(&a,8,0);
    exchange.PostBody({data.data()},{scratch.data()});
    bool failed=false;
    try {exchange.PublishCommit({scratch.data()});}
    catch(const std::runtime_error& e) {failed=std::string(e.what())=="injected body CQ failure";}
    assert(failed && Model::puts==1U && Model::quiets==1U);
    assert(Model::pending.size()==1U && Model::pending[0].bytes==32U);
    assert(ReadWord(remote.data()+16U*1024U+kRouteCommit)==0U);
    Model::pending.clear();
    std::cout<<"PASS body CQ failure prevents commit\n";
}

int main() {
    assert(DispatchServerDedupCountLayout(1024,72).stateBytes < 1024U*1024U);
    TestEpochs(false);TestEpochs(true);TestFinishDoesNotJoinReaders();TestBodyFailure();
}
