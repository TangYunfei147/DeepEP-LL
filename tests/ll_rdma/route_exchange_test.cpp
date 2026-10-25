#include <array>
#include <atomic>
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
static uint32_t ReadWord(uint8_t* p) {
    std::lock_guard<std::mutex> guard(memoryMutex);
    uint32_t value;std::memcpy(&value,p,sizeof(value));return value;
}
static void WriteWord(uint8_t* p,uint32_t value) {
    std::lock_guard<std::mutex> guard(memoryMutex);
    std::memcpy(p,&value,sizeof(value));
}

// A separate NIC thread progresses submitted WQEs even when no core calls
// quiet. A quiet only waits/reaps this core's matching peer/QP completions.
// Delivery delays intentionally differ for body, commit and ACK, and eligible
// requests are selected in reverse order rather than assuming cross-QP order.
class Network {
    enum Kind { Body,Commit,Ack };
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
    std::mutex mutex;
    std::condition_variable cv;
    std::vector<Flight> flights;
    bool stop=false,failBody;
    unsigned coreCount;
    std::thread nic;

    void Run() {
        std::unique_lock<std::mutex> guard(mutex);
        while(!stop) {
            size_t next=flights.size();
            auto now=std::chrono::steady_clock::now();
            for(size_t i=flights.size();i--;) {
                if(!flights[i].done && flights[i].due<=now) {next=i;break;}
            }
            if(next==flights.size()) {cv.wait_for(guard,100us);continue;}
            auto& f=flights[next];
            {
                std::lock_guard<std::mutex> memoryGuard(memoryMutex);
                // WQEs retain the registered GM operand, never the UB scratch.
                assert(std::memcmp(f.request.src,f.source.data(),f.source.size())==0);
                std::memcpy(f.request.dst,f.request.src,f.request.bytes);
            }
            // Even if delivery happened, a failed body CQ must stop commit.
            if(failBody && f.kind==Body) f.error="injected body CQ failure";
            f.done=true;cv.notify_all();
        }
    }
public:
    explicit Network(unsigned cores,bool failure=false):failBody(failure),coreCount(cores),nic([this]{Run();}) {}
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
        else if(offset-bank==kRouteCommit) kind=Commit;
        else {assert(offset-bank==kRouteAck);kind=Ack;}
        auto local=localBase+side*16U*1024U;
        const unsigned epoch=ReadWord(local+bank+kRouteSequence);
        assert(epoch && request.qp==0U);
        if(kind==Commit) {
            bool completed=false;
            for(const auto& f:flights) if(f.side==side && f.bank==bank && f.epoch==epoch && f.kind==Body)
                completed=f.done && f.error.empty();
            assert(completed); // failed or unfinished body can never publish
        }
        if(kind==Ack) {
            assert(ReadWord(local+bank+kRouteCommit)==epoch);
            for(unsigned c=0;c<coreCount;++c)
                assert(ReadWord(local+bank+kRouteReadDone+c*512U)==epoch);
        }
        std::vector<uint8_t> source(request.bytes);
        {std::lock_guard<std::mutex> memoryGuard(memoryMutex);std::memcpy(source.data(),request.src,request.bytes);}
        const auto delay=kind==Body?1ms:kind==Commit?(side?4ms:2ms):(side?8ms:1ms);
        request.id=flights.size();
        flights.push_back({request,side,epoch,bank,kind,std::move(source),std::chrono::steady_clock::now()+delay});
        cv.notify_all();
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

static CommArgs Args(unsigned side,std::vector<uint8_t>& local) {
    CommArgs a{};a.rank=side*16U;a.rankSize=32;a.reserved0=16;a.peerMemBytes=4U<<20;
    a.reservedConfig[0]=LlRdma::kBackend;a.reservedConfig[1]=cores;
    a.reservedAddresses[0]=local.data();a.peerMems[a.rank]=local.data()+a.rank*1024U;
    return a;
}

static void TestEpochs() {
    constexpr unsigned rounds=12;
    const unsigned routes[rounds]={8,16,64,256,8192,8,16,1024,1,24,512,8};
    const unsigned bankIds[rounds]={0,1,1,0,1,0,0,0,1,1,0,1};
    std::vector<uint8_t> memory[2]={std::vector<uint8_t>(5U<<20),std::vector<uint8_t>(5U<<20)};
    Network network(16);
    KernelEnd end[2];
    std::vector<std::thread> workers;
    auto value=[](unsigned source,unsigned iteration,unsigned index) {
        return int32_t((source*17U+iteration*31U+index)%384U);
    };
    for(unsigned side=0;side<2;++side) for(unsigned cpuCore=0;cpuCore<16;++cpuCore) {
        workers.emplace_back([&,side,cpuCore] {
            core=cpuCore;cores=16;
            unsigned rank=side*16,peer=rank^16;
            network.Attach(side,memory[side].data(),memory[1-side].data());
            auto a=Args(side,memory[side]);
            std::vector<int32_t> data(kRouteExchangeMaxBytes/4);
            std::array<uint8_t,256> postScratch{},commitScratch{},ackScratch{};
            for(unsigned iteration=0;iteration<rounds;++iteration) {
                for(unsigned i=0;i<routes[iteration];++i) data[i]=value(rank,iteration,i);
                PairedRouteExchange exchange;
                exchange.Init(&a,routes[iteration],uint64_t(bankIds[iteration])*bankBytes);
                // Mirrors SendCnt's host expert clamp, including empty core0
                // on host1 and other empty cores that immediately join ACK.
                bool consumes=side==0?cpuCore<8:cpuCore>=8;
                if(consumes || cpuCore==0) {
                    exchange.PostBody({data.data()},{postScratch.data()});
                    if(cpuCore==0) {
                        assert(Model::puts==iteration*3U+1U && Model::quiets==iteration*2U);
                        assert(postScratch[0]==0xa5 && postScratch[127]==0xa5 && postScratch[128]==0);
                        // Reuse the original routes and Put workspace for local
                        // count work before any completion call. NIC sees GM.
                        std::fill(data.begin(),data.end(),-1);
                        postScratch.fill(0xc9);
                        network.Wait(Model::pending[0].id); // NIC progresses without quiet
                        assert(Model::quiets==iteration*2U);
                    }
                    exchange.PublishCommit({commitScratch.data()});
                    if(cpuCore==0) {
                        assert(Model::puts==iteration*3U+2U && Model::quiets==iteration*2U+1U);
                        assert(Model::pending.size()==1); // commit has no immediate quiet
                        commitScratch.fill(0xd7);
                    }
                    if(consumes) {
                        if(cpuCore==7 || cpuCore==15) std::this_thread::sleep_for(3ms);
                        exchange.ReadPeer({data.data()});
                        for(unsigned i=0;i<routes[iteration];++i) assert(data[i]==value(peer,iteration,i));
                        // Delayed reconstruction/local count-frame retirement
                        // must postpone ACK even after all NIC bodies arrive.
                        if(cpuCore==7 || cpuCore==15) std::this_thread::sleep_for(2ms);
                    }
                    exchange.Complete({ackScratch.data()});
                } else exchange.Complete({});
                assert(Model::pending.empty());
                postScratch.fill(0);
                end[side].Wait(); // actual same-stream kernel completion, not a protocol barrier
            }
            assert(Model::gets==0);
            assert(Model::puts==(cpuCore==0?rounds*3U:0U));
            assert(Model::quiets==(cpuCore==0?rounds*2U:0U));
        });
    }
    for(auto& worker:workers) worker.join();
    std::cout<<"PASS paired route exchange: autonomous NIC, 12 epochs, variable shape, same/alternating bank, delayed commit/ACK/consumers, empty cores, independent scratch; 3 puts / 2 quiets per rank\n";
}

static void TestBankReuse() {
    std::vector<uint8_t> local(5U<<20),remote(5U<<20);
    Network network(1);core=0;cores=1;
    network.Attach(0,local.data(),remote.data());
    auto a=Args(0,local);
    std::vector<int32_t> data(64,11);
    std::array<uint8_t,256> scratch{};
    auto publish=[&](PairedRouteExchange& exchange,uint64_t bank,uint32_t epoch){
        exchange.PostBody({data.data()},{scratch.data()});
        exchange.PublishCommit({scratch.data()});
        WriteWord(local.data()+bank+kRouteCommit,epoch); // peer's independent publication
        exchange.Complete({scratch.data()});
    };
    PairedRouteExchange first;first.Init(&a,8,0);publish(first,0,1);
    assert(ReadWord(local.data()+kRouteAck)==0); // incoming ACK deliberately withheld
    PairedRouteExchange alternate;alternate.Init(&a,16,bankBytes);publish(alternate,bankBytes,2);
    assert(ReadWord(local.data()+kRouteAck)==0); // other bank does not wait for bank0 ACK

    std::atomic<bool> started{false};
    auto reuse=std::async(std::launch::async,[&]{
        core=0;cores=1;network.Attach(0,local.data(),remote.data());
        PairedRouteExchange same;same.Init(&a,24,0);
        std::fill(data.begin(),data.end(),29);
        started=true;publish(same,0,3);
        assert(Model::puts==3 && Model::quiets==2);
    });
    while(!started) std::this_thread::yield();
    assert(reuse.wait_for(20ms)==std::future_status::timeout);
    assert(ReadWord(local.data()+kRouteTx)==11 && ReadWord(local.data()+kRouteSequence)==1);
    assert(ReadWord(local.data()+kRouteLastSent)==1);
    WriteWord(local.data()+kRouteAck,2); // a different bank's epoch cannot release this bank
    assert(reuse.wait_for(20ms)==std::future_status::timeout);
    WriteWord(local.data()+kRouteAck,1);
    assert(reuse.wait_for(5s)==std::future_status::ready);reuse.get();
    assert(ReadWord(local.data()+kRouteTx)==29 && ReadWord(local.data()+kRouteLastSent)==3);
    std::cout<<"PASS same-bank ACK guard and independent alternate bank\n";
}

static void TestBodyFailure() {
    std::vector<uint8_t> local(5U<<20),remote(5U<<20);
    Network network(1,true);core=0;cores=1;
    network.Attach(0,local.data(),remote.data());
    auto a=Args(0,local);
    std::vector<int32_t> data(64,1);std::array<uint8_t,256> scratch{};
    PairedRouteExchange exchange;exchange.Init(&a,8,0);
    exchange.PostBody({data.data()},{scratch.data()});
    bool failed=false;
    try {exchange.PublishCommit({scratch.data()});}
    catch(const std::runtime_error& e) {failed=std::string(e.what())=="injected body CQ failure";}
    assert(failed && Model::puts==1 && Model::quiets==1);
    assert(Model::pending.size()==1 && Model::pending[0].bytes==32);
    assert(ReadWord(local.data()+kRouteLastSent)==0);
    assert(ReadWord(remote.data()+16U*1024U+kRouteCommit)==0);
    Model::pending.clear();
    std::cout<<"PASS body CQ failure prevents commit and reuse watermark advance\n";
}

int main() {
    assert(DispatchServerDedupCountLayout(1024,72).stateBytes < 1024U*1024U);
    TestEpochs();TestBankReuse();TestBodyFailure();
}
