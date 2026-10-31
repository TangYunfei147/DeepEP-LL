#include <cassert>
#include <thread>
#include <iostream>
#include <random>
#define DEEPEP_TEST_METADATA_COPY 1
#include "kernels/dispatch/low_latency/device/dispatch_server_dedup_counts.h"
using namespace AscendC;
using namespace DispatchDedup;
struct SourceSnapshot { const void* source;std::vector<uint8_t> bytes; };
static std::vector<SourceSnapshot> countSources;
static thread_local std::vector<SourceSnapshot> taskSources;
static thread_local uintptr_t taskBegin=0,taskEnd=0;
static thread_local uint32_t taskWrites=0,taskWords=0;
static void retireTaskSources() {
    for(const auto &snapshot:taskSources)
        assert(std::memcmp(snapshot.source,snapshot.bytes.data(),snapshot.bytes.size())==0);
    taskSources.clear();
}
static void rememberTaskSource(void* dst,const void* source,uint32_t bytes) {
    const auto address=reinterpret_cast<uintptr_t>(dst);
    if(address<taskBegin || address>=taskEnd)return;
    assert(bytes && bytes<=256 && bytes%4==0 && address+bytes<=taskEnd);
    assert(address==taskBegin+taskWords*4U);
    if(taskWrites)assert(taskWords==taskWrites*64U); // Only the last write may be short.
    auto p=static_cast<const uint8_t*>(source);
    taskSources.push_back({source,std::vector<uint8_t>(p,p+bytes)});
    ++taskWrites;taskWords+=bytes/4;
}
static void rememberSource(void*,const void* source,uint32_t bytes) {
    auto p=static_cast<const uint8_t*>(source);
    countSources.push_back({source,std::vector<uint8_t>(p,p+bytes)});
}
int main(){
    constexpr uint32_t P=16,C=72,K=4;
    unsigned cases=0;
    std::vector<std::vector<uint8_t>> memories(P,std::vector<uint8_t>(2U<<20));
    for(uint32_t E:{16U,32U,1024U,16U,384U,32U})for(uint32_t B:{1U,8U,32U,65U,256U}){
        const uint32_t L=E/P,R=PayloadCores(C);
        auto layout=DispatchServerDedupCountLayout(E,C);
        std::vector<std::vector<int32_t>> ids(P,std::vector<int32_t>(B*K)),prefix(P,std::vector<int32_t>(E)),tasks(P,std::vector<int32_t>(P*B*K));
        std::vector<std::vector<int64_t>> experts(P,std::vector<int64_t>(L));
        std::vector<CommArgs> args(P);
        std::mt19937 rng(1337+E+B);
        for(uint32_t rank=0;rank<P;++rank){
            args[rank].rank=rank;args[rank].rankSize=P;args[rank].peerMemBytes=memories[rank].size();
            for(uint32_t peer=0;peer<P;++peer)args[rank].peerMems[peer]=memories[peer].data();
        }
        // Two physical banks can independently retain the same metadata layout.
        // Reuse the actual storage without clearing it between cases/rounds.
        for(uint32_t iteration=0;iteration<3;++iteration){
            for(uint32_t rank=0;rank<P;++rank)for(uint32_t row=0;row<B;++row)for(uint32_t k=0;k<K;++k)
                ids[rank][row*K+k]=iteration==1?-1:iteration==2?int32_t(k):
                    (row+k+rank)%7==0?-1:int32_t((row*K+k+rank*17)%E);
            for(auto &table:tasks)std::fill(table.begin(),table.end(),-987654);
            std::vector<TPipe> pipes(P*kLlMetadataCores);
            std::vector<ServerDedupCounts> counts(P*kLlMetadataCores);
            for(uint32_t rank=0;rank<P;++rank)for(uint32_t m=0;m<kLlMetadataCores;++m){
                cores=C;core=m;uint32_t at=rank*kLlMetadataCores+m;
                counts[at].Init(reinterpret_cast<uint8_t*>(&args[rank]),reinterpret_cast<uint8_t*>(ids[rank].data()),
                    reinterpret_cast<uint8_t*>(experts[rank].data()),reinterpret_cast<uint8_t*>(prefix[rank].data()),
                    B,K,E,1,0,&pipes[at],nullptr,reinterpret_cast<uint8_t*>(tasks[rank].data()));
                // Hold every local MTE source live until SendCnt returns.
                // Computing later round-robin targets may not overwrite it.
                countSources.clear();observeStore=rememberSource;
                counts[at].SendCnt();observeStore=nullptr;
                for(const auto &snapshot:countSources)
                    assert(std::memcmp(snapshot.source,snapshot.bytes.data(),snapshot.bytes.size())==0);
            }
            for(uint32_t rank=0;rank<P;++rank){
              std::vector<std::thread> workers;
              for(uint32_t m=0;m<kLlMetadataCores;++m)workers.emplace_back([&,rank,m]{
                cores=C;core=m;auto &counter=counts[rank*kLlMetadataCores+m];
                // Independent CPU counts determine each producer's exact output interval.
                uint32_t first=E/kLlMetadataCores*m+std::min(m,E%kLlMetadataCores);
                uint32_t last=first+E/kLlMetadataCores+(m<E%kLlMetadataCores);
                uint32_t before=0,total=0;
                for(uint32_t record=0;record<last;++record){
                    const auto &routes=ids[record%P];
                    uint32_t n=std::count(routes.begin(),routes.end(),int32_t(rank*L+record/P));
                    if(record<first)before+=n;else total+=n;
                }
                taskBegin=reinterpret_cast<uintptr_t>(tasks[rank].data()+before);
                taskEnd=taskBegin+total*4U;taskWrites=0;taskWords=0;taskSources.clear();
                observeStore=rememberTaskSource;observeMte3Completion=retireTaskSources;
                counter.RecvCnt();counter.CalcCumsum();counter.CompleteCumsum();counter.Finish();counter.RetireOwnScratch();
                retireTaskSources();observeStore=nullptr;observeMte3Completion=nullptr;
                assert(taskWords==total && taskWrites==(total+63U)/64U);
                for(auto &event:events)for(bool live:event)assert(!live);
              });
              for(auto &worker:workers)worker.join();
            }
            for(uint32_t rank=0;rank<P;++rank){
                uint32_t sum=0;
                for(uint32_t e=0;e<L;++e){uint32_t expertCount=0;for(uint32_t source=0;source<P;++source){
                    uint32_t n=std::count(ids[source].begin(),ids[source].end(),int32_t(rank*L+e));
                    for(uint32_t q=0;q<n;++q)assert(tasks[rank][sum+q]==int32_t(((source+rank)%P*L+e)*B+q));
                    sum+=n;expertCount+=n;assert(prefix[rank][e*P+source]==int32_t(sum));
                }assert(experts[rank][e]==expertCount);}
                for(uint32_t i=sum;i<tasks[rank].size();++i)assert(tasks[rank][i]==-987654);
                for(uint32_t r=0;r<R;++r){uint32_t info[8];std::memcpy(info,memories[rank].data()+layout.cntOffset+layout.outputInfoOffset+r*32,32);
                    assert(info[0]==BalancedBegin(sum,r,R) && info[1]==BalancedBegin(sum,r+1,R));
                    float flag;std::memcpy(&flag,memories[rank].data()+layout.cntOffset+layout.outputReadyOffset+r*32,4);assert(flag==1);
                }
                // Model the R-owned notification clear after acquisition.
                std::memset(memories[rank].data()+layout.cntOffset+layout.outputReadyOffset,0,R*32);
            }
            ++cases;
        }
    }
    std::cout<<"PASS metadata count exchange, prefix, balanced tasks "<<cases<<" cases\n";
}
