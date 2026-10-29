#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <set>
#include <vector>
#define DEEPEP_LL_RDMA 1
#define DEEPEP_TEST_DEFERRED_MTE3 1
#include "kernels/dispatch/low_latency/common/ll_count_assignment.h"
#include "kernels/dispatch/low_latency/common/ep_memory_server_dedup_count_layout.h"
#include "kernels/dispatch/low_latency/common/ep_memory_server_dedup_layout.h"
#include "kernels/combine/low_latency/combine_layout.h"
#include "kernels/dispatch/low_latency/device/ll_count_frame_receive.h"
#include "kernels/dispatch/low_latency/device/ll_count_frame_transport.h"
using namespace DispatchDedup;
using namespace AscendC;
static constexpr uint32_t Ready = 0x3f800000U;
static uint32_t Load(const uint8_t* p) { uint32_t v; std::memcpy(&v,p,4); return v; }
static void Store(uint8_t* p,uint32_t v) { std::memcpy(p,&v,4); }
static uint32_t Value(uint32_t source,uint32_t expert,uint32_t iteration) {
    return iteration==0U ? 0U : (source+expert)%3U ? 0U : (source*13U+expert*7U+iteration)%8193U;
}

int main() {
    unsigned assignments=0, snapshots=0, exchanges=0;
    for(unsigned world: {32U,64U,128U,256U}) for(unsigned experts=world;experts<=1024;experts+=world)
      for(unsigned cores=16;cores<=72;cores+=2) for(bool all: {false,true})
        for(unsigned rank=0;rank<world;++rank) {
          std::vector<unsigned> owners(experts),remoteOwners(world);
          const unsigned countCores=all?cores:cores/2;
          for(unsigned core=0;core<cores;++core) {
            auto a=CountAssignment(rank,world,experts,cores,core,all,true);
            assert(a.first<=a.end && a.end<=experts);
            assert(a.alignedRecords%8==0 && a.alignedRecords>=a.end-a.first);
            for(unsigned e=a.first;e<a.end;++e) {
              assert(e/(16*(experts/world))==rank/16); ++owners[e];
              auto f=CountFrame(experts,world,countCores,e/(experts/world),rank,e%(experts/world),true);
              assert(f.firstExpert==std::max(a.first,e/(experts/world)*(experts/world))%(experts/world));
            }
            assert(a.remoteRank==CountRemotePeer(rank,world,core));
            for(unsigned task=core;task<world-16;task+=cores) {
              const auto peer=CountRemotePeer(rank,world,task);
              assert(peer<world && peer/16!=rank/16 && task%cores==core);
              assert(a.remoteRecords>=experts/world+1); ++remoteOwners[peer];
              for(unsigned e=peer*(experts/world);e<(peer+1)*(experts/world);++e) ++owners[e];
            }
            for(unsigned routes: {1U,8U,256U,8192U,12000U}) {
              const uint64_t routeBytes=(routes*4+255U)/256U*256U;
              const uint64_t actual=3*routeBytes+std::max(routeBytes,uint64_t(a.remoteRecords)*32U)+uint64_t(a.alignedRecords)*36U;
              assert(actual<=DispatchServerDedupCountSendUbBytes(routes,experts,countCores,experts/world+1));
            }
            const unsigned base=experts/countCores,extra=experts%countCores;
            const unsigned localWorker=all?core:core-cores/2;
            unsigned oldLocalCount=0;
            if(all || core>=cores/2) {
              const unsigned begin=localWorker*base+std::min(localWorker,extra);
              const unsigned finish=begin+base+(localWorker<extra);
              const unsigned low=std::max(begin,rank/16*(16*(experts/world)));
              const unsigned high=std::min(finish,(rank/16+1)*(16*(experts/world)));
              oldLocalCount=high>low?high-low:0;
            }
            const unsigned oldReserved=oldLocalCount?((experts+countCores-1)/countCores+7)/8*8:0;
            assert(a.alignedRecords==oldReserved); // payload lanes cannot shrink
          }
          for(auto n:owners) assert(n==1); // every source/expert, including zero
          for(unsigned dst=0;dst<world;++dst) assert(remoteOwners[dst]==unsigned(dst/16!=rank/16));
          ++assignments;
        }
    // MTE fallback retains the original expert split exactly.
    for(unsigned world: {16U,32U,64U,128U}) for(unsigned cores: {16U,56U,72U})
      for(bool all: {false,true}) {
        std::vector<unsigned> seen(384);
        for(unsigned core=0;core<cores;++core) {
          auto a=CountAssignment(0,world,384,cores,core,all,false);
          assert(a.remoteRank==world);
          for(unsigned e=a.first;e<a.end;++e) ++seen[e];
        }
        for(auto n:seen) assert(n==1);
      }

    // Small route scratch grows to fit the rank frame; large route scratch is
    // already big enough. Neither changes the persistent payload reservation.
    EpMemoryServerDedupLayout layout;
    assert(DispatchServerDedupCountSendUbBytes(1,1024,72,33)==2624);
    assert(DispatchServerDedupCountSendUbBytes(12033,1024,72,33)==
           DispatchServerDedupCountSendUbBytes(12033,1024,72));
    assert(BuildDispatchServerDedupLayout(32,415,256,29,1024,&layout,false,72)==0);
    // Check the guide's concrete E1024/H3584/K8/capacity256 FP8+BF16 case
    // against real Host allocation/UB rules at the target 56-core setting.
    for(unsigned world: {16U,32U,64U,128U,256U}) {
        assert(BuildDispatchServerDedupLayout(world,256,3584,8,1024,&layout,
                    EpServerDedupInputMode::PrequantizedFp8Packs,56)==0);
        assert(DispatchDedupPacketUbFits(layout,256,8,1024,world,56));
        CombineDedup::Layout combine;
        assert(CombineDedup::BuildLayout(world,256,3584,8,1024,combine,56));
    }
    unsigned defaultLanes=0;
    const auto align=[](uint64_t n,uint64_t a){return (n+a-1)/a*a;};
    for(unsigned coreCount: {16U,28U,56U,72U})
      for(unsigned b: {1U,2U,4U,8U,16U,24U,32U,48U,64U,96U,128U,192U,256U}) {
        assert(BuildDispatchServerDedupLayout(32,b,3584,8,384,&layout,EpServerDedupInputMode::PrequantizedFp8Packs,coreCount)==0);
        assert(DispatchDedupPacketUbFits(layout,b,8,384,32,coreCount));
        for(unsigned rank: {0U,16U}) for(unsigned worker=0;worker<coreCount/2;++worker) {
          auto a=CountAssignment(rank,32,384,coreCount,worker,true,true);
          const unsigned tokens=b/(coreCount/2)+(worker<b%(coreCount/2));
          if(!tokens)continue;
          const uint64_t q=align(b*8*4,32);
          const uint64_t used=a.alignedRecords*32+2*layout.data.packedPayloadBytes+
              align(b*8*4,256)+3*q+2*align(tokens*8*4,32)+align(tokens*4,32)+
              align(tokens*8*16,32)+64+32;
          assert((kEpServerDedupCountUbBytes-used)/layout.data.tokenStride>=8);
          ++defaultLanes;
        }
      }

    std::vector<uint8_t> local(1<<20),remote(1<<20),stage(72*LlRdma::kCoreBytes),scratch(256),records(8192);
    Model::local=local.data(); Model::remote=remote.data();
    CommArgs args{}; args.rankSize=32; args.reserved0=16; args.peerMemBytes=512*1024;
    args.reservedAddresses[0]=local.data(); args.reservedAddresses[1]=stage.data();
    args.reservedConfig[0]=LlRdma::kBackend;
    for(unsigned world: {32U,64U,128U,256U}) for(unsigned experts: {32U,64U,384U,1024U})
      if(experts>=world && experts%world==0) for(unsigned coreCount: {16U,28U,56U,72U})
      for(bool all: {false,true}) for(unsigned dst: {0U,world/2-1,world/2,world-1}) {
        const unsigned localExperts=experts/world, countCores=all?coreCount:coreCount/2, receivers=coreCount/2;
        cores=coreCount; args.rankSize=world; args.reservedConfig[1]=coreCount;
        auto target=remote.data()+dst*1024;
        std::fill(remote.begin(),remote.end(),0);
        for(unsigned iteration=0;iteration<3;++iteration) {
          // Reverse source/core completion; a whole zero-count round is valid.
          for(unsigned source=world;source-->0;) {
            args.rank=source; args.peerMems[dst]=target;
            for(unsigned worker=coreCount;worker-->0;) {
              core=worker;
              const auto a=CountAssignment(source,world,experts,coreCount,worker,all,true);
              unsigned first,end;
              const bool cross=source/16!=dst/16;
              if(cross) {
                const unsigned task=dst<source/16*16?dst:dst-16;
                if(task%coreCount!=worker) continue;
                assert(CountRemotePeer(source,world,task)==dst);
                first=0; end=localExperts;
              } else {
                unsigned begin=std::max(a.first,dst*localExperts),finish=std::min(a.end,(dst+1)*localExperts);
                if(finish<=begin) continue;
                first=begin-dst*localExperts; end=finish-dst*localExperts;
              }
              auto f=CountFrame(experts,world,countCores,dst,source,first,true);
              assert(f.firstExpert==first && f.records==end-first);
              assert(f.frameOffset+f.records*32<=f.commitOffset);
              assert(f.commitOffset+32<=(uint64_t(source)+1)*localExperts*512);
              assert(Load(target+f.commitOffset)==0); // previous round cleared
              std::fill(records.begin(),records.end(),0);
              for(unsigned i=0;i<f.records;++i) {
                Store(records.data()+i*32,Ready);
                Store(records.data()+i*32+4,Value(source,dst*localExperts+first+i,iteration));
              }
              Store(records.data()+f.records*32,Ready); // dedicated SO tail
              const auto puts=Model::puts,quiets=Model::quiets;
              if(cross) {
                Model::onPost=[&](Model::Request& r) {
                  assert(deferredMte3.empty()); // registered source must be ready
                  assert(r.qp==worker && r.peer==dst && r.bytes==(f.records+1)*32);
                  assert(r.dst==target+f.frameOffset);
                  assert(std::memcmp(r.src,records.data(),r.bytes)==0);
                };
                Model::before=[&](const Model::Request& r) {
                  // Receiver cannot acquire the frame before its tail, even if
                  // some body records are already visible. Each WQE is ordered.
                  for(unsigned i=0;i<r.bytes-32;i+=32) {
                    std::memcpy(r.dst+i,r.src+i,32);
                    assert(Load(target+f.commitOffset)==0);
                  }
                };
                LlTransport::DirectCountFrame(&args,dst,f.frameOffset,{records.data()},f.records,{scratch.data()});
                assert(Model::puts==puts+1 && Model::quiets==quiets+1 && Model::pending.empty());
                Model::onPost={}; Model::before={};
              } else {
                LlTransport::CountFrame(&args,dst,f.frameOffset,f.commitOffset,{records.data()},f.records,{scratch.data()});
                assert(Model::puts==puts && Model::quiets==quiets);
                assert(Load(target+f.commitOffset)==0); // local commit DMA remains in flight
                LlTransport::Sync<HardEvent::MTE3_S>(7);
              }
              assert(Load(target+f.commitOffset)==Ready);
              std::fill(records.begin(),records.end(),0xa5); // source may now be reused
              ++exchanges;
            }
          }
          unsigned at=0; uint32_t prefix=0,expectedPrefix=0;
          std::set<uint64_t> clearBody,clearCommit;
          for(unsigned receiver=0;receiver<receivers;++receiver) {
            const unsigned count=experts/receivers+(receiver<experts%receivers);
            LlCountReadPlan plan;
            assert(BuildCountReadPlan(experts,world,countCores,dst,at,count,plan,true));
            assert(plan.size<=9);
            std::vector<uint32_t> snapshot(count*8+8,0xdeadbeef);
            ReadCountFrames({reinterpret_cast<float*>(snapshot.data())},{reinterpret_cast<float*>(target)},plan,false);
            for(unsigned i=0;i<count;++i) assert(snapshot[i*8]==Ready);
            ReadCountFrames({reinterpret_cast<float*>(snapshot.data())},{reinterpret_cast<float*>(target)},plan,true);
            for(unsigned i=0;i<count;++i) {
              const unsigned logical=at+i,source=logical%world,expert=logical/world;
              const auto f=CountFrame(experts,world,countCores,dst,source,expert,true);
              assert(snapshot[i*8]==Ready);
              const auto expected=Value(source,dst*localExperts+expert,iteration);
              assert(snapshot[i*8+1]==expected);
              prefix+=snapshot[i*8+1]; expectedPrefix+=expected; assert(prefix==expectedPrefix);
              assert(clearBody.insert(f.recordOffset).second);
              if(expert==f.firstExpert) assert(clearCommit.insert(f.commitOffset).second);
            }
            for(unsigned i=count*8;i<snapshot.size();++i) assert(snapshot[i]==0xdeadbeef);
            at+=count; ++snapshots;
          }
          assert(at==experts);
          // The existing cumsum join ensures EVERY snapshot has finished before
          // any shared commit is cleared. Owner of first logical record clears.
          for(auto offset:clearBody) { assert(!clearCommit.count(offset)); std::memset(target+offset,0,32); }
          for(auto offset:clearCommit) std::memset(target+offset,0,32);
          for(unsigned i=0;i<experts*512;++i) assert(target[i]==0);
        }
      }
    std::cout<<"PASS direct counts: "<<assignments<<" assignments, "<<exchanges<<" publications, "
             <<snapshots<<" snapshots, "<<defaultLanes<<" default-shape sender lane checks; zero counts, mixed frame layouts, delayed MTE3, SO tail, exact prefixes, reuse\n";
}
