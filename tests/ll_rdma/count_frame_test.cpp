#include <cassert>
#include <iostream>
#include <set>
#include <vector>
#define DEEPEP_LL_RDMA 1
#include "kernels/dispatch/low_latency/common/ll_count_frame_layout.h"
#include "kernels/dispatch/low_latency/device/ll_count_frame_transport.h"
using namespace AscendC;
int main() {
    // Inverse mapping must match the existing uneven expert/core partition.
    for(unsigned world: {32U,64U,128U,256U}) for(unsigned experts=world;experts<=1024;experts+=world)
      for(unsigned countCores=8;countCores<=72;++countCores) {
        unsigned local=experts/world;
        for(unsigned dst=0;dst<world;++dst) {
          std::set<uint64_t> records;
          for(unsigned src: {0U,world-1}) for(unsigned e=0;e<local;++e) {
            auto f=DispatchDedup::CountFrame(experts,world,countCores,dst,src,e);
            assert(f.records>0 && f.firstExpert<=e && e<f.firstExpert+f.records);
            assert(f.recordOffset>=f.frameOffset && f.recordOffset+32<=f.commitOffset);
            assert(f.commitOffset+32<=uint64_t(experts)*512);
            assert(records.insert(f.recordOffset).second);
            unsigned global=dst*local+e;
            unsigned owner=0,start=0,end=0;
            for(;owner<countCores;++owner) {
              unsigned n=experts/countCores+(owner<experts%countCores);
              end=start+n;if(global>=start&&global<end)break;start=end;
            }
            assert(f.firstExpert==std::max(start,dst*local)-dst*local);
            assert(f.records==std::min(end,(dst+1)*local)-std::max(start,dst*local));
          }
        }
      }
    std::vector<uint8_t> local(1<<20),remote(1<<20),stage(16*LlRdma::kCoreBytes),scratch(256),records(1024);
    Model::local=local.data();Model::remote=remote.data();
    DispatchDedup::CommArgs a;a.rank=0;a.rankSize=32;a.reserved0=16;a.peerMemBytes=512*1024;
    a.reservedAddresses[0]=local.data();a.reservedAddresses[1]=stage.data();a.reservedConfig[0]=LlRdma::kBackend;
    a.reservedConfig[1]=16;a.reservedAddresses[2]=stage.data()+16U*LlRdma::kCoreBytes;core=0;cores=16;
    unsigned totalFrames=0;
    for(unsigned dst=16;dst<32;++dst) {
      std::fill(remote.begin(),remote.end(),0);
      for(unsigned e=0;e<12;) {
        auto f=DispatchDedup::CountFrame(384,32,56,dst,0,e);
        std::fill(records.begin(),records.end(),0);
        for(unsigned i=0;i<f.records;++i) {
          uint32_t ready=0x3f800000, count=(e+i)%3; // includes zero counts
          std::memcpy(records.data()+i*32,&ready,4);
          std::memcpy(records.data()+i*32+4,&count,4);
        }
        auto target=remote.data()+dst*1024;
        Model::before=[&](const Model::Request& r) {
          if(r.dst==target+f.commitOffset)
            assert(std::memcmp(target+f.frameOffset,records.data(),f.records*32)==0);
        };
        auto puts=Model::puts,quiets=Model::quiets;
        LlTransport::CountFrame(&a,dst,f.frameOffset,f.commitOffset,{records.data()},f.records,{scratch.data()});
        assert(Model::puts-puts==2 && Model::quiets-quiets==2 && Model::pending.empty());
        for(unsigned i=0;i<f.records;++i) {
          auto r=DispatchDedup::CountFrame(384,32,56,dst,0,e+i);
          assert(std::memcmp(target+r.recordOffset,records.data()+i*32,32)==0);
        }
        e+=f.records;++totalFrames;
      }
    }
    Model::before={};
    // Each logical count has exactly one consumer, but a shared commit can
    // span several consumer cores. Preserve it until ALL snapshots complete.
    for(unsigned recvCores: {8U,16U,28U,36U}) {
      std::vector<unsigned> owner(384);
      unsigned at=0;
      for(unsigned c=0;c<recvCores;++c)
        for(unsigned j=0;j<384/recvCores+(c<384%recvCores);++j) owner[at++]=c;
      std::set<uint64_t> commits,body;
      bool shared=false;
      for(unsigned logical=0;logical<384;++logical) {
        auto frame=DispatchDedup::CountFrame(384,32,56,16,logical%32,logical/32);
        assert(body.insert(frame.recordOffset).second);
        if(logical/32==frame.firstExpert) assert(commits.insert(frame.commitOffset).second);
        if(owner[logical]!=owner[frame.firstExpert*32+logical%32]) shared=true;
      }
      assert(shared && body.size()==384);
      for(auto commit:commits) assert(!body.count(commit));
    }
    auto f=DispatchDedup::CountFrame(384,32,56,16,0,0);
    auto puts=Model::puts;
    Model::before=[](const Model::Request&) {throw std::runtime_error("CQ failure");};
    bool failed=false;
    try {LlTransport::CountFrame(&a,16,f.frameOffset,f.commitOffset,{records.data()},f.records,{scratch.data()});}
    catch(const std::runtime_error&) {failed=true;}
    assert(failed && Model::puts-puts==1); // never publish commit after body failure
    std::cout<<"PASS packed counts; EP32 E384 cores56 cross-host puts "<<2*totalFrames<<" vs 384\n";
}
