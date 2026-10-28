#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <vector>
#include "kernels/dispatch/low_latency/common/ll_dispatch_packet_plan.h"
#include "kernels/dispatch/low_latency/common/ll_sender_route_plan.h"
using namespace DispatchDedup;
struct Words {
    std::vector<uint32_t>* v;
    uint32_t GetValue(uint32_t i) const { return v->at(i); }
    void SetValue(uint32_t i,uint32_t x) { v->at(i)=x; }
};
int main() {
    unsigned cases=0;
    bool noninitialCoreAtB1=false;
    for(unsigned world:{32U,64U,128U,256U})for(unsigned sendCores=8;sendCores<=36;++sendCores)
    for(unsigned rank:{0U,world-1U})for(unsigned routes:{8U,8193U}) {
        std::set<unsigned> all;
        for(unsigned peer=0;peer<world;++peer)if(!LlSameHost(rank,peer)) {
            std::set<unsigned> owners;
            for(unsigned qp=0;qp<8;++qp) {
                auto core=LlDispatchQpOwner(rank,world,sendCores,peer,qp,routes);
                assert(core<sendCores);owners.insert(core);all.insert(core);
            }
            assert(owners.size()==8);
            if(world==32 && routes<=8192) {
                if(peer==(rank^16U))assert(LlDispatchQpOwner(rank,world,sendCores,peer,0,routes)==0);
            } else {
                const auto countCore=peer/(world/8);
                assert(LlDispatchQpOwner(rank,world,sendCores,peer,countCore,routes)==countCore);
            }
        }
        assert(all.size()==sendCores);
    }
    for(unsigned qp=0;qp<8;++qp)assert(LlDispatchQpOwner(0,32,28,17,qp,8)==(24+qp)%28);
    for(unsigned world:{16U,32U,64U,128U,256U}) for(unsigned b:{1U,2U,17U,256U})
    for(unsigned k:{1U,8U,16U,32U}) for(unsigned hot:{0U,1U}) {
        const unsigned e=world*2U,group=LlGroupForWorld(world),local=2U,rank=world/2U;
        if(k>e) continue;
        std::mt19937 rng(19+world+b+k+hot);
        std::vector<uint32_t> ids(b*k),pool(e);
        std::iota(pool.begin(),pool.end(),0U);
        for(unsigned token=0;token<b;++token) {
            // Hot routes repeatedly target one host; still valid unique experts.
            const unsigned poolSize=hot?std::max(k,32U):e;
            std::iota(pool.begin(),pool.end(),0U);
            std::shuffle(pool.begin(),pool.begin()+poolSize,rng);
            std::copy_n(pool.begin(),k,ids.begin()+token*k);
        }
        const bool mesh=world==32;
        std::vector<uint64_t> expectedMask(b*k,UINT64_MAX);
        std::map<unsigned,std::vector<unsigned>> remote;
        for(unsigned token=0;token<b;++token) {
            std::map<unsigned,std::vector<unsigned>> groups;
            for(unsigned j=0;j<k;++j) groups[ids[token*k+j]/local/group].push_back(j);
            for(auto& [server,slots]:groups) {
                if(server==rank/group || slots.size()==1) {
                    for(auto j:slots) expectedMask[token*k+j]=0;
                } else {
                    unsigned first=slots[0];
                    if(mesh) {
                        std::array<unsigned,4> counts{};
                        for(auto j:slots) ++counts[(ids[token*k+j]/local%16)/4];
                        for(auto j:slots)
                            if(counts[(ids[token*k+j]/local%16)/4]>counts[(ids[token*k+first]/local%16)/4]) first=j;
                    }
                    uint32_t mask=0;for(auto j:slots)mask|=1U<<j;
                    expectedMask[token*k+first]=mask;
                }
            }
        }
        for(unsigned i=0;i<b*k;++i) {
            const auto packet=LlPacketForRoute(Words{&ids},i,k,local,rank,group,true,mesh);
            assert(packet.present==(expectedMask[i]!=UINT64_MAX));
            if(packet.present) {
                assert(packet.mask==expectedMask[i]);
                const unsigned peer=ids[i]/local;
                if(!LlSameHost(rank,peer))remote[peer].push_back(i);
            }
        }
        std::vector<uint32_t> baseQ(b*k),a(b*k),z(b*k),counts(e);
        LlBuildRouteQ(Words{&ids},b*k,k,e,Words{&a},Words{&z},Words{&baseQ},b*k);
        for(unsigned i=0;i<b*k;++i) {
            assert(baseQ[i]==(i<k?0U:counts[ids[i]]));++counts[ids[i]];
        }
      for(unsigned sendCores:{8U,28U,36U}) {
        std::vector<unsigned> seen(b*k);
        for(unsigned core=0;core<sendCores;++core) {
            auto q=baseQ;std::vector<uint32_t> scratch(b*k);
            const auto owned=LlBuildRemotePacketOwners(Words{&ids},Words{&q},Words{&scratch},
                b*k,k,local,rank,group,true,mesh,core,sendCores,world);
            unsigned checked=0;
            for(const auto& [peer,packets]:remote) {
                for(unsigned ordinal=0;ordinal<packets.size();++ordinal) {
                    const auto i=packets[ordinal];
                    assert(LlRemotePacketQp(q[i])==ordinal%8);
                    const auto owner=LlDispatchQpOwner(rank,world,sendCores,peer,ordinal%8,b*k);
                    assert(LlOwnsRemotePacket(q[i],core)==(owner==core));
                    if(LlOwnsRemotePacket(q[i],core)){++seen[i];++checked;}
                }
            }
            assert(checked==owned);
            if(owned && b==1 && core>=8)noninitialCoreAtB1=true;
            for(unsigned i=0;i<b*k;++i) {
                assert((q[i]&kLlRouteQMask)==baseQ[i]);
                if(expectedMask[i]==UINT64_MAX || LlSameHost(rank,ids[i]/local))assert(!(q[i]&kLlRemotePacket));
            }
        }
        for(unsigned i=0;i<b*k;++i)assert(seen[i]==unsigned(expectedMask[i]!=UINT64_MAX && !LlSameHost(rank,ids[i]/local)));
      }
        ++cases;
    }
    assert(noninitialCoreAtB1);
    std::cout<<"PASS "<<cases<<" packet plans: unchanged dedup/first-hop/q, per-peer round robin, unique eight QP owners across the full S-core pool\n";
}
