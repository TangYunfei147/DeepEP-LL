#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
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
        std::vector<unsigned> seen(b*k);
        for(unsigned core=0;core<8;++core) {
            auto q=baseQ;std::vector<uint32_t> scratch(b*k);
            const auto owned=LlBuildRemotePacketOwners(Words{&ids},Words{&q},Words{&scratch},
                b*k,k,local,rank,group,true,mesh,core);
            unsigned checked=0;
            for(const auto& [peer,packets]:remote) {
                for(unsigned ordinal=0;ordinal<packets.size();++ordinal) {
                    const auto i=packets[ordinal];
                    assert(LlOwnsRemotePacket(q[i],core)==(ordinal%8==core));
                    if(LlOwnsRemotePacket(q[i],core)){++seen[i];++checked;}
                }
            }
            assert(checked==owned);
            if(owned)assert(core<b); // Empty original S0..S7 token owners need not start.
            for(unsigned i=0;i<b*k;++i) {
                assert((q[i]&kLlRouteQMask)==baseQ[i]);
                if(expectedMask[i]==UINT64_MAX || LlSameHost(rank,ids[i]/local))assert(!(q[i]&kLlRemotePacket));
            }
        }
        for(unsigned i=0;i<b*k;++i)assert(seen[i]==unsigned(expectedMask[i]!=UINT64_MAX && !LlSameHost(rank,ids[i]/local)));
        ++cases;
    }
    std::cout<<"PASS "<<cases<<" packet plans: unchanged dedup/first-hop/q, per-peer round robin, unique eight QP owners\n";
}
