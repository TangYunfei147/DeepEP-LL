#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>
#include "kernels/dispatch/low_latency/common/ll_fullmesh_first_hop.h"

using namespace DispatchDedup;

static void Check(const std::vector<uint32_t>& ranks, uint32_t mask) {
    LlFullMeshFirstHop policy;
    std::array<unsigned,4> counts{};
    unsigned earliest=32, members=0;
    for(unsigned k=0;k<ranks.size();++k) if(mask & (1U<<k)) {
        policy.Add(ranks[k],k);
        ++counts[(ranks[k]%16)/4];
        earliest=std::min(earliest,k);
        ++members;
    }
    if(!mask) {assert(policy.FirstK()==0);return;}
    const unsigned selected=policy.FirstK();
    assert(selected<ranks.size() && (mask & (1U<<selected)));
    const unsigned winningGroup=(ranks[selected]%16)/4;
    const unsigned maximum=*std::max_element(counts.begin(),counts.end());
    assert(counts[winningGroup]==maximum);
    // The selected target minimizes equal-sized relay traffic over Clos. Ties
    // preserve the earliest original top-k member, rather than injecting jitter.
    for(unsigned k=0;k<ranks.size();++k) if(mask & (1U<<k)) {
        if(counts[(ranks[k]%16)/4]==maximum) assert(selected<=k);
    }
    assert(members-counts[winningGroup] <= members-counts[(ranks[earliest]%16)/4]);
    const uint32_t relay=LlRelayMemberMask(mask,selected);
    assert((relay & (1U<<selected))==0 && (relay | (1U<<selected))==mask);
    // Distinct top-k slots remain distinct even if they are experts on the same
    // card. Preserve each member's q and weight when the leader moves past it.
    std::array<unsigned,32> visits{};
    std::array<int,32> q{},weight{};
    for(unsigned k=0;k<ranks.size();++k) {q[k]=int(k*13+7);weight[k]=int(k%7)-3;}
    ++visits[selected];
    int result=weight[selected]*(q[selected]+1),expected=0;
    for(unsigned k=0;k<ranks.size();++k) {
        if(mask & (1U<<k)) expected+=weight[k]*(q[k]+1);
        if(relay & (1U<<k)) {++visits[k];result+=weight[k]*(q[k]+1);}
        assert(visits[k]==unsigned((mask & (1U<<k))!=0));
    }
    assert(result==expected);
}

int main() {
    // The first route is across Clos from the three-route majority. Selecting
    // k=1 must retain k=0 in the relay tail; the old firstK+1 scan would drop it.
    LlFullMeshFirstHop majority;
    for(unsigned k=0;k<4;++k) majority.Add(std::array<unsigned,4>{16,20,21,22}[k],k);
    assert(majority.FirstK()==1);
    Check({16,20,21,22},15U);
    Check({31,16,17,18,19},31U);
    Check({16,20},3U); // stable tie
    Check({16,16,16,16},15U); // several experts on one target card
    std::vector<uint32_t> boundary(32,31U);
    Check(boundary,UINT32_MAX);
    Check(boundary,1U<<31);
    Check(boundary,0U);
    std::mt19937 rng(251025);
    unsigned checked=0;
    for(unsigned host=0;host<2;++host)
        for(unsigned k=1;k<=32;++k)
            for(unsigned sample=0;sample<300;++sample) {
                std::vector<uint32_t> ranks(k);
                for(auto& rank:ranks) rank=host*16+rng()%16;
                const uint32_t valid=k==32?UINT32_MAX:(1U<<k)-1U;
                Check(ranks,rng()&valid);
                ++checked;
            }
    std::cout<<"PASS fullmesh first-hop: "<<checked<<" route sets, minimal Clos relays, member/q/weight conservation\n";
}
