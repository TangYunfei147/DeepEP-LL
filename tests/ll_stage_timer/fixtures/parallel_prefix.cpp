#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <algorithm>
#include <random>
#include <vector>
#include <numeric>
#include "csrc/kernels/dispatch/low_latency/common/ll_parallel_prefix.h"
#include "csrc/kernels/dispatch/low_latency/common/ep_memory_server_dedup_layout.h"
#include "csrc/kernels/dispatch/low_latency/common/ll_overlap_layout.h"
#include "csrc/kernels/dispatch/low_latency/device/dispatch_prefix_scan.h"
using namespace DispatchDedup;
int main(){
    std::mt19937 rng(163503);
    for(unsigned receivers:{1U,2U,26U,28U,55U}) {
        const unsigned send=56-receivers;
        std::vector<uint32_t> totals(receivers),totalTags(receivers),readyTags(receivers);
        uint32_t generation=0,legacyEpoch=0;
        for(unsigned round=0;round<45;++round){
            const unsigned e=std::vector<unsigned>{1,7,16,32,384,1024}[round%6];
            if(round==20){ // Exercise wrap with the same previous tag on all participants.
                generation=UINT32_MAX;std::fill(totalTags.begin(),totalTags.end(),generation);
                std::fill(readyTags.begin(),readyTags.end(),generation);
            }
            const unsigned next=LlNextPrefixGeneration(generation);
            assert(next!=generation && next!=0);
            std::vector<uint32_t> counts(e),reference(e),output(e,0xdeadbeef),state(receivers);
            for(auto& v:counts)v=(round%3==0)?0:rng()%33;
            std::partial_sum(counts.begin(),counts.end(),reference.begin());
            std::vector<std::vector<uint32_t>> local(receivers);
            unsigned coverage=0;
            for(unsigned r=0;r<receivers;++r){
                const auto s=LlParallelPrefixSlice(e,receivers,r);assert(s.begin==coverage);coverage+=s.count;
                local[r].resize((s.count+63)/64*64,999999U);
                std::copy(counts.begin()+s.begin,counts.begin()+s.begin+s.count,local[r].begin());
                assert(LlNextPrefixGeneration(totalTags[r])==next);
                const auto totalAddress=LlPrefixTotalControlByte(send+r);
                const auto flagAddress=LlPrefixReadyControlByte(send+r);
                assert(totalAddress%32==0 && flagAddress==totalAddress+32);
                assert(totalAddress>=(send+r)*512U+64 && flagAddress+32<=(send+r+1)*512U);
            }
            assert(coverage==e);
            unsigned done=0,steps=0;
            while(done<receivers){
                assert(++steps<receivers*10000);
                const unsigned r=rng()%receivers;
                const auto s=LlParallelPrefixSlice(e,receivers,r);
                if(state[r]==0){
                    R0PrefixScan(local[r].data(),s.count);
                    totals[r]=s.count?local[r][s.count-1]:0;totalTags[r]=next;state[r]=1;
                } else if(state[r]==1 && (!s.count || std::all_of(totalTags.begin(),totalTags.begin()+r,[&](auto t){return t==next;}))){
                    uint32_t offset=0;if(s.count)for(unsigned p=0;p<r;++p)offset+=totals[p];
                    for(unsigned i=0;i<s.count;++i)output[s.begin+i]=local[r][i]+offset;
                    readyTags[r]=next;state[r]=2; // Ready only after this slice is written.
                } else if(state[r]==2 && std::all_of(readyTags.begin(),readyTags.end(),[&](auto t){return t==next;})){
                    assert(output==reference);
                    if(r==0)legacyEpoch=(round+1)%2;
                    state[r]=3;++done;
                }
            }
            generation=next;assert(legacyEpoch==(round+1)%2);
            // A dedup call does not touch +64/+96. Its old R0 epoch still toggles;
            // next small call advances its own generation instead of accepting old flags.
            legacyEpoch^=1;
            for(unsigned r=0;r<receivers;++r)assert(totalTags[r]==generation && readyTags[r]==generation);
        }
    }
    for(unsigned e:{16U,384U,1024U})for(unsigned r:{1U,26U,28U,55U}){
        const unsigned n=(e+r-1)/r;
        auto align=[](uint64_t n,uint64_t a){return (n+a-1)/a*a;};
        const uint64_t stage=32+96+align(n*32+256,256)+align(n*4,256)+
            std::max<uint64_t>(align(n*4,256),align(r*32,256));
        assert(stage==DispatchParallelPrefixStageUbBytes(e,r));
        assert(DispatchParallelPrefixCompleteUbBytes(e,16,r)==r*32+align(e*4,32)+align((e/16)*8,32));
    }
    // Existing dedup plan is numerically unchanged at the documented production case.
    auto d=DispatchServerDedupDataLayout(256,3584,8,384,EpServerDedupInputMode::PrequantizedFp8Packs);
    auto p=LlMakeOverlapLayout(d,256,8,384,16,0,0,EpServerDedupInputMode::PrequantizedFp8Packs,true);
    assert(p.enabled && p.totalBytes==105792 && p.sendLanes==6 && p.recvLanes==6);
    for(unsigned b:{1U,2U,4U,8U,16U,32U})for(unsigned c=0;c<56;++c){
        auto d=DispatchServerDedupDataLayout(b,3584,8,384,EpServerDedupInputMode::PrequantizedFp8Packs);
        auto p=LlMakeOverlapLayout(d,b,8,384,16,0,c,EpServerDedupInputMode::PrequantizedFp8Packs,false);
        assert(p.enabled && p.totalBytes<=kLlDispatchUbBytes);
        assert(p.countBytes==(c<kLlSendCoreNum?0:DispatchParallelPrefixStageUbBytes(384)));
    }
}
