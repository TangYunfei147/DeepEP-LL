#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <vector>
#include <random>
#include "csrc/kernels/dispatch/low_latency/common/ll_group_local_send.h"
#include "csrc/kernels/dispatch/low_latency/device/dispatch_frequency_histogram.h"
int main(){
    using namespace DispatchDedup;
    std::mt19937 rng(41755);
    for(uint32_t b:{1U,2U,4U,8U,16U,32U})
    for(uint32_t k:{1U,3U,8U,32U})
    for(uint32_t ranks:{16U,32U,64U})
    for(bool repeated:{false,true}) {
        constexpr uint32_t experts=384;
        std::vector<uint32_t> ids((b*k+63)/64*64,383),reference(experts);
        for(uint32_t i=0;i<b*k;++i){ids[i]=repeated?rng()%4:rng()%experts;++reference[ids[i]];}
        for(uint32_t rank:{0U,ranks/2U,ranks-1U}){
            std::vector<uint32_t> seen(experts);
            for(uint32_t core=0;core<kLlActiveCoreNum;++core){
                auto plan=LlMakeGroupCountPlan(experts,ranks,rank,core);
                for(uint32_t p=0;p<plan.size;++p){
                    auto range=plan.ranges[p];
                    std::vector<uint32_t> histogram(range.end-range.begin+64,0xdeadbeef);
                    CountFrequency<true>(histogram.data(),ids.data(),b*k,range.begin,range.end-range.begin);
                    for(uint32_t e=range.begin;e<range.end;++e){
                        assert(histogram[e-range.begin]==reference[e]);++seen[e];
                    }
                    for(uint32_t i=range.end-range.begin;i<histogram.size();++i)assert(histogram[i]==0xdeadbeef);
                }
            }
            // Every expert, including zeros, has exactly one count publisher.
            for(auto n:seen)assert(n==1);
        }
    }
}
