#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <vector>
#include <random>
#include "csrc/kernels/dispatch/low_latency/common/ll_small_b.h"
#include "csrc/kernels/dispatch/low_latency/common/ep_memory_server_dedup_mode.h"
struct Tensor {
    std::vector<uint32_t>* v;
    uint32_t GetValue(uint32_t i){return v->at(i);}
    void SetValue(uint32_t i,uint32_t x){v->at(i)=x;}
};
int main(){
    using namespace DispatchDedup;
    std::mt19937 rng(71249);
    for(uint32_t b:{1U,2U,4U,8U,16U,32U,33U,256U})
    for(uint32_t k:{1U,3U,8U,32U})
    for(uint32_t s:{1U,10U,26U,28U,30U,55U}) {
        for(uint32_t m:{1U,32U,33U,256U})
            assert(LlUseSmallDirect(b,EpServerDedupEnabled(m))==(b<=32&&m<=32));
        for(uint32_t c=0;c<s;++c){
            auto route=LlLocalQShard(b,k,s,c,false);
            auto token=LlLocalQShard(b,k,s,c,true);
            // Skipping an empty route owner never skips source-weight output.
            assert(!token.count||route.count);
            if(b>4)continue;
            for(bool repeated:{false,true}){
                std::vector<uint32_t> ids(b*k),q(route.count),expected(b*k),hist(384);
                for(auto& id:ids)id=repeated ? 7U : rng()%384;
                for(uint32_t i=0;i<b*k;++i){expected[i]=i<k?0U:hist[ids[i]];++hist[ids[i]];}
                LlBuildTinyDirectQ(Tensor{&ids},Tensor{&q},k,route);
                for(uint32_t j=0;j<route.count;++j)assert(q[j]==expected[route.begin+j]);
                // Reordered packet use still addresses the original route ordinal.
                for(uint32_t j=route.count;j>0;--j)assert(q[j-1]==expected[route.begin+j-1]);
                if(b==1)for(auto v:q)assert(v==0);
            }
        }
    }
}
