#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstdint>
#include <vector>
#include <random>
#include <algorithm>
#include "csrc/kernels/dispatch/low_latency/common/ll_local_q.h"
struct Tensor {
 std::vector<uint32_t>* v;
 uint32_t GetValue(uint32_t i){return v->at(i);}
 void SetValue(uint32_t i,uint32_t x){v->at(i)=x;}
};
int main(){
 std::mt19937 rng(6217);
 for(unsigned b:{1U,3U,8U,32U,48U,256U})
 for(unsigned k:{1U,3U,8U,32U})
 for(unsigned e:{32U,384U,1024U})
 for(unsigned s:{1U,10U,28U,55U})
 for(bool dedup:{false,true}) for(bool repeated:{false,true}) {
  std::vector<uint32_t> ids(b*k),ref(b*k),hist(e);
  for(unsigned t=0;t<b;++t) {
   std::vector<unsigned> pool(e);
   for(unsigned i=0;i<e;++i)pool[i]=i;
   std::shuffle(pool.begin(),pool.end(),rng);
   for(unsigned j=0;j<k;++j)ids[t*k+j]=repeated?rng()%8:pool[j];
  }
  for(unsigned i=0;i<b*k;++i) {ref[i]=i<k?0:hist[ids[i]];++hist[ids[i]];}
  unsigned coverage=0;
  for(unsigned c=0;c<s;++c) {
   auto part=DispatchDedup::LlLocalQShard(b,k,s,c,dedup);
   assert(part.begin==coverage);
   coverage+=part.count;
   std::vector<uint32_t> q(part.count),counts(e,0xdeadbeef);
   DispatchDedup::LlBuildLocalQ(Tensor{&ids},Tensor{&counts},Tensor{&q},e,k,part);
   for(unsigned i=0;i<part.count;++i)assert(q[i]==ref[part.begin+i]);
   // Reordered DIRECT and FIRST_HOP tail reads must still use original q.
   for(unsigned i=part.count;i>0;--i)assert(q[i-1]==ref[part.begin+i-1]);
   if(!part.count)for(auto value:counts)assert(value==0xdeadbeef);
  }
  assert(coverage==b*k);
 }
}
