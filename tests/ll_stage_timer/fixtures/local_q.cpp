#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstdint>
#include <vector>
#include <random>
#include <algorithm>
#include "csrc/kernels/dispatch/low_latency/common/ll_local_q.h"
#include "csrc/kernels/dispatch/low_latency/device/dispatch_frequency_histogram.h"
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
  std::vector<uint32_t> ids((b*k+63)/64*64,7),ref(b*k),hist(e);
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
   if(part.count) {
    DispatchDedup::CountFrequency<true>(counts.data(),ids.data(),part.begin,0,e);
    DispatchDedup::LlFinishLocalQ(Tensor{&ids},Tensor{&counts},Tensor{&q},k,part);
   }
   for(unsigned i=0;i<part.count;++i)assert(q[i]==ref[part.begin+i]);
   // Reordered DIRECT and FIRST_HOP tail reads must still use original q.
   for(unsigned i=part.count;i>0;--i)assert(q[i-1]==ref[part.begin+i-1]);
   if(!part.count)for(auto value:counts)assert(value==0xdeadbeef);
  }
  assert(coverage==b*k);
 }
 for(unsigned begin:{0U,63U,64U,65U,32768U,32769U,65536U}) {
  std::vector<uint32_t> ids((begin+80+63)/64*64,383U),counts(384,0xdeadbeef),q(80);
  if(begin<=32768)
   DispatchDedup::CountFrequency<true>(counts.data(),ids.data(),begin,0,384);
  else
   DispatchDedup::CountFrequency<false>(counts.data(),ids.data(),begin,0,384);
  DispatchDedup::LlFinishLocalQ(Tensor{&ids},Tensor{&counts},Tensor{&q},8,{begin,80});
  for(unsigned j=0;j<80;++j)assert(q[j]==(begin+j<8?0:begin+j));
 }

}
