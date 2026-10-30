#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <stdint.h>
#include "csrc/kernels/dispatch/low_latency/device/dispatch_prefix_scan.h"
int main(){
 alignas(256) uint32_t data[1088],expected[1088];
 uint32_t state=13579;
 for(uint32_t count=0;count<=1024;++count)for(unsigned pattern=0;pattern<4;++pattern){
  uint32_t sum=0;
  for(unsigned i=0;i<1088;++i){
   state=state*1664525U+1013904223U;
   data[i]=pattern==0?0U:pattern==1?1U:pattern==2?(i==0?0x40000000U:0U):state%2049U;
   if(i>=count)data[i]=0xdeadbeefU;
   sum+=data[i];expected[i]=i<count?sum:0xdeadbeefU;
  }
  loads=stores=0;
  DispatchDedup::R0PrefixScan(data,count);
  for(unsigned i=0;i<1088;++i)assert(data[i]==expected[i]);
  assert(loads==(count+63)/64 && stores==loads);
 }
}
