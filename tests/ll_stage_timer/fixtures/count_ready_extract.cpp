#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstdint>
#include <algorithm>
#include <iterator>
#include <cstdio>
#include "csrc/kernels/dispatch/low_latency/device/dispatch_count_ready_extract.h"

int main(){
 alignas(256) uint32_t input[1032*8],saved[1032*8],output[1088],result[24];
 uint32_t random=0x8491a1U;
 unsigned cases=0;
 for(unsigned count=0;count<=1024;++count){
  for(unsigned mode=0;mode<8;++mode){
   std::fill(std::begin(input),std::end(input),0xdeadbeefU);
   std::fill(std::begin(output),std::end(output),0xdeadbeefU);
   std::fill(std::begin(result),std::end(result),0xdeadbeefU);
   for(unsigned i=0;i<count;++i){
    random=random*1664525U+1013904223U;
    input[i*8]=0x3f800000U;
    // Exercise zero, subnormal bit patterns, >2^24, high bits and uint32 max.
    input[i*8+1]=mode==0?0U:mode==1?1U:mode==2?0x01000001U:mode==3?UINT32_MAX:random;
   }
   bool expected=true;
   if(count && mode>=4){
    unsigned bad=mode==4?0U:mode==5?count-1:mode==6?count/2:random%count;
    input[bad*8]=mode==4?0U:mode==5?0x40000000U:mode==6?0x3f7fffffU:0x7fc00001U;
    expected=false;
   }
   std::copy(std::begin(input),std::end(input),saved);
   CountMock::Bind(input,output,result+8,count);
   DispatchDedup::P1CountReadyAndExtract(input,output,result+8,count);
   assert((result[8]!=0)==expected);
   assert(CountMock::reads==count*2 && CountMock::outputWrites==count && CountMock::resultWrites==1);
   assert(std::equal(std::begin(input),std::end(input),saved));
   for(unsigned i=0;i<1088;++i)assert(output[i]==(i<count?input[i*8+1]:0xdeadbeefU));
   for(unsigned i=0;i<24;++i)if(i!=8)assert(result[i]==0xdeadbeefU);
   ++cases;
  }
 }
 // Every lane in a multi-block snapshot must independently veto readiness.
 for(unsigned count:{1U,7U,8U,63U,64U,65U,384U,1024U})for(unsigned bad=0;bad<count;++bad){
  for(unsigned i=0;i<count;++i){input[i*8]=0x3f800000U;input[i*8+1]=i;}
  input[bad*8]=0;
  CountMock::Bind(input,output,result+8,count);
  DispatchDedup::P1CountReadyAndExtract(input,output,result+8,count);
  assert(result[8]==0);
  ++cases;
 }
 std::printf("actual count VF: %u cases passed\n",cases);
}
