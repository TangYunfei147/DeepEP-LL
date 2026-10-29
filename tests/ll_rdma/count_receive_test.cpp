#include <cassert>
#include <iostream>
#include <vector>
#include "kernels/dispatch/low_latency/device/ll_count_frame_receive.h"
using namespace DispatchDedup;
using namespace AscendC;
int main() {
    unsigned cases=0;
    for(unsigned world: {32U,64U,128U,256U})
      for(unsigned experts=world;experts<=1024;experts+=world) {
        std::vector<uint32_t> gm(experts*512U/4U);
        for(unsigned i=0;i<gm.size();++i) gm[i]=i*31U+17U;
        for(unsigned cores=16;cores<=72;cores+=2)
          for(unsigned countCores: {cores,cores/2})
            for(unsigned destination: {0U,world/2,world-1}) {
              const unsigned receivers=cores/2;
              unsigned at=0,totalRuns=0;
              for(unsigned receiver=0;receiver<receivers;++receiver) {
                unsigned count=experts/receivers+(receiver<experts%receivers);
                LlCountReadPlan plan;
                assert(BuildCountReadPlan(experts,world,countCores,destination,at,count,plan));
                assert(plan.size<=5);
                totalRuns+=plan.size;
                for(bool body: {false,true}) {
                  std::vector<uint32_t> output(count*8U+64U,0xdeadbeef);
                  unsigned before=stridedCopies;
                  ReadCountFrames({reinterpret_cast<float*>(output.data())},
                                  {reinterpret_cast<float*>(gm.data())},plan,body);
                  assert(stridedCopies-before==plan.size);
                  for(unsigned i=0;i<count;++i) {
                    unsigned logical=at+i;
                    auto f=CountFrame(experts,world,countCores,destination,logical%world,logical/world);
                    unsigned offset=(body?f.recordOffset:f.commitOffset)/4U;
                    for(unsigned word=0;word<8;++word) assert(output[i*8U+word]==gm[offset+word]);
                  }
                  for(unsigned i=count*8U;i<output.size();++i) assert(output[i]==0xdeadbeef);
                }
                at+=count;++cases;
              }
              assert(at==experts && totalRuns<=experts);
              if(world==32 && experts==384 && cores==56 && countCores==56 && destination==0) {
                assert(totalRuns==38);
                std::cout<<"EP32 E384 AIV56 commit DMA per full sweep: "<<totalRuns<<" vs 384\n";
              }
            }
      }
    LlCountReadPlan plan;
    assert(!BuildCountReadPlan(384,16,56,0,0,14,plan));
    assert(!BuildCountReadPlan(384,32,0,0,0,14,plan));
    assert(!BuildCountReadPlan(384,32,56,32,0,14,plan));
    assert(!BuildCountReadPlan(384,32,56,0,380,8,plan));
    assert(!BuildCountReadPlan(1024,32,16,0,0,513,plan));
    assert(BuildCountReadPlan(1024,32,16,0,31,128,plan) && plan.size==5);
    std::cout<<"PASS "<<cases<<" receiver partitions; body/commit order and UB padding preserved\n";
}
