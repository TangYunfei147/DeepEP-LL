#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <initializer_list>
#include "csrc/kernels/dispatch/low_latency/common/ep_memory_server_dedup_layout.h"
#include "csrc/kernels/dispatch/low_latency/common/ll_overlap_layout.h"
using namespace DispatchDedup;
int main(){
 for(unsigned e:{384U,1024U})for(unsigned b:{1U,3U,8U,32U,256U})
 for(unsigned k:{1U,3U,8U,32U})for(bool dedup:{false,true})
 for(auto mode:{EpServerDedupInputMode::Plain16,EpServerDedupInputMode::Quantize16,EpServerDedupInputMode::PrequantizedFp8Packs}){
  EpMemoryServerDedupLayout layout{};
  if(BuildDispatchServerDedupLayout(16,b,3584,k,e,&layout,mode))continue;
  for(unsigned core=0;core<56;++core){
   auto p=LlMakeOverlapLayout(layout.data,b,k,e,16,0,core,mode,dedup);
   if(p.enabled){
    assert(p.totalBytes<=kLlDispatchUbBytes);
    assert(p.sendLanes>=1 && p.sendLanes<=6);
   }
  }
 }
 auto d=DispatchServerDedupDataLayout(256,3584,8,384,EpServerDedupInputMode::PrequantizedFp8Packs);
 auto p=LlMakeOverlapLayout(d,256,8,384,16,0,0,EpServerDedupInputMode::PrequantizedFp8Packs,true);
 assert(p.enabled && p.sendLanes==6 && p.recvLanes==6 && p.totalBytes==105792);
}
