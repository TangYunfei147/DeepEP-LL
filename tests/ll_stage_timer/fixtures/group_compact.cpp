#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "csrc/kernels/dispatch/low_latency/common/ll_group_compact.h"
using namespace DispatchDedup;
int main() {
 uint32_t world,experts,group,send,rank,dedup,b,k;
 while (scanf("%u%u%u%u%u%u%u%u",&world,&experts,&group,&send,&rank,&dedup,&b,&k)==8) {
  const uint32_t groups=world/group,own=rank/group,phase=rank%group;
  auto plan=LlMakeSharedGroupPlan(b,k,groups,send);
  auto* ids=(uint32_t*)calloc(b*k,4);
  auto* order=(uint32_t*)calloc(groups,4);
  auto* table=(uint32_t*)calloc(plan.gmBytes/4,4);
  auto* raw=(uint32_t*)calloc(plan.descriptorBytes/4,4);
  auto* cursors=(uint32_t*)calloc(plan.headerBytes/4,4);
  auto* masks=(uint32_t*)calloc(groups,4);
  if (!ids||!order||!table||!raw||!cursors||!masks) return 2;
  for (uint32_t i=0;i<b*k;++i) if(scanf("%u",&ids[i])!=1)return 3;
  for (uint32_t i=0;i<groups;++i) if(scanf("%u",&order[i])!=1)return 3;
  uint32_t classifiedTokens=0;
  for(uint32_t producer=0;producer<send;++producer) {
   const uint32_t base=b/send,extra=b%send;
   const uint32_t start=producer*base+(producer<extra?producer:extra);
   const uint32_t count=base+(producer<extra);
   uint32_t* header=table+producer*plan.headerBytes/4;
   uint32_t tasks=0;
   for(uint32_t token=start;token<start+count;++token) {
    ++classifiedTokens;
    for(uint32_t g=0;g<groups;++g)masks[g]=0;
    for(uint32_t slot=0;slot<k;++slot) {
     const uint32_t g=ids[token*k+slot]/(experts/world*group);
     if(g!=own)masks[g]|=uint32_t(1)<<slot;
    }
    for(uint32_t g=0;g<groups;++g) {
     if(!masks[g])continue;
     const bool firstHop=dedup && (masks[g]&(masks[g]-1U));
     for(uint32_t slot=0;slot<k;++slot) if(masks[g]&(uint32_t(1)<<slot)) {
      const uint32_t bucket=firstHop?g:groups+g;
      raw[tasks*2]=(token*k+slot)|(bucket<<16);
      raw[tasks*2+1]=firstHop?masks[g]:0;
      ++tasks;++header[bucket+1];
      if(firstHop)break;
     }
    }
   }
   uint32_t prefix=0;
   for(uint32_t bucket=0;bucket<2*groups;++bucket) {
    cursors[bucket]=prefix;prefix+=header[bucket+1];header[bucket+1]=prefix;
   }
   auto* descriptors=table+(plan.tableBytes+producer*plan.descriptorBytes)/4;
   for(uint32_t i=0;i<tasks;++i) {
    uint32_t tagged=raw[i*2],offset=cursors[tagged>>16]++;
    descriptors[offset*2]=tagged&0xffffU;descriptors[offset*2+1]=raw[i*2+1];
   }
  }
  if(classifiedTokens!=b)return 4;
  for(uint32_t core=0;core<send;++core) {
   uint32_t owner=phase%send,owned=0;
   for(uint32_t pass=dedup?0:1;pass<2;++pass)for(uint32_t step=0;step<groups;++step) {
    const uint32_t g=order[step],bucket=pass*groups+g;
    if(g==own)continue;
    for(uint32_t producer=0;producer<send;++producer) {
     auto* header=table+producer*plan.headerBytes/4;
     uint32_t start=header[bucket],n=header[bucket+1]-start;
     auto slice=LlOwnedGroupTasks(n,owner,core,send);owner=slice.nextOwner;
     auto* descriptors=table+(plan.tableBytes+producer*plan.descriptorBytes)/4;
     for(uint32_t i=0;i<slice.count;++i) {
      uint32_t pos=start+slice.first+i*send;
      printf("%u %u %u\n",core,descriptors[pos*2],descriptors[pos*2+1]);
     }
     owned+=slice.count;
    }
   }
   if(owned*32>plan.cacheBytes)return 5;
  }
  free(ids);free(order);free(table);free(raw);free(cursors);free(masks);
  printf("END\n");
 }
}
