#ifdef NDEBUG
#undef NDEBUG
#endif
#include <algorithm>
#include <cassert>
#include <random>
#include <vector>
#include "csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_receive_tasks.h"
using namespace DispatchDedup;
using namespace AscendC;
int main(){
 std::mt19937 rng(42021);
 unsigned cases=0,alignedSegments=0;
 for(unsigned ranks:{16U,32U,64U,128U}) for(unsigned local:{1U,3U,8U})
 for(unsigned bs:{1U,3U,8U,256U}) for(unsigned mode:{0U,1U,2U}) {
  const unsigned experts=ranks*local,rank=ranks-1;
  TPipe pipe;TBuf<> pb;pipe.InitBuffer(pb,DispatchDedupAlign(experts*4,32));
  auto prefix=pb.Get<int32_t>();std::vector<unsigned> expectedPrefix;
  unsigned total=0;
  for(unsigned r=0;r<experts;++r) {
   unsigned n=mode==0?0U:(mode==1?(r%7==0?bs:0U):rng()%(bs+1));
   total+=n;prefix.SetValue(r,int32_t(total));expectedPrefix.push_back(total);
  }
  unsigned covered=0;
  for(unsigned core=0;core<56;++core) {
   ServerDedupReceiveTasks task;scalarStores=vectorCalls=waits=0;
   task.Build(prefix,rank,ranks,experts,bs,core,&pipe,56);
   unsigned begin=uint64_t(core)*total/56,end=uint64_t(core+1)*total/56;
   assert(task.Begin()==begin && task.Count()==end-begin && task.PendingCount()==end-begin);
   if(begin==end) {assert(vectorCalls==0 && scalarStores==0);continue;}
   assert(vectorCalls>=1 && waits==1 && queued.empty());
   alignedSegments+=vectorCalls-1;
   assert(scalarStores<=task.Count()); // pending identity has no scalar stores
   for(unsigned j=0;j<task.Count();++j) {
    unsigned row=begin+j;
    unsigned record=std::upper_bound(expectedPrefix.begin(),expectedPrefix.end(),row)-expectedPrefix.begin();
    unsigned previous=record?expectedPrefix[record-1]:0;
    unsigned region=((record%ranks+rank)%ranks)*local+record/ranks;
    assert(task.Position(j)==region*bs+row-previous);
    assert(task.PendingIndex(j)==j);
   }
   unsigned last=task.Count()-1;task.RemovePending(0);
   assert(task.PendingCount()==last);
   if(last)assert(task.PendingIndex(0)==last);
   covered+=task.Count();++cases;
  }
  assert(covered==total);
 }
 assert(cases && alignedSegments);
}
