"""Source ownership, stable q and Host UB models; no device communication."""
from collections import Counter
from pathlib import Path
import random
import shutil
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[2]


def plan(experts,ranks,rank,send,core,group):
    """CPU scheduling model, cross-checked against actual C++ when available."""
    if core<send:return []
    recv=56-send; size=experts//ranks*group; first=rank//group*size; worker=core-send
    total=experts-size
    begin=worker*(total//recv)+min(worker,total%recv)
    end=begin+total//recv+(worker<total%recv)
    result=[]
    if begin<min(end,first):result.append((begin,min(end,first),0))
    if max(begin,first)<end:result.append((max(begin,first)+size,end+size,max(begin,first)-begin))
    lo=worker*(size//recv)+min(worker,size%recv)
    hi=lo+size//recv+(worker<size%recv)
    if lo<hi:result.append((first+lo,first+hi,end-begin))
    return result



@pytest.mark.parametrize('send',[1,10,28,55])
@pytest.mark.parametrize('group',[1,4,16])
def test_actual_cpp_count_ownership_and_ub_bounds(tmp_path,send,group):
    compiler=shutil.which('g++')
    if compiler is None:pytest.skip('C++ compiler unavailable; source plan/layout requires target compiler check')
    source=tmp_path/'plan.cpp'
    source.write_text(r'''
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <vector>
#include "csrc/kernels/dispatch/low_latency/common/ep_memory_server_dedup_layout.h"
using namespace DispatchDedup;
int main(){
 for(unsigned ranks: {16U,32U,64U,128U,256U}){
  for(unsigned experts: {ranks,2U*ranks,4U*ranks}){
   for(unsigned rank=0;rank<ranks;++rank){
    std::vector<unsigned> hits(experts,0);
    const unsigned groupSize=experts/ranks*kLlRanksPerGroup;
    const unsigned first=rank/kLlRanksPerGroup*groupSize;
    for(unsigned core=0;core<56;++core){
     auto p=LlMakeGroupCountPlan(experts,ranks,rank,core);
     assert(p.size<=3);
     if(core<kLlSendCoreNum)assert(p.count==0 && p.size==0 && !p.local);
     bool hasLocal=false;
     unsigned records=0;
     for(unsigned part=0;part<p.size;++part){auto v=p.ranges[part];
      assert(v.record==records && v.begin<v.end && v.end<=experts);
      for(unsigned e=v.begin;e<v.end;++e){
       assert(core>=kLlSendCoreNum);
       assert(((first<=e && e<first+groupSize))==v.local);++hits[e];++records;
       hasLocal=hasLocal||v.local;
      }
     }
     assert(records==p.count && hasLocal==p.local);
     assert(p.count<=LlGroupCountMaxRecords(experts,ranks,core>=kLlSendCoreNum));
    }
    for(auto hit:hits)assert(hit==1);
   }
  }
 }
 for(unsigned routes: {1U,8U,63U,64U,2048U,8192U}){
  const unsigned e=384,ranks=16;
  const uint64_t rb=(routes*4U+255U)/256U*256U;
  const uint64_t remote=(LlGroupCountMaxRecords(e,ranks,false)+7U)/8U*8U;
  const uint64_t local=(LlGroupCountMaxRecords(e,ranks,true)+7U)/8U*8U;
  assert(DispatchGroupLocalCountSendUbBytes(routes,e,ranks)>=4U*rb+36U*remote || !remote);
  assert(DispatchGroupLocalCountSendUbBytes(routes,e,ranks)>=6U*rb+36U*local);
 }
 EpMemoryServerDedupLayout layout;
 assert(BuildDispatchServerDedupLayout(16,256,3584,8,384,&layout,true,56)==0);
 assert(DispatchDedupPacketUbFits(layout,256,8,384,16,56));
}
''',encoding='utf-8')
    executable=tmp_path/'plan'
    subprocess.run([compiler,'-std=c++17','-O2','-Wall','-Wextra','-Werror',f'-DDEEPEP_LL_SEND_CORES={send}',
                    f'-DDEEPEP_LL_RECV_CORES={56-send}',f'-DDEEPEP_LL_RANKS_PER_GROUP={group}',
                    f'-I{ROOT}',str(source),'-o',str(executable)],check=True)
    subprocess.run([str(executable)],check=True,timeout=30)


@pytest.mark.parametrize('send',[1,10,28,55])
@pytest.mark.parametrize('group',[1,4,16])
@pytest.mark.parametrize('kind',['random','local_hotspot','remote_only'])
def test_fused_local_tasks_match_original_route_slots(send,group,kind):
    ranks=16;experts=384;rank=5;k=8;bs=19
    rng=random.Random(1234)
    first=rank//group*(experts//ranks*group);end=first+experts//ranks*group
    pool=list(range(experts))
    if kind=='local_hotspot':pool=list(range(first,first+k))
    elif kind=='remote_only':pool=[e for e in pool if not first<=e<end]
    if len(pool)<k:pool=list(range(experts))
    routes=[rng.sample(pool,k) for _ in range(bs)]
    flat=[e for row in routes for e in row]
    # Independent reference: the existing per-route prefix scan.
    reference={(i,e):sum(v==e for v in flat[:i]) if i>=k else 0 for i,e in enumerate(flat)}
    descriptors={};counts=Counter();remote_owners={}
    for core in range(56):
        for start,stop,record in plan(experts,ranks,rank,send,core,group):
            histogram=Counter()
            for i,e in enumerate(flat):
                if start<=e<stop:
                    q=histogram[e] if i>=k else 0
                    histogram[e]+=1
                    if first<=e<end:
                        assert (i,e) not in descriptors
                        descriptors[i,e]=(core,q)
                    else:remote_owners[i,e]=core
            counts.update(histogram)
    assert counts==Counter(flat)
    assert set(descriptors).isdisjoint(remote_owners)
    assert set(descriptors)|set(remote_owners)==set(reference)
    for key,(owner,q) in descriptors.items():
        assert owner>=send and first<=key[1]<end and q==reference[key]
    # Source metadata remains token-owned: local bits are recorded even though
    # those packets moved to receive cores; FIRST_HOP ownership remains first-k.
    for token,row in enumerate(routes):
        by_group={}
        for slot,e in enumerate(row):by_group.setdefault(e//(experts//ranks*group),[]).append(slot)
        original_mask=sum(1<<slots[0] if g!=rank//group and len(slots)>1 else sum(1<<v for v in slots)
                          for g,slots in by_group.items())
        local_mask=sum(1<<slot for slot,e in enumerate(row) if first<=e<end)
        remote_mask=sum(1<<slots[0] if len(slots)>1 else 1<<slots[0]
                        for g,slots in by_group.items() if g!=rank//group)
        assert original_mask==local_mask|remote_mask


def test_non_dedup_vector_filter_covers_unaligned_route_shards():
    # Compare the aligned vector-selection design to direct set subtraction.
    for total in range(1,130):
        routes=[(i*13)%384 for i in range(total)]
        for send in (1,10,28,55):
            selected=[]
            for core in range(send):
                begin=core*(total//send)+min(core,total%send)
                count=total//send+(core<total%send)
                aligned=begin//8*8
                indices=[i for i in range(aligned,begin+count) if i>=begin and not 96<=routes[i]<192]
                assert len(indices)<=count
                selected.extend(indices)
            assert selected==[i for i,e in enumerate(routes) if not 96<=e<192]
