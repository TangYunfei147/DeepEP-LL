"""Shared-plan ordering, q, count, memory and relay protocol invariants."""
from collections import Counter
import random
import shutil
import subprocess
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[2]
WORKERS=56

def split(n,c):
    return c*(n//WORKERS)+min(c,n%WORKERS),n//WORKERS+(c<n%WORKERS)

def plan(rows,experts,ranks,rank=0):
    k=len(rows[0]);width=experts//ranks*4
    hist=[];partial={};masks={};events={}
    for c in range(WORKERS):
        first,count=split(len(rows),c);counts=Counter()
        for t in range(first,first+count):
            groups={}
            for j,e in enumerate(rows[t]):
                partial[t,j]=counts[e];counts[e]+=1
                groups.setdefault(e//width,[]).append(j)
            for slots in groups.values():
                mask=sum(1<<j for j in slots)
                for j in slots:
                    masks[t,j]=mask;events[t,j]=slots[1] if len(slots)>1 else slots[0]
        hist.append(counts)
    prefix=[{} for _ in range(WORKERS)];total=Counter()
    for e in range(experts):
        running=0
        for c in range(WORKERS):prefix[c][e]=running;running+=hist[c][e]
        total[e]=running
    result={}
    for c in range(WORKERS):
        first,count=split(len(rows),c)
        for t in range(first,first+count):
            for j,e in enumerate(rows[t]):
                q=0 if t==0 else partial[t,j]+prefix[c][e]
                mask=masks[t,j];slots=[v for v in range(k) if mask&(1<<v)]
                hits=Counter(rows[t][v]//(experts//ranks) for v in slots)
                leader=slots[0] if e//width==rank//4 else min(slots,key=lambda v:(-hits[rows[t][v]//(experts//ranks)],v))
                result[t,j]=(e,q,mask,leader,events[t,j])
    return result,total


@pytest.mark.parametrize('b,k',[(1,1),(3,8),(55,3),(57,32),(256,8)])
@pytest.mark.parametrize('ranks',[16,32,64,128])
def test_parallel_q_counts_and_group_order_match_original(b,k,ranks):
    experts=1024;rng=random.Random(b*100+k+ranks)
    rows=[[rng.randrange(experts) for _ in range(k)] for t in range(b)]
    result,counts=plan(rows,experts,ranks)
    running=Counter();width=experts//ranks*4
    for t,row in enumerate(rows):
        old_groups=[];seen={}
        for j,e in enumerate(row):
            assert result[t,j][1]==(0 if t==0 else running[e]);running[e]+=1
            g=e//width
            seen.setdefault(g,[]).append(j)
            if len(seen[g])==2:old_groups.append(g)
        new_groups=[row[j]//width for j in range(k)
                    if result[t,j][2]&(result[t,j][2]-1) and result[t,j][4]==j]
        assert old_groups==new_groups
        for j,e in enumerate(row):
            mask=result[t,j][2]
            assert [v for v in range(k) if mask&(1<<v)]==seen[e//width]
    assert counts==running
    assert len(result)==b*k


def test_layout_and_majority_leader_in_actual_common_header(tmp_path):
    compiler=shutil.which('g++')
    if not compiler:pytest.skip('g++ required for actual route-plan layout/leader checks')
    source=tmp_path/'plan.cpp'
    source.write_text(r'''
#include <cassert>
#include <vector>
#include "csrc/kernels/dispatch/low_latency/common/ll_route_plan.h"
using namespace DispatchDedup;
struct Ids { const uint32_t* p; uint32_t GetValue(uint32_t i) const { return p[i]; } };
int main(){
 for(unsigned n:{1U,24U,2048U,8191U})for(unsigned e:{16U,384U,1024U}){
  auto p=LlRoutePlanMemory(n,e);
  for(unsigned h:{4U,128U,3584U}) {
   const uint64_t returnEnd=2U*1024*1024+uint64_t(n)*((uint64_t(h)*2+479)/480*512);
   assert(LlRoutePlanOffset(n, h, 1, 2U*1024*1024)>=returnEnd);
  }
  assert(p.totalsOffset>=uint64_t(e)*56*32 && p.idsOffset>=p.totalsOffset+e*32);
  assert(p.qOffset>=p.idsOffset+n*4 && p.masksOffset>=p.qOffset+n*4);
  assert(p.bytes>=p.masksOffset+n*4 && p.bytes%512==0);
 }
 uint32_t ids[8]={0,8,8,9,3,3,4,4};
 for(uint32_t m=1;m<256;++m){
  unsigned counts[kLlRanksPerGroup]={};
  for(unsigned j=0;j<8;++j)if(m&(1U<<j))++counts[ids[j]/4];
  unsigned best=LlPlanFirst(m);
  for(unsigned j=0;j<8;++j)if((m&(1U<<j)) && counts[ids[j]/4]>counts[ids[best]/4])best=j;
  assert(LlChooseLeader(Ids{ids},8,m,4)==best);
 }
 for(unsigned e:{16U,384U,1024U})assert(LlRoutePlanUbBytes(256,8,e,32)<190*1024);
}
''',encoding='utf-8')
    exe=tmp_path/'plan'
    subprocess.run([compiler,'-std=c++17','-I',str(ROOT),str(source),'-o',str(exe)],check=True,capture_output=True)
    subprocess.run([str(exe)],check=True,capture_output=True)


def test_phase_barriers_cover_empty_producers_and_prefixes():
    for b,experts in [(1,16),(3,32),(57,384)]:
        tokens=[split(b,c)[1] for c in range(WORKERS)]
        owned=[split(experts,c)[1] for c in range(WORKERS)]
        assert sum(tokens)==b and sum(owned)==experts
        # Every core, including zero-token and zero-expert owners, participates.
        source=(ROOT/'csrc/kernels/dispatch/low_latency/device/dispatch_shared_route_plan.h').read_text()
        assert source.count('SyncAll<true>()')==3
        assert 'return;' not in source


def test_receiver_and_combine_remain_separate_from_shared_plan():
    recv=(ROOT/'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_recv.h').read_text()
    assert 'route_plan' not in recv and 'PendingCount()' in recv
    layout=(ROOT/'csrc/kernels/dispatch/low_latency/common/ep_memory_server_dedup_data_layout.h').read_text()
    assert 'LlRoutePlanMemory' in layout and 'layout.routePlanOffset+' in layout


@pytest.mark.parametrize('seed',range(12))
def test_majority_heads_preserve_all_members_weights_and_reduce_remote_fanout(seed):
    rng=random.Random(seed);experts=384;ranks=16;k=8
    rows=[rng.sample(range(experts),k) for _ in range(9)]
    source_rank=rng.randrange(ranks)
    records,_=plan(rows,experts,ranks,source_rank)
    local_experts=experts//ranks;width=local_experts*4
    for t,row in enumerate(rows):
        groups={}
        for j,e in enumerate(row):groups.setdefault(e//width,[]).append(j)
        expected=sum((j+1)*(e+1)*(t+1) for j,e in enumerate(row))
        result=0
        for group,slots in groups.items():
            if group==source_rank//4:
                result+=sum((j+1)*(row[j]+1)*(t+1) for j in slots)
                continue
            leader=records[t,slots[0]][3]
            hits=Counter(row[j]//local_experts for j in slots)
            best=max(hits.values())
            assert hits[row[leader]//local_experts]==best
            assert leader==min(j for j in slots if hits[row[j]//local_experts]==best)
            members=[j for j in slots if j!=leader]
            assert sorted([leader]+members)==slots
            assert len(slots)-hits[row[leader]//local_experts] <= len(slots)-hits[row[slots[0]]//local_experts]
            # FIRST_HOP combines its own original weight and every member's
            # weight. The source applies weight 1 to this group's return slot.
            returned=(leader+1)*(row[leader]+1)*(t+1)
            returned+=sum((j+1)*(row[j]+1)*(t+1) for j in members)
            result+=returned
            for j in members:assert records[t,j][1]>=0
        assert result==expected


def test_later_head_keeps_earlier_routes_and_ties_are_stable():
    # All eight routes target group B (ranks4..7). Rank6 wins 4 vs 2/1/1,
    # although the first route targets rank4.
    rows=[[96,144,145,146,168,147,120,97]]
    records,_=plan(rows,384,16,0)
    leader=records[0,0][3]
    assert leader==1
    mask=records[0,0][2]
    members=[];remaining=mask&~(1<<leader)
    while remaining:
        j=(remaining&-remaining).bit_length()-1
        remaining&=remaining-1
        members.append(j)
    assert members==[0,2,3,4,5,6,7]
    assert len(members)==7 and 0 in members
    tied,_=plan([[96,144,145,97]],384,16,0)
    assert tied[0,0][3]==0
