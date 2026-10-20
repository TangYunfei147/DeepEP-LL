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

def plan(rows,experts,ranks):
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
                mask=masks[t,j];leader=(mask&-mask).bit_length()-1
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


def test_layout_and_first_leader_in_actual_common_header(tmp_path):
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
 uint32_t ids[8]={4,21,21,21,7,7,8,8};
 for(uint32_t m=1;m<256;++m)assert(LlChooseLeader(Ids{ids},8,m,4)==LlPlanFirst(m));
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
