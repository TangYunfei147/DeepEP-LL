"""Dynamic dispatch split and control-layout CPU models; no NPU execution."""
from pathlib import Path
import pytest
from tests.ll_rdma.test_combine_overhead import block, run_cpp
from tests.ll_reference_profile.test_parse import sample, timer, setword
from tests.ll_rdma.report_tables import wide_records
ROOT=Path(__file__).resolve().parents[2]


def test_policy_partition_and_target_shape_admission(tmp_path):
    run_cpp(tmp_path,r"""
#include <cassert>
#include <vector>
#include <initializer_list>
#include "kernels/dispatch/low_latency/common/ep_memory_server_dedup_layout.h"
using namespace DispatchDedup;
int main(){
 for(unsigned cores=16;cores<=72;cores+=2)for(unsigned b=1;b<=cores+4;++b){
  auto s=LlDispatchSendCores(cores,b),r=LlDispatchRecvCores(cores,b);
  assert(s+r==cores && s>=1 && r>=1);
  assert(s==(b<cores-1?b:cores/2));
  std::vector<unsigned> tokens(b),experts(384),rows(b*8);
  for(unsigned c=0;c<s;++c){auto n=b/s+(c<b%s),start=c*(b/s)+(c<b%s?c:b%s);
   if(b<cores-1)assert(n==1 && start==c);
   for(unsigned t=start;t<start+n;++t)++tokens[t];
  }
  for(unsigned c=0;c<r;++c){
   auto n=384/r+(c<384%r),start=c*(384/r)+(c<384%r?c:384%r);
   for(unsigned e=start;e<start+n;++e)++experts[e];
   for(unsigned row=c*(b*8)/r;row<(c+1)*(b*8)/r;++row)++rows[row];
  }
  for(auto n:tokens)assert(n==1);for(auto n:experts)assert(n==1);for(auto n:rows)assert(n==1);
  for(auto mode:{EpServerDedupInputMode::Plain16,EpServerDedupInputMode::PrequantizedFp8Packs}){
   EpMemoryServerDedupLayout layout;
   assert(BuildDispatchServerDedupLayout(32,b,3584,8,384,&layout,mode,cores)==DISPATCH_DEDUP_SUCCESS);
   assert(DispatchDedupPacketUbFits(layout,b,8,384,32,cores));
  }
 }
 assert(LlDispatchSendCores(56,1)==1 && LlDispatchRecvCores(56,1)==55);
 assert(LlDispatchSendCores(56,54)==54 && LlDispatchRecvCores(56,54)==2);
 assert(LlDispatchSendCores(56,55)==28 && LlDispatchRecvCores(56,55)==28);
}
""",'dynamic-core-split')


def test_changing_b_never_aliases_count_body_prefix_or_old_ready(tmp_path):
    run_cpp(tmp_path,r"""
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <set>
#include <vector>
#include "kernels/dispatch/low_latency/common/ep_memory_server_dedup_count_layout.h"
#include "kernels/dispatch/low_latency/common/ll_count_frame_layout.h"
using namespace DispatchDedup;
int main(){
 for(unsigned cores:{16U,56U,72U})for(unsigned e:{32U,384U,1024U}){
  const auto l=DispatchServerDedupCountLayout(e,cores);
  assert(l.stateBytes<(1U<<20)); // exclude combine at 1MiB, route exchange at 1.25MiB
  const unsigned cap=cores-1;const uint64_t matrix=uint64_t(cap)*cap*32;
  assert(l.cumsumOffset>=uint64_t(e)*544 && l.prefixOffset>=l.cumsumOffset+matrix);
  assert(l.cumsumFlagOffset>=l.prefixOffset+l.prefixStride);
  std::vector<std::vector<uint32_t>> banks(2,std::vector<uint32_t>(l.stateBytes/4));
  unsigned round=0;
  for(unsigned b:{1U,8U,2U,54U,55U,32U,1U,56U,4U,48U}){
   auto& gm=banks[round++%2];const unsigned r=LlDispatchRecvCores(cores,b),active=std::min(e,r);
   const unsigned producers=(!LlSmallBsCoreSplit(cores,b)&&b<512)?cores:r;
   // Adversarial old body has ready-looking words at every possible byte slot.
   std::fill(gm.begin()+l.cntOffset/4,gm.begin()+(l.cntOffset+e*512U)/4,0x3f800000);
   std::set<uint64_t> commits;
   for(unsigned src=0;src<32;++src)for(unsigned expert=0;expert<e/32;++expert){
    auto f=CountFrame(e,32,producers,0,src,expert);
    assert(f.commitOffset>=e*512U && f.commitOffset+32<=l.cumsumOffset);
    commits.insert(f.commitOffset);
   }
   for(auto address:commits){assert(gm[(l.cntOffset+address)/4]==0);gm[(l.cntOffset+address)/4]=1;}
   for(auto address:commits)gm[(l.cntOffset+address)/4]=0;
   // Reverse producer completion; each active consumer acquires all sums,
   // then clears every complete 32B cell it consumed.
   for(unsigned p=active;p-->0;)for(unsigned c=0;c<active;++c){
    auto index=(l.cntOffset+l.cumsumOffset+(uint64_t(p)*r+c)*32)/4;
    assert(gm[index+1]==0);gm[index]=p+1;gm[index+1]=1;
   }
   for(unsigned c=0;c<active;++c)for(unsigned p=0;p<active;++p){
    auto index=(l.cntOffset+l.cumsumOffset+(uint64_t(p)*r+c)*32)/4;
    assert(gm[index]==p+1 && gm[index+1]==1);std::fill_n(&gm[index],8,0);
   }
   // Shared prefix remains populated across B changes, never in a flag matrix.
   std::fill_n(&gm[(l.cntOffset+l.prefixOffset)/4],e,0x3f800000);
   for(unsigned p=0;p<active;++p)for(unsigned c=0;c<r;++c){
    auto index=(l.cntOffset+l.cumsumFlagOffset+(uint64_t(c)*r+p)*32)/4;
    for(unsigned w=0;w<8;++w){assert(gm[index+w]==0);gm[index+w]=1;}
   }
   for(unsigned c=0;c<r;++c)for(unsigned p=0;p<active;++p){
    auto index=(l.cntOffset+l.cumsumFlagOffset+(uint64_t(c)*r+p)*32)/4;
    for(unsigned w=0;w<8;++w){assert(gm[index+w]==1);gm[index+w]=0;}
   }
  }
 }
}
""",'split-window-reuse')


@pytest.mark.parametrize('cores,b,send',[(16,1,1),(56,1,1),(56,8,8),(56,32,32),(56,54,54),(56,55,28),(72,1,1),(72,70,70)])
def test_profile_and_wide_tables_use_actual_dispatch_roles(cores,b,send):
    saved=sample(cores=cores,bs=b)
    parsed=timer.parse_records(saved['raw'])
    assert parsed['send_cores']==send and parsed['recv_cores']==cores-send
    assert [c['role'] for c in parsed['cores']]==['send']*send+['receive']*(cores-send)
    _,_,rows,_,_=wide_records([saved],timer,1e9)
    assert [r[5] for r in rows]==['send']*send+['receive']*(cores-send)
    setword(saved['raw'],0,62,send+1)
    with pytest.raises(ValueError,match='split'):timer.parse_records(saved['raw'])


def test_production_uses_shared_policy_for_all_roles_and_host_budget():
    root=ROOT/'csrc/kernels/dispatch/low_latency'
    files={name:(root/name).read_text(encoding='utf-8') for name in (
        'dispatch_server_dedup_kernel.h','device/dispatch_server_dedup_send.h','device/dispatch_server_dedup_recv.h',
        'device/dispatch_server_dedup_counts.h','common/ep_memory_server_dedup_layout.h')}
    for name in ('dispatch_server_dedup_kernel.h','device/dispatch_server_dedup_send.h'):
        assert 'LlDispatchSendCores(' in files[name]
    assert 'GetBlockIdx() - LlDispatchSendCores(coreNum_, bs)' in files['device/dispatch_server_dedup_recv.h']
    assert 'useAllCoreCount_ = !LlSmallBsCoreSplit(coreNum_, bs)' in files['device/dispatch_server_dedup_counts.h']
    assert 'const uint64_t sendCores = LlDispatchSendCores' in files['common/ep_memory_server_dedup_layout.h']
    # Core0's original route publisher remains alive even when it owns no count range.
    assert 'count == 0U && pairedRoutes_ && aivId_ != 0U' in files['device/dispatch_server_dedup_counts.h']
