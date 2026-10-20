"""All-core dispatch schedule, deferred q and source metadata semantic models."""
from collections import Counter, defaultdict
import importlib.util
from pathlib import Path
import random
import pytest
ROOT=Path(__file__).resolve().parents[2]

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

@pytest.mark.parametrize('tokens',[1,8,32,48,256])
@pytest.mark.parametrize('group',[1,4,16])
def test_all_core_routes_and_deferred_q_preserve_outputs(tokens,group):
 ranks=16;experts=384;k=8;rank=5;workers=56;rng=random.Random(81+tokens)
 routes=[rng.sample(range(experts),k) for _ in range(tokens)]
 flat=[e for row in routes for e in row];qref=[];hist=Counter()
 for i,e in enumerate(flat):qref.append(0 if i<k else hist[e]);hist[e]+=1
 cells={};masks={};direct_total=0
 for core in range(workers):
  base,rem=divmod(tokens,workers);start=core*base+min(core,rem);count=base+(core<rem)
  direct=[]
  for token in range(start,start+count):
   grouped=defaultdict(list)
   for slot,e in enumerate(routes[token]):grouped[e//(experts//ranks*group)].append(slot)
   old_mask=0;old_weights={};source_mask=0;new_weights={i:float(i+1) for i in range(k)}
   for dest,slots in grouped.items():
    firsthop=dest!=rank//group and len(slots)>1
    if firsthop:
     first=slots[0];old_mask|=1<<first;old_weights[first]=1.0
     source_mask|=1<<first;new_weights[first]=1.0
     for slot in slots:
      e=routes[token][slot];q=qref[token*k+slot]
      assert (e,q) not in cells;cells[e,q]=(token,slot)
    else:
     for slot in slots:
      old_mask|=1<<slot;old_weights[slot]=float(slot+1)
      source_mask|=1<<slot
      # New cache carries no q. It is derived only when issued below.
      direct.append((token-start,slot,routes[token][slot]))
   assert source_mask==old_mask
   assert {slot:new_weights[slot] for slot in old_weights}==old_weights
   assert token not in masks;masks[token]=source_mask
  # Arbitrary issuance order cannot alter immutable original route positions.
  rng.shuffle(direct);direct_total+=len(direct)
  for local,slot,e in direct:
   token=start+local;index=token*k+slot
   q=sum(v==e for v in flat[:index]) if index>=k else 0
   assert q==qref[index]
   assert (e,q) not in cells;cells[e,q]=(token,slot)
 assert len(masks)==tokens and len(cells)==tokens*k
 assert set(cells.values())=={(token,slot) for token in range(tokens) for slot in range(k)}



def test_schema8_all_workers_and_schema7_backward_decoding():
 fixture=load('all_segments','tests/ll_stage_timer/test_segments.py')
 timer=fixture.timer;data=fixture.raw(send=10)
 report=timer.parse_records(data,'dispatch')
 assert report['schema']==8 and report['dispatch_send_cores']==56 and report['dispatch_prefix_cores']==56
 assert {row['role'] for row in report['cores']}=={'send_and_recv'}
 for core,rows in enumerate(data):
  rows[0][1]=7;rows[0][5]=1|(10<<8)|((3 if core<10 else 2)<<16)
 legacy=timer.parse_records(data,'dispatch')
 assert legacy['schema']==7 and legacy['dispatch_send_cores']==10
 assert {row['role'] for row in legacy['cores']}=={'send_and_recv','cumsum'}
 assert timer.reference_binary(data,'dispatch')


def test_all_core_control_storage_fits_without_changing_combine_boundary():
 for experts in (16,384,1024):
  matrix=56*56*32;stride=(experts*4+31)//32*32
  end=72*512+experts*512+2*matrix+56*stride
  state=(end+1023)//1024*1024+1024
  assert state<=1024*1024


def test_known_all_core_comparison_identity_keeps_actual_workers():
 parity=load('all_parity','tests/ll_stage_timer/test_parity.py')
 left=parity.identity();left.update(core_protocol=9010,send_cores=10,recv_cores=46,stage_timer_schema=7)
 right=dict(left,implementation='all-send-recv',core_protocol=11010,stage_timer_schema=8,
            dispatch_send_cores=56,dispatch_prefix_cores=56,dispatch_recv_cores=56)
 parity.comparison.check_identity(left,right)
 def report(identity):return dict(comparison=identity,per_rank_core_stage=[dict(operation='dispatch',rank=0,core=0,stage='SEND_TOKENS',unit='us',mean=10)])
 row=parity.comparison.compare(report(left),report(right),'stage')[0]
 assert row['status']=='work_partition_changed' and row['candidate_over_baseline'] is None


def test_actual_kernel_phase_boundary_with_empty_senders(tmp_path):
 import shutil,subprocess
 compiler=shutil.which('g++')
 if compiler is None:pytest.skip('C++ compiler unavailable; actual orchestration mock requires target CPU compiler')
 source=(ROOT/'csrc/kernels/dispatch/low_latency/dispatch_server_dedup_kernel.h').read_text(encoding='utf-8')
 start=source.index('__aicore__ inline void Process()');brace=source.index('{',start);depth=1;end=brace+1
 while depth:
  depth+=(source[end]=='{')-(source[end]=='}');end+=1
 method=source[start:end]
 prefix=r"""
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>
#define __aicore__
#define LL_SEG_SCOPE(name) ((void)0)
constexpr int PIPE_ALL=0;
template<int>void PipeBarrier(){}
thread_local unsigned core=0;
std::atomic<unsigned> done{0},sendCalls{0},receiveCalls{0},finishCalls{0};
std::mutex mutex_;std::condition_variable cv;unsigned arrived=0;
template<bool>void SyncAll(){std::unique_lock<std::mutex> lock(mutex_);if(++arrived==56)cv.notify_all();else cv.wait(lock,[]{return arrived==56;});}
struct Counts{
 int scratch=0;
 void SendCnt(){} int& ScratchPool(){return scratch;}unsigned ReservedBytes(){return 0;}
 void PrepareReceive(){assert(done==56 && arrived==56);}
 void RecvCnt(){}void CalcCumsum(){}void CompleteCumsum(){}void Finish(){++finishCalls;}
};
struct Sender{template<class... T>void Process(T&&...){
 ++sendCalls;
 // Most workers model empty shards; all must still reach the real barrier.
 if(core==55)std::this_thread::sleep_for(std::chrono::milliseconds(5));
 ++done;
}};
struct Receiver{void Process(int*){assert(done==56 && arrived==56);++receiveCalls;}};
struct Kernel{Counts counts_;Sender sender_;Receiver receiver_;int* timer_=nullptr;
"""
 suffix=r"""
};
int main(){std::vector<std::thread> threads;for(unsigned i=0;i<56;++i)threads.emplace_back([i]{core=i;Kernel kernel;kernel.Process();});for(auto& t:threads)t.join();assert(done==56 && sendCalls==56 && receiveCalls==56 && finishCalls==56);}
"""
 cpp=tmp_path/'orchestration.cpp';cpp.write_text(prefix+method+suffix,encoding='utf-8');exe=tmp_path/'orchestration'
 subprocess.run([compiler,'-std=c++17','-O2','-Wall','-Wextra','-Werror','-pthread',str(cpp),'-o',str(exe)],check=True)
 subprocess.run([str(exe)],check=True,timeout=30)


def test_combine_split_does_not_change_all_core_dispatch_roles():
 parity=load('all_parity_split','tests/ll_stage_timer/test_parity.py')
 left=parity.identity();left.update(implementation='all-send-recv',core_protocol=11010,send_cores=10,recv_cores=46,
                                 stage_timer_schema=8,dispatch_send_cores=56,dispatch_prefix_cores=56,dispatch_recv_cores=56)
 right=dict(left,core_protocol=11028,send_cores=28,recv_cores=28)
 def report(identity):return dict(comparison=identity,per_rank_core_stage=[dict(operation='dispatch',rank=0,core=20,stage='SEND_TOKENS',unit='us',mean=10)])
 assert parity.comparison.compare(report(left),report(right),'stage')[0]['status']=='matched'
