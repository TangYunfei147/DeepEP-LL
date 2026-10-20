"""Ready-only ownership, causal relay progress and the actual portable scheduler."""
import random
from pathlib import Path
import shutil
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[2]


def cas(memory,index,expected,value):
    old=memory[index]
    if old==expected:memory[index]=value
    return old


@pytest.mark.parametrize('rows',[0,1,7,16,17,63,65,511,1031,4097])
def test_actual_queue_with_host_atomics_and_delayed_relay(rows,tmp_path):
    compiler=shutil.which('g++')
    if not compiler:pytest.skip('C++ compiler unavailable; actual atomic scheduler mock requires target compiler')
    source=tmp_path/'queue.cpp'
    source.write_text(r'''
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <thread>
#include <string>
#include <vector>
#include "csrc/kernels/dispatch/low_latency/common/ll_receive_queue.h"
using namespace DispatchDedup;
struct Ops {
 using Pointer=std::atomic<uint32_t>*;
 static uint32_t Add(Pointer p,uint32_t n){return p->fetch_add(n);}
 static uint32_t Cas(Pointer p,uint32_t expected,uint32_t value){p->compare_exchange_strong(expected,value);return expected;}
};
int main(int argc,char** argv){
 const uint32_t rows=uint32_t(std::stoul(argc>1 ? argv[1] : "ROWS"));
 auto layout=LlReceiveQueueMemory(rows);
 const uint64_t bankWords=layout.bytes/4U;
 std::vector<std::atomic<uint32_t>> states(bankWords*2U);
 for(auto& word:states)word.store(0xdeadbeefU);
 std::vector<std::atomic<unsigned>> hits(rows),ready(rows);
 // Two banks and changing data must not inherit any claimed state.
 for(unsigned round=0;round<8;++round){
  const uint32_t active=((round==1 || round==5) ? 0U : round==2 ? rows/2U : round==4 ? rows/3U : rows);
  auto actual=LlReceiveQueueMemory(active);
  auto* bank=states.data()+(round&1U)*bankWords;
  for(uint64_t i=0;i<(kLlReceiveHeaderBytes+actual.claimBytes)/4U;++i)bank[i].store(0);
  for(unsigned i=0;i<rows;++i){hits[i]=0;ready[i]=(i<active && (round%2==0 || i+1==active));}
  std::vector<std::thread> workers;
  for(unsigned core=0;core<8;++core) workers.emplace_back([&]{
   LlReadyReceiveQueue<Ops> q;q.Init(bank,active,8);
   if(!q.Enter())return;
   unsigned batch=0,guard=0;
   while(q.Next(batch)){
    assert(++guard<10000000);
    unsigned begin=batch*kLlReceiveBatch,count=std::min(kLlReceiveBatch,active-begin),mask=0;
    for(unsigned i=0;i<count;++i)if(ready[begin+i].load())mask|=1U<<i;
    if(!mask){std::this_thread::yield();continue;}
    uint32_t observed=0;
    auto owned=q.Claim(batch,mask,LlReceiveValidMask(count),observed);
    for(unsigned i=0;i<count;++i)if(owned&(1U<<i)){
      assert(hits[begin+i].fetch_add(1)==0);
      if(begin+i+1==active) for(unsigned j=0;j<active;++j)ready[j].store(1);
      ready[begin+i].store(0);
    }
   }
  });
  for(auto& w:workers)w.join();
  for(unsigned j=0;j<rows;++j)assert(hits[j].load()==(j<active ? 1U : 0U));
  assert(bank[kLlReceiveDoneOffset/4U].load()==(active+15U)/16U);
 }
 // Cold scanning can advance the shared cursor beyond the queue length.
 // Admission must remain independent so the second worker can still help.
 if(rows>=32){
  for(auto& word:states)word.store(0);
  LlReadyReceiveQueue<Ops> a,b,c;
  a.Init(states.data(),32,8);b.Init(states.data(),32,8);c.Init(states.data(),32,8);
  assert(a.Enter());uint32_t batch=0;
  for(unsigned i=0;i<4;++i)assert(a.Next(batch));
  assert(states[0].load()>=2);
  assert(b.Enter());assert(!c.Enter());
  assert(b.Next(batch) && batch<2);
 }
 // Separate readiness waves in the same batch, with authoritative old values.
 if(rows>=16){
  for(auto& word:states)word.store(0);
  LlReadyReceiveQueue<Ops> a,b;a.Init(states.data(),16,8);b.Init(states.data(),16,8);
  uint32_t seen=0;
  assert(a.Claim(0,0x55U,0xffffU,seen)==0x55U);
  assert(b.Claim(0,0xffU,0xffffU,seen)==0xaaU);
  assert(a.Claim(0,0xffffU,0xffffU,seen)==0xff00U);
  assert(b.Claim(0,0xffffU,0xffffU,seen)==0U);
  assert(a.AllClaimed());
 }
}
'''.replace('ROWS',str(rows)),encoding='utf-8')
    executable=tmp_path/'queue'
    subprocess.run([compiler,'-std=c++17','-O2','-pthread','-Wall','-Wextra','-Werror',f'-I{ROOT}',str(source),'-o',str(executable)],check=True)
    subprocess.run([str(executable)],check=True,timeout=30)


@pytest.mark.parametrize('seed',range(20))
def test_stale_ready_snapshots_cannot_double_own_rows(seed):
    # Independent event-level linearization check: CAS attempts and the resulting
    # payload writes may be arbitrarily interleaved, including after all claims.
    rng=random.Random(seed);rows=65;memory=[0]*5;owners={};pending=[];completed=0
    for batch in range(5):
        valid=(1<<min(16,rows-batch*16))-1
        for core in range(8):pending.append((batch,core,0,valid if core==7 else rng.randrange(1,valid+1)))
    rng.shuffle(pending)
    while pending:
        batch,core,expected,ready=pending.pop()
        old=cas(memory,batch,expected,expected|ready)
        valid=(1<<min(16,rows-batch*16))-1
        if old==expected:
            acquired=ready&~expected
            for bit in range(16):
                if acquired&(1<<bit):
                    row=batch*16+bit
                    assert row not in owners;owners[row]=core
            if acquired and memory[batch]==valid:completed+=1
        elif ready&~old:
            # Requeue at most one stale-snapshot retry, never spin on a row.
            pending.insert(rng.randrange(len(pending)+1),(batch,core,old,ready))
    assert len(owners)==rows and completed==5


def test_ready_claims_do_not_pin_terminal_batches_before_first_hop():
    rows=65;claimed=[0]*5;hits=[0]*rows;ready={64};finished=set();completed=0
    rng=random.Random(42)
    # The last row is the only FIRST_HOP. All earlier rows depend on its relay.
    # Four active workers initially see the four entirely unready front batches.
    for _ in range(100):
        order=list(range(5));rng.shuffle(order)
        for batch in order:
            count=min(16,rows-batch*16);valid=(1<<count)-1
            mask=sum(1<<bit for bit in range(count) if batch*16+bit in ready)
            acquired=mask&~claimed[batch]
            if not acquired:continue
            previous=claimed[batch];claimed[batch]|=acquired
            if claimed[batch]==valid and previous!=valid:completed+=1
            for bit in range(count):
                if acquired&(1<<bit):
                    row=batch*16+bit;hits[row]+=1;finished.add(row)
                    if row==64:ready.update(range(64))
                    ready.discard(row)
        if completed==5:break
    assert all(hit==1 for hit in hits) and len(finished)==65


@pytest.mark.parametrize('experts,ranks,bs',[(16,16,1),(384,16,256),(1024,256,19)])
@pytest.mark.parametrize('recv',[1,28,55])
def test_parallel_position_map_preserves_static_output_indices(experts,ranks,bs,recv):
    rng=random.Random(experts+recv)
    counts=[rng.randrange(bs+1) for _ in range(experts)]
    prefix=[];total=0
    for count in counts:total+=count;prefix.append(total)
    expected=[];rank=ranks//3;local=experts//ranks
    for record,count in enumerate(counts):
        slot=((record%ranks+rank)%ranks*local+record//ranks)*bs
        expected.extend(slot+q for q in range(count))
    output=[None]*total
    for worker in range(recv):
        start=worker*(experts//recv)+min(worker,experts%recv)
        end=start+experts//recv+(worker<experts%recv)
        previous=prefix[start-1] if start else 0
        for record in range(start,end):
            count=prefix[record]-previous
            slot=((record%ranks+rank)%ranks*local+record//ranks)*bs
            assert all(v is None for v in output[previous:prefix[record]])
            output[previous:prefix[record]]=range(slot,slot+count)
            previous=prefix[record]
    assert output==expected


def test_global_claim_completion_does_not_imply_dma_completion():
    claimed=[0];valid=0xffff;completed_chunks=0;in_flight=[];outputs={}
    old=cas(claimed,0,0,valid)
    assert old==0;completed_chunks+=1;in_flight.extend(range(16))
    assert completed_chunks==1 and not outputs
    # Other workers may return, but this owner retains the work through its final
    # MTE drain. A subsequent kernel can only start after all workers return.
    while in_flight:
        row=in_flight.pop();assert row not in outputs;outputs[row]=row
    assert len(outputs)==16


def test_actual_layout_keeps_queue_outside_control_and_live_packets(tmp_path):
    compiler=shutil.which('g++')
    if not compiler:pytest.skip('C++ compiler unavailable; actual dynamic queue memory layout requires compiler')
    source=tmp_path/'layout.cpp'
    source.write_text(r'''
#include <cassert>
#include <algorithm>
#include "csrc/kernels/dispatch/low_latency/common/ep_memory_server_dedup_layout.h"
using namespace DispatchDedup;
int main(){
 for(uint32_t p: {16U,32U,64U,128U,256U})
 for(uint32_t e: {p,2U*p,4U*p})
 for(uint32_t b: {1U,8U,32U,256U})
 for(uint32_t k: {1U,4U,8U})
 for(uint32_t h: {32U,128U,3584U,7168U})
 for(auto mode: {EpServerDedupInputMode::Plain16,EpServerDedupInputMode::Quantize16,EpServerDedupInputMode::PrequantizedFp8Packs}){
  if(mode==EpServerDedupInputMode::PrequantizedFp8Packs && h%128)continue;
  auto d=DispatchServerDedupDataLayout(b,h,k,e,mode,p);
  uint64_t cap=uint64_t(p)*b*std::min(k,e/p);
  auto q=LlReceiveQueueMemory(cap);
  assert(d.receiveQueueOffset==kEpServerDedupControlBytes);
  assert(d.receivePositionsOffset==d.receiveQueueOffset+q.positionOffset);
  assert(d.receiveQueueBytes==q.bytes);
  assert(d.receiveQueueOffset+kLlReceiveHeaderBytes+d.receiveClaimBytes<=d.receivePositionsOffset);
  assert(d.receivePositionsOffset+cap*4<=d.receiveQueueOffset+d.receiveQueueBytes);
  assert(d.receiveQueueOffset+d.receiveQueueBytes<=d.dispatchOffset);
  assert(d.sourceOffset+d.sourceBytes<=d.dispatchOffset);
  assert(d.dispatchOffset+d.dispatchBytes<=d.relayOffset);
  assert(d.requiredBytes==d.relayOffset+d.relayBytes);
  assert(d.receivePositionsOffset%512==0 && d.dispatchOffset%512==0);
 }
}
''',encoding='utf-8')
    executable=tmp_path/'layout'
    subprocess.run([compiler,'-std=c++17','-O2','-Wall','-Wextra','-Werror',f'-I{ROOT}',str(source),'-o',str(executable)],check=True)
    subprocess.run([str(executable)],check=True,timeout=30)


@pytest.mark.parametrize('fragments',[1,16,28,55])
def test_raw_integer_float_sum_clears_actual_not_capacity_words(fragments):
    import struct
    rng=random.Random(fragments)
    counts=[rng.randrange(20000) for _ in range(fragments)]
    value=0.0
    for count in counts:
        raw=struct.unpack('<f',struct.pack('<I',count))[0]
        value=struct.unpack('<f',struct.pack('<f',value+raw))[0]
    actual=struct.unpack('<I',struct.pack('<f',value))[0]
    assert actual==sum(counts) and actual<(1<<23)
    capacity=actual+100000
    def claim_bytes(rows):return (((rows+15)//16*4+511)//512)*512
    assert claim_bytes(actual)<=claim_bytes(capacity)
    assert claim_bytes(0)==0 and claim_bytes(2048)==512 and claim_bytes(2049)==1024


@pytest.mark.parametrize('batches',[1,2,5,16,65,128,257])
def test_cyclic_probe_tickets_cover_rows_after_uint32_wrap(batches):
    cursor=(1<<32)-8;visited=set()
    for _ in range((batches+7)//8+2):
        start=cursor%batches
        visited.update((start+i)%batches for i in range(8))
        cursor=(cursor+8)&0xffffffff
    assert visited==set(range(batches))
