"""Packet detail CPU models: not hardware timing/performance evidence."""
import copy
import subprocess
from pathlib import Path
import pytest
from tests.ll_reference_profile.test_parse import sample, timer, setword
from tests.ll_reference_profile.test_device_profile import headers, compiler, COMMON
from tests.ll_rdma.test_combine_overhead import block, run_cpp
ROOT=Path(__file__).resolve().parents[2]


def sender_raw():
    saved=sample(offset=10**17)
    w=[v for row in saved['raw'][4] for v in row];base=w[0]
    w[65]=3;w[66]=2;w[67]=base+102;w[68]=base+155
    w[69]=base+110;w[70]=base+150
    w[71]=base+120;w[72]=base+130;w[73]=base+135;w[74]=base+140;w[75]=3
    w[76:80]=[0,1,12,1]
    w[80:90]=[6,1,base+159,1,1,5,2,4,5,8]
    w[90:96]=[base+131,base+132,base+134,3,2,1]
    saved['raw'][4]=[w[i:i+8] for i in range(0,timer.WORD_COUNT,8)]
    return saved


def receiver_raw():
    saved=sample();c=15;w=[v for row in saved['raw'][c] for v in row]
    w[96:102]=[2,2,2,2,2,2];w[107]=0;w[108]=1
    # Both found in batch1, scan finds j0 then j1, consumption runs reverse.
    # Position1 decodes source1 with B1/localExperts1/rank0.
    for j,fields in enumerate(([1,1,2,0,0,0,1,0],[2,1,1,1,0,0,2|(3<<8),1])):
        w[128+j*4:132+j*4]=[fields[i]|(fields[i+1]<<32) for i in range(0,8,2)]
    saved['raw'][c]=[w[i:i+8] for i in range(0,timer.WORD_COUNT,8)]
    return saved


def test_second_packet_excludes_member_q_and_batch_quiet():
    d=timer.parse_records(sender_raw()['raw'])['cores'][4]['packet_detail']
    assert d['ticks']==dict(second_schedule_without_q=5,second_member_q=3,second_stage=5,
        second_previous_submit=2,second_defer_gap=1,second_stage_wait=2,
        second_put_call=5,second_enqueue_to_put_return=10,remote_submit_span=53,
        remote_retire_span=57,group_submit_span=40,batch_quiet_total=6)
    assert d['intervals_interleave'] and d['second']['triggered_flush']
    assert d['second']['peer']==1
    assert timer.parse_records(sample()['raw'])['cores'][4]['packet_detail']['second'] is None


def test_stable_identity_and_distinct_discovery_consumption_order():
    saved=receiver_raw();d=timer.parse_records(saved['raw'])['cores'][15]['packet_detail']
    assert [e['ready_order'] for e in d['entries']]==[1,2]
    assert [e['process_order'] for e in d['entries']]==[2,1]
    assert d['entries'][1]['packet_role']=='FIRST_HOP' and d['entries'][1]['forward_members']==3
    assert not d['truncated']
    # Truncation preserves original order numbers instead of inventing complete coverage.
    setword(saved['raw'],15,96,1);setword(saved['raw'],15,97,1);setword(saved['raw'],15,107,1)
    for i in range(132,136):setword(saved['raw'],15,i,0)
    d=timer.parse_records(saved['raw'])['cores'][15]['packet_detail']
    assert d['truncated'] and d['entries'][0]['process_order']==2


@pytest.mark.parametrize('core,index,value',[(4,64,0),(4,75,99),(4,82,1),(4,90,1),(15,107,1),(15,99,1),(15,102,0)])
def test_bad_packet_details_fail_closed(core,index,value):
    saved=sender_raw() if core==4 else receiver_raw()
    setword(saved['raw'],core,index,value)
    with pytest.raises(ValueError):timer.parse_records(saved['raw'])


def test_real_scalar_sender_recorder_writes_only_at_finish(tmp_path):
    headers(tmp_path)
    source=tmp_path/'sender.cpp'
    source.write_text(r"""
#include "ll_reference_profile_device.h"
#include <cassert>
using namespace LlReferenceProfile;
int main(){
 alignas(512) uint64_t data[kWordsPerCore]{};
 DeviceProfile p;p.Init(reinterpret_cast<uint8_t*>(data));
 p.SendLanes(8);
 p.SendBegin(true,false,3,2,192,16);assert(!p.SecondRemote());p.SecondEnqueue();p.SecondPutBegin();p.PutReturned();
 p.SendBegin(false,true,3,0,0,0);p.GroupSubmitted();
 p.SendBegin(true,false,4,1,204,17);assert(p.SecondRemote());
 auto q=p.Clock();auto qend=p.Clock();p.MemberQ(qend-q);
 p.SecondEnqueue();p.SecondPutBegin();p.PutReturned();auto quiet=p.Clock();p.FlushDone(quiet,true);
 p.SendBegin(true,false,5,3,216,18);assert(!p.SecondRemote());p.SecondEnqueue();p.SecondPutBegin();p.PutReturned();
 quiet=p.Clock();p.FlushDone(quiet,false);
 for(auto x:data)assert(x==0); // no packet-loop Scalar GM detail writes
 p.Finish(0,32,8);
 assert(data[64]==1 && data[65]==3 && data[66]==1);
 assert(data[76]==4 && data[77]==1 && data[78]==204 && data[79]==17);
 assert(data[75]==17 && data[83]==1 && data[81]==2);
 assert(data[71]<data[72] && data[72]<data[73] && data[73]<data[74]);
 assert(data[74]<data[68] && data[68]<data[82]);
 assert(data[84]==1 && data[85]==4 && data[86]==2 && data[87]==2);
}
""",encoding='utf-8')
    exe=tmp_path/'sender.exe'
    subprocess.run(compiler()+['-std=c++17','-UNDEBUG','-DDEEPEP_LL_REFERENCE_PROFILE=1',
        '-I'+str(tmp_path),'-I'+str(COMMON),str(source),'-o',str(exe)],check=True,timeout=180)
    subprocess.run([str(exe)],check=True,timeout=30)


def test_real_receiver_ub_ledger_handles_reordered_pending_and_truncation(tmp_path):
    text=(ROOT/'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_recv.h').read_text(encoding='utf-8')
    methods=block(text,'__aicore__ inline void RecordReady(')+'\n'+block(text,'__aicore__ inline void RecordProcessed(')
    run_cpp(tmp_path,r"""
#include <cassert>
#include <cstdint>
#include <cstring>
#define __aicore__
template<class T> struct LocalTensor {
 T* p;T GetValue(unsigned i){return p[i];}void SetValue(unsigned i,T v){p[i]=v;}
 LocalTensor operator[](unsigned i){return {p+i};}
 template<class U>LocalTensor<U> ReinterpretCast(){return {reinterpret_cast<U*>(p)};}
};
enum class EpServerDedupTokenType:uint32_t{Direct=1,FirstHop=2,Terminal=3};
struct Test {
 bool traceEnabled_=true;uint32_t traceCapacity_=2,traceReady_=0,traceBatches_=0,traceProcessed_=0;
 uint32_t log[16]{};
 struct Buf {uint32_t* p;template<class T>LocalTensor<T> Get(){return {reinterpret_cast<T*>(p)};}} packetTraceBuf_{log};
 uint32_t positions[4]={100,101,102,103};Buf positionBuf_{positions};
 struct Layout {uint32_t relayReadIndexWireOffset=32,readIndexPayloadOffset=32;} data_;
"""+methods+r"""
};
int main(){Test t;uint32_t pending[]={3,1,0,2};
 t.RecordReady({pending},7,4); // discover j3,j1,j0; only j1,j0 fit
 assert(t.traceReady_==3 && t.traceBatches_==1);
 assert(t.log[0]==3 && t.log[8]==2);
 uint32_t raw[16]{};raw[0]=7;raw[1]=9;raw[2]=4;raw[3]=2;raw[8]=3;
 LocalTensor<uint8_t> wire{reinterpret_cast<uint8_t*>(raw)};
 t.RecordProcessed(0,wire,0,true);t.RecordProcessed(1,wire,0,true);t.RecordProcessed(3,wire,0,true);
 assert(t.log[2]==1 && t.log[10]==2 && t.traceProcessed_==3);
 assert(t.log[3]==7 && t.log[4]==9 && t.log[5]==4 && t.log[6]==(2|(3<<8)) && t.log[7]==100);
 // No capacity means no UB writes, but complete coverage counters remain truthful.
 Test z;z.traceCapacity_=0;z.RecordReady({pending},15,4);z.RecordProcessed(3,wire,0,true);
 assert(z.traceReady_==4 && z.traceProcessed_==1);for(auto x:z.log)assert(x==0);
}
""",'receiver-ledger')


def test_trace_uses_only_spare_ub_and_exports_after_business():
    recv=(ROOT/'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_recv.h').read_text(encoding='utf-8')
    kernel=(ROOT/'csrc/kernels/dispatch/low_latency/dispatch_server_dedup_kernel.h').read_text(encoding='utf-8')
    assert 'kEpServerDedupCountUbBytes-fixed-lanes_*stride_' in recv
    allocation=recv[recv.index('traceCapacity_=spare'):recv.index('Duplicate<uint32_t>(seenBuf_')]
    assert 'lanes_ =' not in allocation
    assert kernel.index('profile_->Mark<21>')<kernel.index('receiver_.ExportPacketTrace')
    export=block(recv,'__aicore__ inline void ExportPacketTrace(')
    assert export.count('DataCopy(')==1



def test_actual_batch_auto_flush_is_not_charged_to_second_put(tmp_path):
    (tmp_path/'shmem.h').write_text('#pragma once\ninline void dcci_cachelines(uint8_t*,uint64_t) {}\n')
    source=tmp_path/'batch-on.cpp'
    source.write_text(r"""
#include <cassert>
#include <vector>
#define DEEPEP_LL_RDMA 1
#include "kernels/dispatch/low_latency/device/ll_dispatch_rdma_batch.h"
using namespace AscendC;
int main(){
 core=0;cores=16;cycles=100;
 std::vector<uint8_t> local(1U<<20),remote(1U<<20),stage(16U*LlRdma::kCoreBytes),ub(65536);
 Model::local=local.data();Model::remote=remote.data();
 DispatchDedup::CommArgs a{};a.rank=0;a.rankSize=32;a.peerMemBytes=1U<<20;a.reserved0=16;
 a.reservedConfig[0]=LlRdma::kBackend;a.reservedConfig[1]=16;
 a.reservedAddresses[0]=local.data();a.reservedAddresses[1]=stage.data();
 uint64_t raw[16*LlReferenceProfile::kWordsPerCore]{};
 LlReferenceProfile::DeviceProfile p;p.Init(reinterpret_cast<uint8_t*>(raw));
 LlTransport::DispatchRdmaBatch batch;batch.SetProfile(&p);batch.Init(&a,65536);
 Model::onPost=[](Model::Request&){cycles+=10;};
 Model::onQuiet=[](uint32_t,uint32_t){cycles+=20;};
 for(unsigned t=0;t<3;++t){p.SendBegin(true,false,t,0,192,16);batch.Enqueue(16,t*65536,{ub.data()});}
 batch.Flush({ub.data()});p.Finish(0,32,3);
 assert(raw[65]==3 && raw[81]==2 && raw[83]==1);
 assert(raw[74]-raw[73]==10); // second put excludes following 20-cycle quiet
 assert(raw[80]==40 && raw[82]>raw[74]);
 assert(Model::quiets==2 && Model::pending.empty());
}
""",encoding='utf-8')
    exe=tmp_path/'batch-on.exe'
    subprocess.run(compiler()+['-std=c++17','-UNDEBUG','-DDEEPEP_LL_REFERENCE_PROFILE=1',
        '-I'+str(tmp_path),'-I'+str(ROOT/'tests/ll_rdma/stubs'),'-I'+str(ROOT/'csrc'),
        str(source),'-o',str(exe)],check=True,timeout=180)
    subprocess.run([str(exe)],check=True,timeout=30)


def test_receiver_batch_table_keeps_tuple_alignment_and_unknown_truncated_count():
    from tests.ll_reference_profile.export_xlsx import receiver_batch_rows
    saved=receiver_raw()
    setword(saved['raw'],15,108,1)
    rows=receiver_batch_rows([saved],timer,1e9)
    first=rows[1]
    assert first[:6]==[0,15,1,2,2,True]
    assert first[6:9]==['(DIRECT, FIRST_HOP)','(0, 1)','(0, 0)']
    assert first[9]=='((DIRECT, 0, 0), (FIRST_HOP, 1, 0))'
    assert first[11]=='(2, 1)'
    # A truncated trace can omit an entire successful batch.
    setword(saved['raw'],15,108,2)
    setword(saved['raw'],15,96,1);setword(saved['raw'],15,97,1);setword(saved['raw'],15,107,1)
    for i in range(132,136):setword(saved['raw'],15,i,0)
    rows=receiver_batch_rows([saved],timer,1e9)
    assert rows[1][3] is None and rows[1][4:6]==[1,False]
    assert rows[2][3] is None and rows[2][4]==0



def test_delayed_second_packet_keeps_identity_when_next_packet_is_local(tmp_path):
    (tmp_path/'shmem.h').write_text('#pragma once\ninline void dcci_cachelines(uint8_t*,uint64_t) {}\n')
    source=tmp_path/'deferred-profile.cpp'
    source.write_text(r"""
#include <cassert>
#include <vector>
#define DEEPEP_LL_RDMA 1
#include "kernels/dispatch/low_latency/device/ll_dispatch_rdma_batch.h"
using namespace AscendC;
int main(){
 core=0;cores=16;cycles=100;
 std::vector<uint8_t> local(1U<<20),remote(1U<<20),stage(16U*LlRdma::kCoreBytes),aUB(65536),bUB(65536);
 Model::local=local.data();Model::remote=remote.data();
 DispatchDedup::CommArgs a{};a.rank=0;a.rankSize=32;a.peerMemBytes=1U<<20;a.reserved0=16;
 a.reservedConfig[0]=LlRdma::kBackend;a.reservedConfig[1]=16;
 a.reservedAddresses[0]=local.data();a.reservedAddresses[1]=stage.data();
 uint64_t raw[16*LlReferenceProfile::kWordsPerCore]{};
 LlReferenceProfile::DeviceProfile p;p.Init(reinterpret_cast<uint8_t*>(raw));
 LlTransport::DispatchRdmaBatch batch;batch.SetProfile(&p);batch.Init(&a,65536);
 Model::onPost=[](Model::Request&){cycles+=10;};Model::onQuiet=[](uint32_t,uint32_t){cycles+=20;};
 p.SendBegin(true,false,0,0,192,16);batch.Enqueue(16,0,{aUB.data()},true);
 cycles+=5;p.SendBegin(true,false,1,3,193,16); // prepare packet2 while packet1 staged
 auto begin=p.Clock();batch.SubmitPending();p.PreviousSubmit(p.Clock()-begin);
 cycles+=7;batch.Enqueue(16,65536,{bUB.data()},true);
 cycles+=9;p.SendBegin(false,true,7,1,2,0); // mutable current identity is now local
 batch.SubmitPending(); // must still record remote packet2, including its full batch quiet
 cycles+=3;p.GroupSubmitted();p.Finish(0,32,8);
 assert(raw[65]==2 && raw[66]==1 && raw[76]==1 && raw[77]==3 && raw[78]==193);
 assert(raw[74]-raw[73]==10 && raw[83]==1 && raw[80]==20);
 assert(raw[93]==2 && raw[94]==10 && raw[95]==1);
 assert(raw[90]<=raw[91] && raw[91]<=raw[92] && raw[92]<=raw[73]);
 assert(raw[69]<raw[74] && raw[86]>raw[85]); // time overlap without packet-order interleave
}
""",encoding='utf-8')
    exe=tmp_path/'deferred-profile.exe'
    subprocess.run(compiler()+['-std=c++17','-UNDEBUG','-DDEEPEP_LL_REFERENCE_PROFILE=1',
        '-I'+str(tmp_path),'-I'+str(ROOT/'tests/ll_rdma/stubs'),'-I'+str(ROOT/'csrc'),
        str(source),'-o',str(exe)],check=True,timeout=180)
    subprocess.run([str(exe)],check=True,timeout=30)
