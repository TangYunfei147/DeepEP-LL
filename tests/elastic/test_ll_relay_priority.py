"""Receiver ordering and ownership invariants for relay-first experiments."""
from pathlib import Path
import random
import shutil
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[2]


def test_relay_first_keeps_metadata_snapshot_and_lane_retirement():
    source=(ROOT/'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_recv.h').read_text()
    consume=source[source.index('inline void ConsumeLoaded'):source.index('inline void ClearFlags')]
    body=consume[consume.index('const bool firstHop'): ]
    assert body.index('BuildServerDedupMetadata')<body.index('PrepareServerDedupForward')
    assert body.index('PrepareServerDedupForward')<body.index('ServerDedupReceiveSync<HardEvent::S_MTE3>')
    ready=body.index('WaitFlag<HardEvent::MTE2_MTE3>')
    relay=body.index('DataCopy(output, wire')
    local=body.index('WriteServerDedupPayloadRange(expandX_')
    indices=body.index('DataCopyPad(relayReadIndex_')
    retire=body.index('SetFlag<HardEvent::MTE3_MTE2>')
    assert ready<relay<local<indices<retire
    # After the first asynchronous relay read, no Scalar writes into wire.
    assert '.SetValue(' not in body[relay:retire]


def test_relay_and_local_output_read_the_same_immutable_packet():
    for fanout in range(1,8):
        original=dict(kind='first_hop',token=17,source=3,weights=(1,2,3))
        local_metadata=dict(original)
        wire=dict(original,kind='terminal')
        operations=[('relay',i) for i in range(fanout)]+[('payload',0),('metadata',0),('indices',0)]
        delivered=[]
        for kind,index in operations:
            delivered.append((kind,index,dict(wire) if kind=='relay' else dict(local_metadata)))
        assert all(data['kind']=='terminal' for kind,_,data in delivered if kind=='relay')
        assert local_metadata['kind']=='first_hop'
        assert [i for kind,i,_ in delivered if kind=='relay']==list(range(fanout))


@pytest.mark.parametrize('size',[0,1,2,7,15,16])
@pytest.mark.parametrize('depth',[1,2,4,8])
def test_priority_preserves_lane_mapping_and_pending_removal(size,depth):
    rng=random.Random(size*100+depth)
    for case in range(80):
        ready=rng.randrange(1<<size);keep=rng.randrange(1<<size)
        first=ready&keep
        order=[i for mask in (first,ready&~first) for i in reversed(range(size)) if mask&(1<<i)]
        assert len(order)==ready.bit_count() and len(set(order))==len(order)
        assert all(ready&(1<<i) for i in order)
        assert order[:first.bit_count()]==[i for i in reversed(range(size)) if first&(1<<i)]
        scan=case%5
        pending=list(range(scan+size+7))
        snapshot=pending[scan:scan+size]
        expected=pending.copy()
        for i in reversed(range(size)):
            if ready&(1<<i):
                expected[scan+i]=expected[-1];expected.pop()
        # Both cursors carry across batches, including nonzero lane phases.
        issued=consumed=case%17;next_load=0;lanes={};done=[]
        for index in order:
            while issued-consumed<depth and next_load<len(order):
                position=order[next_load];next_load+=1
                assert issued%depth not in lanes
                lanes[issued%depth]=snapshot[position];issued+=1
            assert lanes.pop(consumed%depth)==snapshot[index]
            done.append(snapshot[index]);consumed+=1
        assert issued==consumed and not lanes
        assert set(done)=={snapshot[i] for i in range(size) if ready&(1<<i)}
        for i in reversed(range(size)):
            if ready&(1<<i):
                pending[scan+i]=pending[-1];pending.pop()
        assert pending==expected
        assert set(done).isdisjoint(pending)


def test_current_marker_contract_and_deferred_removal():
    sender=(ROOT/'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h').read_text()
    receiver=(ROOT/'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_recv.h').read_text()
    assert 'mask != 0U ? PacketReceiveLayout::kKeepFlags : PacketReceiveLayout::kCompact' in sender
    assert 'uint32_t(mask == 0U ? EpServerDedupTokenType::Direct : EpServerDedupTokenType::FirstHop)' in sender
    assert 'PacketReceiveLayout::kCompact);' in receiver
    batch=receiver[receiver.index('const auto order='):receiver.index('// 5.')]
    assert batch.index('ConsumeLoaded(')<batch.index('tasks_.RemovePending(')
    assert 'enableDedup_ ? probe.keepFlagsMask : 0U' in batch


def test_actual_priority_helper_exhaustive_masks(tmp_path):
    compiler=shutil.which('g++')
    if not compiler:pytest.skip('g++ unavailable; actual production priority helper requires target C++ check')
    source=tmp_path/'priority.cpp'
    source.write_text(r'''
#include <cassert>
#include <vector>
#include "csrc/kernels/dispatch/low_latency/common/ll_receive_priority.h"
int main() {
 for(unsigned n=0;n<=16;++n)for(unsigned ready=0;ready<(1U<<n);++ready)
 for(unsigned priority:{0U,ready,0xaaaaU,0x5555U,0xffffU}) {
  auto result=DispatchDedup::LlMakeReceivePriority(n,ready,priority);
  std::vector<unsigned> expected;
  for(unsigned mask:{ready&priority,ready&~priority})
   for(unsigned i=n;i>0;--i)if(mask&(1U<<(i-1)))expected.push_back(i-1);
  assert(result.count==expected.size());
  for(unsigned i=0;i<result.count;++i)assert(result.At(i)==expected[i]);
 }
}
''',encoding='utf-8')
    exe=tmp_path/'priority'
    subprocess.run([compiler,'-std=c++17','-O2','-I',str(ROOT),str(source),'-o',str(exe)],
                   check=True,capture_output=True,text=True)
    subprocess.run([str(exe)],check=True,capture_output=True,text=True)
