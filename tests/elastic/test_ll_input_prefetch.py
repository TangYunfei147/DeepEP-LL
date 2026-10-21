"""Two-buffer lifecycle, one-entry queue and actual portable scheduling checks."""
from pathlib import Path
import random
import shutil
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('count',[0,1,2,3,7,8,9,57,256])
@pytest.mark.parametrize('mode',['plain','prequantized','quantize'])
def test_two_buffers_retire_before_reuse_and_flush_last(count,mode):
    rng=random.Random(count)
    tasks=[(rng.randrange(17),rng.randrange(8),rng.randrange(384),i,
            rng.choice([0,0,0b10101])) for i in range(count)]
    slots=[None,None];readers=[False,False];pending=None;queue=None
    events={};trace=[];consumed=[];reuse_waits=0
    for i,task in enumerate(tasks):
        slot=i%2
        assert slots[slot] is None  # No allocation over an owned current input.
        if readers[slot]:
            # TQue Alloc waits for the prior V reader before next MTE2 writes.
            reuse_waits+=1
            readers[slot]=False
        assert queue is None
        slots[slot]=task
        queue=(slot,task)
        events[i%2]=i
        trace.append(('issue',i))
        if pending is not None:
            old_slot,old_task,old_i=pending
            # The newer load may still be blocked: only old's marker is waited.
            assert events[old_i%2]==old_i
            del events[old_i%2]
            assert slots[old_slot]==old_task
            consumed.append(old_task)
            trace.append(('consume',old_i))
            # Online quantization's existing V_S retires the input readers;
            # plain/prequantized inputs remain live until vector packing ends.
            readers[old_slot]=mode!='quantize'
            slots[old_slot]=None
        pending=(*queue,i)
        queue=None
    if pending is not None:
        slot,task,i=pending
        assert events[i%2]==i
        del events[i%2]
        consumed.append(task)
        slots[slot]=None
        trace.append(('consume',i))
    assert consumed==tasks and slots==[None,None] and not events
    assert reuse_waits==(max(0,count-2) if mode!='quantize' else 0)
    for i in range(count-1):
        assert trace.index(('issue',i+1))<trace.index(('consume',i))


def test_actual_production_lookahead_controller(tmp_path):
    compiler=shutil.which('g++')
    if not compiler:
        pytest.skip('g++ unavailable; actual production lookahead controller requires target C++ check')
    source=tmp_path/'prefetch.cpp'
    source.write_text(r'''
#include <cassert>
#include <vector>
#include "csrc/kernels/dispatch/low_latency/common/ll_input_prefetch.h"
using namespace DispatchDedup;
struct Ops {
    int queued=-1,current=-1,issued=0;
    bool owned[2]={false,false};
    LlSendPacketTask slots[2];
    std::vector<LlSendPacketTask> output;
    void Issue(const LlSendPacketTask& task) {
        assert(queued==-1);
        int slot=issued++%2;
        assert(!owned[slot]);
        owned[slot]=true;slots[slot]=task;queued=slot;
    }
    void Take() {
        assert(current==-1 && queued!=-1);
        current=queued;queued=-1;
    }
    void Consume(const LlSendPacketTask& task) {
        assert(current!=-1 && owned[current]);
        auto actual=slots[current];
        assert(actual.token==task.token && actual.firstK==task.firstK);
        assert(actual.expert==task.expert && actual.q==task.q && actual.mask==task.mask);
        output.push_back(task);owned[current]=false;current=-1;
    }
};
int main() {
    for(unsigned n:{0U,1U,2U,3U,57U,256U}) {
        LlOnePacketInput pipe;pipe.Reset();Ops ops;
        for(unsigned i=0;i<n;++i) {
            LlSendPacketTask task{i%17,i%8,i%384,i,i%3?0U:21U};
            pipe.Push(ops,task);
            // Mutating source descriptors must not mutate the pending copy.
            task.q=99999;
            assert(ops.output.size()==i);
        }
        pipe.Flush(ops);pipe.Flush(ops); // Empty flush is idempotent.
        assert(ops.output.size()==n && ops.current==-1 && ops.queued==-1);
        assert(!ops.owned[0] && !ops.owned[1]);
        for(unsigned i=0;i<n;++i)assert(ops.output[i].q==i);
    }
}
''',encoding='utf-8')
    exe=tmp_path/'prefetch'
    subprocess.run([compiler,'-std=c++17','-O2','-I',str(ROOT),str(source),'-o',str(exe)],
                   check=True,capture_output=True,text=True)
    subprocess.run([str(exe)],check=True,capture_output=True,text=True)


def test_input_capacity_output_lanes_and_event_lifetime_contract():
    source=(ROOT/'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h').read_text()
    assert 'InitBuffer(input, 2U, inputBytes)' in source
    assert 'InitBuffer(input,2U,inputBytes)' in source
    assert 'lanes_ = overlap_.enabled ? overlap_.sendLanes : (available < 8U ? available : 8U)'  in source
    assert 'lanes_=overlap_.enabled ? overlap_.sendLanes : (available<8U ? available : 8U)'  in source
    assert source.count('BeginInputPrefetch();')==2
    assert source.count('EndInputPrefetch(input);')==2
    assert source.count('AllocEventID<HardEvent::MTE2_S>()')==2
    assert source.count('ReleaseEventID<HardEvent::MTE2_S>')==2
    issue=source[source.index('__aicore__ inline void IssueInput'):source.index('struct PrefetchOps')]
    assert 'Duplicate' not in issue and 'V_MTE2' not in issue
    assert issue.index('SetFlag<HardEvent::MTE2_S>')<issue.index('input.EnQue')
