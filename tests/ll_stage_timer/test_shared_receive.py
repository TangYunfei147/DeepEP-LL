"""Shared dispatch receive: publication lifetime, disjoint ownership and diagnostics."""
import importlib.util
from itertools import accumulate
from pathlib import Path
import random
import pytest

ROOT = Path(__file__).resolve().parents[2]
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

parity = load('shared_parity', 'tests/ll_stage_timer/test_parity.py')









@pytest.mark.parametrize('send', [1,10,28,55])
def test_asynchronous_prefix_publication_and_two_bank_reuse(send):
    # A consumer may clear its own ready row while another is still sending.
    # Producers publish payload before each consumer's flag; no consumer waits
    # for another consumer. Simulate delayed arrivals and reuse both banks.
    recv=56-send; experts=16; producers=min(recv,experts)
    banks=[dict(prefix=[[None]*experts for _ in range(56)], flags=[[0]*recv for _ in range(56)]) for _ in range(2)]
    rng=random.Random(send)
    for iteration in range(6):
        bank=banks[iteration%2]
        expected=list(accumulate([(i+iteration)%4 for i in range(experts)]))
        actions=[]
        for producer in range(producers):
            base, rem=divmod(experts,recv)
            begin=producer*base+min(producer,rem); end=begin+base+(producer<rem)
            actions.append([(c,producer,begin,end) for c in range(56)])
        cursor=[0]*producers; consumed=set()
        arrival=[rng.randrange(0,56*producers+10) if c<send else 0 for c in range(56)]
        for tick in range(56*producers+max(arrival)+2):
            pending=[p for p in range(producers) if cursor[p]<56]
            if pending:
                p=rng.choice(pending);c,p,begin,end=actions[p][cursor[p]];cursor[p]+=1
                bank['prefix'][c][begin:end]=expected[begin:end]
                bank['flags'][c][p]=1
            for c in range(56):
                if c in consumed or tick<arrival[c]: continue
                if all(bank['flags'][c][p] for p in range(producers)):
                    assert bank['prefix'][c]==expected
                    bank['flags'][c][:producers]=[0]*producers
                    consumed.add(c)
        assert len(consumed)==56
        assert not any(any(row) for row in bank['flags'])


@pytest.mark.parametrize('total', [0,1,7,55,56,57,1024,10003])
def test_receive_output_and_cleanup_ownership_are_disjoint(total):
    slices=[range(c*total//56,(c+1)*total//56) for c in range(56)]
    assert [row for part in slices for row in part]==list(range(total))
    assert max(map(len,slices))-min(map(len,slices))<=1




def test_production_wiring_keeps_producers_separate_from_consumers():
    base=ROOT/'csrc/kernels/dispatch/low_latency'
    kernel=(base/'dispatch_server_dedup_kernel.h').read_text(encoding='utf-8')
    counts=(base/'device/dispatch_server_dedup_counts.h').read_text(encoding='utf-8')
    receiver=(base/'device/dispatch_server_dedup_recv.h').read_text(encoding='utf-8')
    assert kernel.index('sender_.Process(')<kernel.rindex('counts_.CompleteCumsum();')<kernel.rindex('receiver_.Process();')
    assert kernel.count('receiver_.Process();')==2
    assert 'recvCoreNum_ = enableDedup ? kLlDispatchRecvCoreNum : kLlRecvCoreNum;' in receiver
    assert 'receiver_ = enableDedup ? GetBlockIdx()' in receiver
    assert 'if (aivId_!=kLlSendCoreNum) return;' in counts
    assert 'DataCopyPad(workspace,prefix,copy)' in counts
    assert 'for (uint32_t index = 0; index < coreNum_; index++)' not in counts
    complete=counts[counts.index('inline void ServerDedupCounts::CompleteCumsum()'):]
    assert complete.index('PipeBarrier<PIPE_ALL>();')<complete.index('tpipe_->Reset();')<complete.index('WaitCumSumFlag();')

    assert complete.index("tpipe_->Reset();")<complete.index("SetCtrlSpr<60,60>(0);")<complete.index("WaitCumSumFlag();")
