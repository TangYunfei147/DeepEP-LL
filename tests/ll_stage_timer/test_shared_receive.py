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

timer = load('shared_timer', 'deep_ep/utils/ll_stage_timer.py')
expert = load('shared_expert', 'deep_ep/utils/ll_expert_timer.py')
parity = load('shared_parity', 'tests/ll_stage_timer/test_parity.py')

def raw(operation='dispatch', send=10, schema=6):
    result = []
    for core in range(56):
        words = [0] * 256
        words[:8] = [timer.MAGIC, schema, 1 if operation=='dispatch' else 2, 56,
                     core, 1 | (send << 8), 1, len(timer.legacy.STAGES)]
        for stage, begin, end in [('kernel', 1, 100), ('sender', 2, 30), ('receiver', 35, 90), ('recv_consume', 40, 80)]:
            if stage=='sender' and core>=send: continue
            offset = 8 + (timer.legacy.STAGES.index(stage)-1)*4
            words[offset:offset+4] = [begin, end, end-begin, 1]
        result.append([words[n:n+8] for n in range(0,256,8)])
    return result


def test_schema6_roles_distinguish_send_then_receive_and_legacy():
    report = timer.parse_records(raw(), 'dispatch')
    core0 = {row['stage']: row['role'] for row in report['records'] if row['core']==0}
    assert core0 == dict(kernel='send_receive', sender='send', receiver='receive', recv_consume='receive')
    assert report['dispatch_recv_cores']==56 and report['recv_cores']==46
    assert timer.parse_records(raw(schema=5), 'dispatch')['dispatch_recv_cores']==46
    combine = timer.parse_records(raw(operation='combine'), 'combine')
    assert all(row['role']=='send' for row in combine['records'] if row['core']==0)
    mixed=raw(); mixed[5][0][1]=5
    with pytest.raises(ValueError, match='Mixed timer schemas'): timer.parse_records(mixed, 'dispatch')


def test_expert_records_on_original_sender_require_shared_mode():
    data=[[[0]*8 for _ in range(8+16*4)] for _ in range(56)]
    for core, rows in enumerate(data): rows[0]=[expert.MAGIC,1,0,core,16,1,1,4]
    data[0][8]=[10,10,0,1,0,0,0,0]
    with pytest.raises(ValueError, match='sender core'): expert.parse_records(data,0,16,16,send_cores=10)
    report=expert.parse_records(data,0,16,16,send_cores=10,shared_receive=True)
    assert report['records'][0]['core']==0


def test_expert_stage_participation_must_agree(tmp_path):
    analyzer=load('shared_analyzer','tests/ll_stage_timer/analyze.py')
    data=[[[0]*8 for _ in range(8+16*4)] for _ in range(56)]
    for core, rows in enumerate(data): rows[0]=[expert.MAGIC,1,0,core,16,1,1,4]
    data[0][8]=[10,10,0,1,0,0,0,0]
    saved=dict(operation='dispatch',raw=raw(), metadata=dict(rank=0,world_size=16,iteration=0,config=dict(num_experts=16)),
               expert=dict(raw=data,rank=0,world=16,experts=16,send_cores=10,shared_receive=True))
    path=tmp_path/'sample.json'; path.write_text(__import__('json').dumps(saved))
    assert analyzer.expert_reports([path],1e9)[0][0]['core']==0
    saved['expert'].pop('shared_receive');path.write_text(__import__('json').dumps(saved))
    with pytest.raises(ValueError,match='participation'): analyzer.expert_reports([path],1e9)


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


def test_shared_protocol_comparison_preserves_partition_warning():
    left=parity.identity();left.update(core_protocol=7010,send_cores=10,recv_cores=46,stage_timer_schema=5)
    right=dict(left,implementation='g4-shared-recv',core_protocol=8010,stage_timer_schema=6,dispatch_recv_cores=56)
    parity.comparison.check_identity(left,right)
    def report(identity):
        return dict(comparison=identity,per_rank_core_stage=[dict(operation='dispatch',rank=0,core=10,stage='receiver',unit='us',mean=10)])
    row=parity.comparison.compare(report(left),report(right),'stage')[0]
    assert row['status']=='work_partition_changed' and row['candidate_over_baseline'] is None
    bad=dict(right,stage_timer_schema=5)
    with pytest.raises(ValueError,match='schema'): parity.comparison.check_identity(left,bad)


def test_production_wiring_keeps_producers_separate_from_consumers():
    base=ROOT/'csrc/kernels/dispatch/low_latency'
    kernel=(base/'dispatch_server_dedup_kernel.h').read_text(encoding='utf-8')
    counts=(base/'device/dispatch_server_dedup_counts.h').read_text(encoding='utf-8')
    receiver=(base/'device/dispatch_server_dedup_recv.h').read_text(encoding='utf-8')
    assert kernel.index('sender_.Process(')<kernel.index('counts_.CompleteCumsum();')<kernel.index('receiver_.Process(timer_);')
    assert kernel.count('receiver_.Process(timer_);')==1
    assert 'recvCoreNum_ = kLlDispatchRecvCoreNum;' in receiver
    assert 'receiver_ = GetBlockIdx();' in receiver
    assert 'aivId_ * aivUsedCumSum_ * UB_ALIGN' in counts
    assert 'for (uint32_t index = 0; index < coreNum_; index++)' in counts
    complete=counts[counts.index('inline void ServerDedupCounts::CompleteCumsum()'):]
    assert complete.index('PipeBarrier<PIPE_ALL>();')<complete.index('tpipe_->Reset();')<complete.index('WaitCumSumFlag();')

    assert complete.index("tpipe_->Reset();")<complete.index("SetCtrlSpr<60, 60>(0);")<complete.index("WaitCumSumFlag();")
