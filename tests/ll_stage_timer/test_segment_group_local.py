"""Adapt schema7 coverage and labels to group-local initial sends on R workers."""
import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('group_segment_fixture',ROOT/'tests/ll_stage_timer/test_segments.py')
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)


def test_local_sender_role_is_inferred_from_actual_work_not_core_id():
    data=fixture.raw();core=10
    words=[v for row in data[core] for v in row];count=words[6]
    records=[words[8+2*i:10+2*i] for i in range(count)]
    split=next(i for i,pair in enumerate(records) if pair[0]&0xffffffff>=0x8000)
    records[split:split]=[[2,120],[3,180]]
    records.append([(1<<32)|0x8001,60])
    words[6]=len(records)
    for i,pair in enumerate(records):words[8+2*i:10+2*i]=pair
    data[core]=[words[i:i+8] for i in range(0,len(words),8)]
    report=fixture.timer.parse_records(data,'dispatch')
    local=report['cores'][core];empty=report['cores'][core+1]
    assert local['role']=='send_and_recv' and local['initial_role']=='cumsum'
    assert empty['role']=='cumsum' and empty['initial_role']=='cumsum'
    assert all(row['role']=='send_and_recv' for row in report['records'] if row['core']==core)
    assert all(row['initial_role']=='cumsum' for row in report['intervals'] if row['core']==core)
    # Binary compatibility is retained: reference format roles describe initial workers.
    binary=fixture.timer.reference_binary(data,'dispatch')
    assert fixture.struct.unpack_from('<3I',binary,64+core*64)[2]==2
    trace=fixture.timer.chrome_trace([dict(raw=data,operation='dispatch',frequency_hz=1e9)])
    lane=next(row for row in trace['traceEvents'] if row['ph']=='M' and row['tid']==core)
    assert 'send_and_recv' in lane['args']['name']


def test_group_local_preparation_and_retirement_have_separate_intervals():
    sender=(ROOT/'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h').read_text(encoding='utf-8')
    body=sender[sender.index('inline void ProcessGroupLocal('):sender.index('inline void SendWithDedup(')]
    assert body.index('if (!tasks) return;')<body.index('LL_SEG_SCOPE(SEND_TOKENS)')
    assert body.index('LL_SEG_BEGIN(localPrepareBegin, PREPARE_INPUT)')<body.index('scratch_->InitBuffer(input')
    assert body.index('LL_SEG_END(localPrepareBegin, PREPARE_INPUT)')<body.index('for (uint32_t task=0;task<tasks')
    assert body.index('LL_SEG_BEGIN(localDrainBegin, SEND_DRAIN)')<body.index('WaitFlag<HardEvent::MTE3_V>(lane)')
    assert body.index('scratch_->Reset()')<body.index('LL_SEG_END(localDrainBegin, SEND_DRAIN)')
    assert 'CAL_TOKEN' not in body
    kernel=(ROOT/'csrc/kernels/dispatch/low_latency/dispatch_server_dedup_kernel.h').read_text(encoding='utf-8')
    assert kernel.index('sender_.ProcessGroupLocal(')<kernel.index('LL_SEG_SCOPE(CAL_CUMSUM)')<kernel.index('counts_.RecvCnt()')


def test_nondedup_remote_filter_is_measured_as_preparation():
    sender=(ROOT/'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h').read_text(encoding='utf-8')
    body=sender[sender.index('inline void SendWithoutDedup('):sender.index('\nprivate:')]
    assert body.index('LL_SEG_BEGIN(remotePrepareBegin, PREPARE_INPUT)')<body.index('GatherMask(')
    assert body.index('GatherMask(')<body.index('LL_SEG_END(remotePrepareBegin, PREPARE_INPUT)')<body.index('SendPacket(')


def test_intermittent_local_work_does_not_split_kernel_statistics(tmp_path):
    analyzer=fixture.load('group_segment_analyzer','tests/ll_stage_timer/analyze.py')
    paths=[]
    for iteration,full in enumerate((False,True)):
        saved=dict(raw=fixture.raw(full=full),operation='dispatch',frequency_hz=1e9,
                   metadata=dict(rank=0,world_size=1,iteration=iteration,correctness='passed',config=dict(B=8)))
        path=tmp_path/f'sample_{iteration}.json';path.write_text(fixture.json.dumps(saved));paths.append(path)
    report=analyzer.analyze(paths,1e9)
    rows=[r for r in report['per_rank_core_stage'] if r['core']==10 and r['stage']=='KERNEL']
    assert len(rows)==1 and rows[0]['samples']==2
    assert rows[0]['role']=='send_and_recv' and rows[0]['initial_role']=='cumsum'
