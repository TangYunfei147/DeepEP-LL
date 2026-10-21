"""Packet counts, missing-data semantics and weighted per-packet timing."""
import importlib.util
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[2]
def load(name, path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
packet=load('packets','tests/ll_stage_timer/packet_timing.py')
fixtures=load('packet_fixtures','tests/ll_stage_timer/test_segments.py')
timer=fixtures.timer

def sample(iteration=0, sent=4, recv=6, first=1, terminal=3, ticks=80, schema=14):
    records=[]
    for name,calls,total in [('KERNEL',1,1000),('INITIAL_PACKETS',sent,0),
                             ('RECV_READY_TOKENS',recv,0),('RECV_FIRST_HOP',first,0),
                             ('RECV_TERMINAL',terminal,0),('RELAY_PACKETS',3,0),
                             ('SEND_PACKET',sent,ticks)]:
        if calls:
            records.append(dict(core=0,stage=name,calls=calls,total_ticks=total,average_ticks=total/calls))
    return dict(rank=2,iteration=iteration), dict(schema=schema,operation='dispatch',boundary_mode='observe',records=records)

def test_counts_weighting_and_unmeasured_empty_sender(tmp_path):
    result=packet.report([sample(),sample(1,sent=1,ticks=100)])
    row=result['per_core'][0]
    assert row['sent_total']==5 and row['sent_per_round']==2.5
    assert row['SEND_PACKET_per_call_ticks']==36  # 180/5, not mean(20,100).
    assert row['received_direct_total']==4
    assert row['received_forwarded_total']==6
    assert row['STATIC_BEFORE_SEND_mean_ticks'] is None
    packet.write(tmp_path,result)
    text=(tmp_path/'dispatch_packets.md').read_text(encoding='utf-8')
    assert '2.50' in text and '不是网络到达延迟' in text
    assert (tmp_path/'dispatch_packet_samples.csv').is_file()
    empty=packet.report([sample(sent=0,ticks=0)])['per_core'][0]
    assert empty['sent_total']==0 and empty['SEND_PACKET_per_call_ticks'] is None

def test_legacy_missing_and_invalid_counts():
    assert not packet.report([sample(schema=13)])['samples']
    with pytest.raises(ValueError,match='classes exceed'):
        packet.report([sample(recv=2)])

def test_coverage_and_trace_advertise_partial_metrics():
    raw=fixtures.raw(schema=14)
    # Change one sampled poll interval stage and its metric to TOKEN_INPUT.
    # Two intervals plus 3 calls still parse, but explicitly carry sample-only
    # coverage rather than claiming a whole-kernel input wait share.
    for rows in raw:
        for i,row in enumerate(rows):
            if i==0:continue
            for j in range(0,8,2):
                event=row[j]&0xffffffff
                if event in (58,59): row[j]=32+(event&1)
                elif event==0x8000+29: row[j]=(row[j]&~0xffffffff)|(0x8000+16)
    decoded=timer.parse_records(raw,'dispatch')
    assert all(r['coverage']=='first_two_only' for r in decoded['records'] if r['stage']=='TOKEN_INPUT')
    trace=timer.chrome_trace([dict(raw=raw,operation='dispatch',frequency_hz=1e9)])
    assert not trace['metadata']['all_calls_in_metrics']
    old=timer.parse_records(fixtures.raw(schema=13),'dispatch')
    assert all(r['coverage']=='all_calls' for r in old['records'])

def test_counts_read_original_type_and_output_excludes_relay():
    source=(ROOT/'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_recv.h').read_text(encoding='utf-8')
    assert source.index('LL_SEG_COUNT(RECV_TERMINAL') < source.index('PrepareServerDedupForward(tail)')
    assert source.index('LL_SEG_BEGIN(finalOutputBegin, OUTPUT_ISSUE)') > source.index('LL_SEG_COUNT(RELAY_PACKETS')
    assert 'LL_SEG_BEGIN(segBegin3, RECV_INPUT_WAIT);\n        WaitFlag<HardEvent::MTE2_S>(lane);' in source


def test_input_wait_uses_full_packet_count_not_sparse_input_count():
    analyze=load('packet_analyze','tests/ll_stage_timer/analyze.py')
    samples=[]
    for iteration in range(3):
        meta,decoded=sample(iteration)
        for stage,calls,ticks in [('SEND_TOKENS',1,800),('TOKEN_INPUT',2,10),
                                 ('TOKEN_QUANT',2,20),('TOKEN_QUANT_WAIT',4,200)]:
            decoded['records'].append(dict(core=0,stage=stage,calls=calls,total_ticks=ticks))
        samples.append((meta,decoded))
    result=analyze.input_wait_report(samples)
    assert result['status']=='疑似瓶颈'
    assert result['per_core'][0]['weighted_ticks_per_call']==50
