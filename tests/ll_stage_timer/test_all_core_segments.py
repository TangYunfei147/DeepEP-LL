"""All-core send retirement vs business phase synchronization, without inferred gaps."""
import importlib.util
from pathlib import Path
import json
import pytest
ROOT=Path(__file__).resolve().parents[2]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
timer=load('phase_timer','deep_ep/utils/ll_stage_timer.py')

def raw(missing=False,overlap=False,extra_call=False,schema=8,operation='dispatch',empty=False):
    result=[]
    for core in range(56):
        spans=[(50,20,80),(31,200,220),(1,110,400)]
        if not missing:spans.append((31,250 if overlap else 410,440 if overlap else 600))
        spans.extend([(2,620,700),(0,100,1000)])
        metrics={50:(1,60),31:(3 if extra_call else 2,220 if extra_call else 210),1:(1,290),2:(1,80),0:(1,900)}
        if empty:
            spans=[row for row in spans if row[0]!=1 and row!=(31,200,220)]
            metrics.pop(1);metrics[31]=(1,190)
        events=[]
        for stage,begin,end in spans:events.extend([(stage*2,begin),(stage*2+1,end)])
        events.extend([((calls<<32)|0x8000|stage,total) for stage,(calls,total) in metrics.items()])
        words=[0]*(timer.SLOTS*8)
        role=(3 if schema==8 or core<10 else 2) if operation=='dispatch' else (1 if core<10 else 4)
        words[:8]=[timer.MAGIC,schema,1 if operation=='dispatch' else 2,56,core,1|(10<<8)|(role<<16),len(events),int(missing)+int(extra_call)]
        for i,pair in enumerate(events):words[8+i*2:10+i*2]=pair
        result.append([words[i:i+8] for i in range(0,len(words),8)])
    return result

def test_two_drains_are_distinguished_using_actual_parent_interval():
    report=timer.parse_records(raw(),'dispatch',1e9)
    part=report['send_phase_parts'][0]
    assert part['sampled_complete'] and part['sender_lane_drain_ticks']==20 and part['send_phase_sync_ticks']==190
    assert part['send_phase_sync_us']==pytest.approx(.19)
    intervals=[r for r in report['intervals'] if r['core']==0 and r['stage']=='SEND_DRAIN']
    assert [r['detail'] for r in intervals]==['sender_lane_drain','send_phase_drain_sync']
    metric=next(r for r in report['records'] if r['core']==0 and r['stage']=='SEND_DRAIN')
    assert metric['total_ticks']==210 and metric['calls']==2
    trace=timer.chrome_trace([dict(raw=raw(),operation='dispatch',frequency_hz=1e9)])
    assert any(e['name']=='SEND_DRAIN [send_phase_drain_sync]' for e in trace['traceEvents'])

@pytest.mark.parametrize('damage',['missing','overlap','extra_call'])
def test_partial_or_unexpected_samples_are_not_reconstructed(damage):
    report=timer.parse_records(raw(**{damage:True}),'dispatch')
    assert not report['send_phase_parts'][0]['sampled_complete']
    assert report['send_phase_parts'][0]['send_phase_sync_ticks'] is None
    assert all(not r['detail'] for r in report['intervals'])

@pytest.mark.parametrize('schema,operation',[(7,'dispatch'),(8,'combine')])
def test_other_schedules_do_not_get_all_core_phase_labels(schema,operation):
    report=timer.parse_records(raw(schema=schema,operation=operation),operation)
    assert not report['send_phase_parts']
    assert all(not r['detail'] for r in report['intervals'])

def test_phase_csv_and_prepare_receive_probe(tmp_path):
    analyzer=load('phase_analyze','tests/ll_stage_timer/analyze.py')
    source=tmp_path/'input';source.mkdir()
    (source/'sample_0.json').write_text(json.dumps(dict(raw=raw(),operation='dispatch',frequency_hz=1e9,
        metadata=dict(rank=0,world_size=1,iteration=0,config=dict(B=8),correctness='passed'))))
    output=tmp_path/'output';analyzer.process_directory(source,output,1e9,0)
    assert (output/'send_phase.csv').is_file()
    counts=(ROOT/'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_counts.h').read_text(encoding='utf-8')
    body=counts[counts.index('inline void PrepareReceive()'):counts.index('// Valid after CompleteCumsum')]
    assert body.index('LL_SEG_SCOPE(INIT_COUNT_BUFFERS)')<body.index('tpipe_->Reset()')<body.index('SetCtrlSpr<60,60>(0)')


def test_empty_sender_still_reports_the_business_phase_barrier():
    report=timer.parse_records(raw(empty=True),'dispatch')
    part=report['send_phase_parts'][0]
    assert part['sampled_complete'] and not part['sender_executed']
    assert part['sender_lane_drain_ticks'] is None and part['send_phase_sync_ticks']==190
    drains=[r for r in report['intervals'] if r['stage']=='SEND_DRAIN']
    assert all(r['detail']=='send_phase_drain_sync' for r in drains)
