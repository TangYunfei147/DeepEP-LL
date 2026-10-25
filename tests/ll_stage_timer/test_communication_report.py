"""Schema13 end-to-end preparation/communication records and report invariants."""
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
timer=load('comm_timer','deep_ep/utils/ll_stage_timer.py')
comm=load('comm_report','tests/ll_stage_timer/communication_report.py')
source_checks=load('comm_source','tests/ll_stage_timer/test_segment_source.py')

def raw(poll_mode='serial',snapshots=1):
    all_rows=[]
    for core in range(56):
        records=[];metrics={}
        def duration(name,start,end):
            stage=timer.STAGES.index(name)
            calls,total=metrics.get(stage,(0,0))
            if calls<2:records.extend(((stage*2,start),(stage*2+1,end)))
            metrics[stage]=(calls+1,total+end-start)
        def count(name,value):metrics[timer.STAGES.index(name)]=(value,0)
        duration('INIT',10,90)
        duration('KERNEL',100,1100)
        if core==0:
            duration('BEFORE_FIRST_SEND',100,400)
            duration('SEND_SETUP',110,210)
            duration('Q_SEED_SUBMIT',130,140)
            duration('Q_SEED_WAIT',140,160)
            duration('Q_LOCAL_UPDATE',160,180)
            count('TX_PACKETS',2)
        if core==1:
            duration('BEFORE_FIRST_RECV_POLL',100,500)
            for snapshot in range(snapshots):
                start=500+snapshot*150
                duration('RX_FLAG_ISSUE',start,start+20)
                duration('RX_FLAG_WAIT',start+20,start+50)
                if poll_mode=='split':
                    duration('RX_FLAG_REDUCE',start+50,start+70)
                    duration('RECV_POLL',start,start+70)
                    duration('RX_FLAG_REDUCE',start+90,start+92)
                    duration('RX_FLAG_DECODE',start+92,start+99)
                    duration('RECV_POLL',start+90,start+100)
                else:
                    assert poll_mode=='serial'
                    duration('RX_FLAG_REDUCE',start+50,start+80)
                    duration('RX_FLAG_DECODE',start+80,start+95)
                    duration('RECV_POLL',start,start+100)
            count('RX_PROBED',16*snapshots);count('RX_READY',3);count('RX_PACKETS',3)
            if snapshots>1:count('RX_EMPTY_POLLS',snapshots-1)
        omitted=sum(calls for stage,(calls,_) in metrics.items() if stage not in timer.COUNTERS)-len(records)//2
        records.extend((0x8000+stage+(calls<<32),ticks) for stage,(calls,ticks) in metrics.items())
        words=[0]*(80*timer.FIELDS)  # Frozen schema13 offline fixture.
        words[:8]=[timer.MAGIC,13,1,56,core,1|(28<<8)|((3 if core<28 else 2)<<16),len(records),omitted]
        for i,(event,value) in enumerate(records):words[8+2*i:10+2*i]=[event,value]
        all_rows.append([words[i:i+8] for i in range(0,len(words),8)])
    return all_rows

def test_new_layout_names_and_end_to_end_units():
    header=(ROOT/'csrc/kernels/dispatch/low_latency/common/ll_stage_timer_layout.h').read_text()
    import re
    dictionary=re.search(r'kNames\[\]\s*=\s*\{(.*?)\};',header,re.S).group(1)
    names=re.findall(r'"([A-Z_]+)"',dictionary)
    assert names==list(timer.STAGES)
    assert timer.SCHEMA_SLOTS[13]==80 and timer.SCHEMA_METRICS[13]==108
    decoded=timer.parse_records(raw(),'dispatch',1e9)
    r=comm.report([(dict(rank=0,iteration=0),decoded)])
    send=r['samples'][0];receive=r['samples'][1];empty=r['samples'][2]
    assert send['before_first_send_ticks']==300 and send['before_first_send_us']==.3
    assert send['entry_to_first_send_ticks']==390 and send['entry_to_first_send_us']==.39
    assert send['send_setup_ticks']==100  # Includes Q stages; do not add them again.
    assert receive['ready_fraction']==3/16
    assert receive['probes_per_consumed_packet']==16/3
    assert receive['poll_snapshots']==receive['poll_batches']==receive['recv_poll_segments']==1
    assert receive['empty_poll_fraction']==0 and receive['poll_snapshot_status']=='measured'
    assert empty['first_send_status']=='no_send' and empty['before_first_send_ticks'] is None
    assert empty['first_recv_status']=='no_receive'
    assert empty['poll_snapshots']==0 and empty['empty_poll_fraction'] is None
    assert empty['poll_snapshot_status']=='no_receive'
    assert 'submission' in r['note'].lower()
    assert 'INIT至首发' in comm.markdown(r)
    decoded=timer.parse_records(raw(),'dispatch')
    assert comm.report([(dict(rank=0,iteration=0),decoded)])['samples'][0]['before_first_send_us'] is None

@pytest.mark.parametrize('poll_mode,segments',[('serial',2),('split',4)])
def test_empty_snapshot_rate_uses_completed_decodes(poll_mode,segments):
    decoded=timer.parse_records(raw(poll_mode,snapshots=2),'dispatch',1e9)
    result=comm.report([(dict(rank=0,iteration=0),decoded)])
    receive=result['samples'][1]
    assert receive['recv_poll_segments']==segments
    assert receive['poll_snapshots']==receive['poll_batches']==2
    assert receive['rx_empty_polls']==1 and receive['empty_poll_fraction']==.5
    assert receive['poll_snapshot_status']=='measured'
    assert result['per_core'][1]['empty_poll_fraction_p50']==.5
    reduce=next(r for r in result['costs'] if r['core']==1 and r['stage']=='RX_FLAG_REDUCE')
    assert reduce['calls']==segments
    assert 'corrected compatibility alias' in result['note']
    assert '旧报表poll_batches' in comm.markdown(result)

@pytest.mark.parametrize('missing',[('RX_FLAG_DECODE',),('RX_FLAG_ISSUE','RX_FLAG_WAIT','RX_FLAG_DECODE')])
def test_missing_snapshot_denominator_is_explicitly_unavailable(missing):
    decoded=timer.parse_records(raw('split',snapshots=2),'dispatch',1e9)
    decoded['records']=[r for r in decoded['records'] if r['stage'] not in missing]
    receive=comm.report([(dict(rank=0,iteration=0),decoded)])['samples'][1]
    assert receive['recv_poll_segments']==4
    assert receive['poll_snapshots'] is None and receive['poll_batches'] is None
    assert receive['empty_poll_fraction'] is None
    assert receive['poll_snapshot_status']=='missing_flag_snapshot_metrics'

@pytest.mark.parametrize('stage,calls',[
    ('RX_FLAG_ISSUE',1),('RX_FLAG_WAIT',3),('RX_FLAG_DECODE',1),
    ('RX_FLAG_DECODE',0),('RX_EMPTY_POLLS',3),('RX_EMPTY_POLLS',-1)])
def test_inconsistent_snapshot_metrics_rejected(stage,calls):
    decoded=timer.parse_records(raw('split',snapshots=2),'dispatch',1e9)
    next(r for r in decoded['records'] if r['core']==1 and r['stage']==stage)['calls']=calls
    with pytest.raises(ValueError,match='snapshot|RX_EMPTY_POLLS'):
        comm.report([(dict(rank=0,iteration=0),decoded)])

@pytest.mark.parametrize('missing',['RX_FLAG_ISSUE','RX_FLAG_WAIT'])
def test_decode_without_matching_issue_or_wait_rejected(missing):
    decoded=timer.parse_records(raw('split',snapshots=2),'dispatch',1e9)
    decoded['records']=[r for r in decoded['records'] if r['stage']!=missing]
    with pytest.raises(ValueError,match='snapshot calls'):
        comm.report([(dict(rank=0,iteration=0),decoded)])

def test_missing_decode_does_not_hide_inconsistent_available_metrics():
    decoded=timer.parse_records(raw('split',snapshots=2),'dispatch',1e9)
    decoded['records']=[r for r in decoded['records'] if r['stage']!='RX_FLAG_DECODE']
    next(r for r in decoded['records'] if r['core']==1 and r['stage']=='RX_FLAG_WAIT')['calls']=1
    with pytest.raises(ValueError,match='snapshot calls'):
        comm.report([(dict(rank=0,iteration=0),decoded)])

def test_missing_envelope_and_counter_mismatch_rejected():
    for missing in ('BEFORE_FIRST_SEND','RX_READY'):
        decoded=timer.parse_records(raw(),'dispatch',1e9)
        decoded['records']=[r for r in decoded['records'] if r['stage']!=missing]
        with pytest.raises(ValueError):comm.report([(dict(rank=0,iteration=0),decoded)])

def test_new_report_does_not_invent_old_measurements():
    decoded=timer.parse_records(raw(),'dispatch',1e9);decoded['schema']=12
    result=comm.report([(dict(rank=0,iteration=0),decoded)])
    assert result['status']=='requires_schema13_dispatch' and not result['samples']

def test_observe_business_tokens_match_recorded_candidate():
    if timer.SCHEMA != 13:
        pytest.skip('Historical schema13 source snapshot; current ed1 business equivalence is checked by test_sparse_control_native')
    manifest=json.loads((Path(__file__).with_name('communication_baseline.json')).read_text())
    for relative,digest in manifest['files'].items():
        tokens=source_checks.normalize((ROOT/relative).read_text(encoding='utf-8'))
        assert hashlib.sha256(' '.join(tokens).encode()).hexdigest()==digest,relative

def test_analyze_exports_new_report(tmp_path):
    analyzer=load('comm_analyze','tests/ll_stage_timer/analyze.py')
    path=tmp_path/'sample_0000.json'
    meta=dict(rank=0,iteration=0,world_size=1,correctness='passed',
              config=dict(B=256,M=256,H=3584,num_topk=8,num_experts=384,dtype='fp8'))
    path.write_text(json.dumps(dict(raw=raw(),operation='dispatch',frequency_hz=1e9,metadata=meta)))
    result=analyzer.analyze([path],None)
    assert result['communication']['status']=='available'
    assert result['communication']['samples'][0]['before_first_send_us']==.3
    output=tmp_path/'report'
    analyzer.process_directory(tmp_path,output,1e9,0)
    for name in ('communication_bottleneck.md','communication_samples.csv','communication_costs.csv','communication_per_core.csv'):
        assert (output/name).is_file()
    trace=timer.chrome_trace([dict(raw=raw(),operation='dispatch',frequency_hz=1e9,metadata=meta)])
    assert any(r.get('name')=='BEFORE_FIRST_SEND' and r.get('ph')=='X' for r in trace['traceEvents'])

def test_launcher_identity_matches_current_host():
    descriptor=json.loads((ROOT/'tests/ll_stage_timer/branch.json').read_text())
    host=(ROOT/'csrc/ops/low_latency.cpp').read_text()
    assert f'module.attr("low_latency_implementation") = "{descriptor["implementation"]}"' in host

def test_mixed_diagnostic_schemas_rejected_for_current_protocol():
    parity=load('comm_parity','tests/ll_stage_timer/test_parity.py')
    left=parity.identity()
    left.update(core_protocol=19028,send_cores=28,recv_cores=28,active_aiv=56,stage_timer_schema=13)
    right=dict(left,stage_timer_schema=12)
    with pytest.raises(ValueError,match='different stage schemas'):
        parity.comparison.check_identity(left,right)
