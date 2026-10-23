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

def raw():
    all_rows=[]
    for core in range(56):
        records=[];metrics={}
        def duration(name,start,end):
            stage=timer.STAGES.index(name)
            records.extend(((stage*2,start),(stage*2+1,end)))
            metrics[stage]=(1,end-start)
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
            duration('RECV_POLL',500,600)
            duration('RX_FLAG_WAIT',520,550)
            count('RX_PROBED',16);count('RX_READY',3);count('RX_PACKETS',3)
        records.extend((0x8000+stage+(calls<<32),ticks) for stage,(calls,ticks) in metrics.items())
        words=[0]*(timer.SCHEMA_SLOTS[13]*timer.FIELDS)
        words[:8]=[timer.MAGIC,13,1,56,core,1|(28<<8)|((3 if core<28 else 2)<<16),len(records),0]
        for i,(event,value) in enumerate(records):words[8+2*i:10+2*i]=[event,value]
        all_rows.append([words[i:i+8] for i in range(0,len(words),8)])
    return all_rows

def test_new_layout_names_and_end_to_end_units():
    header=(ROOT/'csrc/kernels/dispatch/low_latency/common/ll_stage_timer_layout.h').read_text()
    import re
    names=re.findall(r'"([A-Z_]+)"',re.search(r'kNames\[\] = \{(.*?)\};',header,re.S).group(1))
    assert names==list(timer.STAGES)
    assert timer.SCHEMA==15 and timer.SLOTS==8 and timer.METRICS==136
    assert timer.SCHEMA_SLOTS[13]==80 and timer.SCHEMA_METRICS[13]==108
    decoded=timer.parse_records(raw(),'dispatch',1e9)
    r=comm.report([(dict(rank=0,iteration=0),decoded)])
    send=r['samples'][0];receive=r['samples'][1];empty=r['samples'][2]
    assert send['before_first_send_ticks']==300 and send['before_first_send_us']==.3
    assert send['entry_to_first_send_ticks']==390 and send['entry_to_first_send_us']==.39
    assert send['send_setup_ticks']==100  # Includes Q stages; do not add them again.
    assert receive['ready_fraction']==3/16
    assert receive['probes_per_consumed_packet']==16/3
    assert empty['first_send_status']=='no_send' and empty['before_first_send_ticks'] is None
    assert empty['first_recv_status']=='no_receive'
    assert 'submission' in r['note'].lower()
    assert 'INIT至首发' in comm.markdown(r)
    decoded=timer.parse_records(raw(),'dispatch')
    assert comm.report([(dict(rank=0,iteration=0),decoded)])['samples'][0]['before_first_send_us'] is None

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
