"""Same-core arithmetic, sparse validity, cumulative scopes and raw int64 bits."""
import importlib.util
from pathlib import Path
import re

import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('systemcycle_parser',ROOT/'deep_ep/utils/ll_stage_timer_systemcycle.py')
timer=importlib.util.module_from_spec(spec);spec.loader.exec_module(timer)


def raw(operation='dispatch',rank=0,iteration=0,send=30):
    data=[];op=1 if operation=='dispatch' else 2
    for core in range(56):
        # Large, unrelated epochs ensure no float conversion or cross-core clock subtraction.
        origin=(1<<63)+rank*1_000_000_000+core*1_000_000+iteration*100_000
        points=dict(ENTRY=1000,ENTRY_SETUP_DONE=1100,MAGIC_DONE=1300,INIT_DONE=1500,WORK_DONE=3800)
        if op==1:
            points.update(LAYOUT_DONE=1200,COUNTS_INIT_DONE=1400,COUNTS_BEGIN=1510,
                SOURCE_BEGIN=1600,SEND_PREPARED=1800,FIRST_SEND=1900,SOURCE_SUBMIT_DONE=2400,SOURCE_RETURN=2500,
                COMPLETE_PREFIX_BEGIN=2600,PREFIX_WAIT_ENTER=2610,PREFIX_WAIT_EXIT=2650,PREFIX_READY=2660,
                COMPLETE_PREFIX_RETURN=2700,RECV_BEGIN=2800,RECV_TASKS_DONE=3000,RECV_BUFFERS_DONE=3100,
                RECV_PREPARED=3110,FIRST_RECV_PROBE=3120,FIRST_RECV_READY=3200,RECV_PENDING_EMPTY=3500,
                RECV_DONE=3600,RETIRE_BEGIN=3610,RETIRE_END=3620,FINISH_BEGIN=3630,SEND_DONE=3700)
            if core>=send:points['COUNTS_SENT']=1550
            if core==send:
                points.update(COUNTS_WAIT_BEGIN=1560,COUNTS_RECEIVED=1570,PREFIX_BEGIN=1580,
                              PREFIX_LOCAL_DONE=1585,PREFIX_RETURN=1590)
        else:
            points.update(COMBINE_PREPARE_BEGIN=1600,COMBINE_PREPARE_END=1700,
                          COMBINE_ROLE_BEGIN=1800,COMBINE_ROLE_END=3500)
        role=(3 if core<send else 2) if op==1 else (1 if core<send else 4)
        words=[0]*128
        words[:8]=[timer.MAGIC,18,op,56,core,1|(send<<8)|(role<<16)|(9<<24),0,
                   6|(6<<8)|(1<<16) if op==1 else 0]
        for name,value in points.items():
            index=timer.POINTS.index(name);words[6]|=1<<index;words[8+index]=origin+value
        for index in timer.METRIC_OPERATIONS[operation]:
            words[7]|=1<<(24+index)
            words[48+2*index:50+2*index]=[0,core%3] if index in timer.COUNTER_INDICES else [20+index,2]
        # Torch copies uint64 device bits through an int64 carrier.
        words=[word-(1<<64) if word>=(1<<63) else word for word in words]
        data.append([words[i:i+8] for i in range(0,128,8)])
    return data


def word(data,core,index):return data[core][index//8][index%8]&((1<<64)-1)


def setword(data,core,index,value):data[core][index//8][index%8]=value


@pytest.mark.parametrize('operation',('dispatch','combine'))
def test_same_core_high_precision_ticks_and_only_main_span_timeline(operation):
    result=timer.parse_records(raw(operation),'dispatch' if operation=='dispatch' else 'combine',1e9)
    total=next(row for row in result['records'] if row['core']==0 and row['stage']=='ENTRY_TO_WORK_DONE')
    assert total['total_ticks']==2800 and total['total_us']==2.8
    assert all(row['first_tick']>1<<63 for row in result['intervals'])
    assert not {row['stage'] for row in result['intervals']} & set(timer.METRIC_NAMES)
    assert result['timer_profile_id']==9 and result['recording']=='systemcycle-stages'
    assert all(row['calls']==2 for row in result['cycle_metrics'] if row['time_measured'])


def test_valid_zero_is_distinct_from_unobserved_or_inapplicable():
    data=raw('combine');result=timer.parse_records(data,'combine',1e9)
    counter=next(row for row in result['cycle_metrics'] if row['core']==0 and row['stage']=='COMBINE_READY_SLOTS')
    assert word(data,0,7)&(1<<63) and data[0][0][7]<0
    assert counter['calls']==0 and counter['value']==0 and not counter['time_measured']
    assert not any(row['stage']=='SOURCE' for row in result['records'])
    setword(data,0,7,word(data,0,7)&~(1<<63))
    result=timer.parse_records(data,'combine')
    assert not any(row['core']==0 and row['stage']=='COMBINE_READY_SLOTS' for row in result['cycle_metrics'])


def test_missing_phase_endpoint_is_not_zero_duration():
    result=timer.parse_records(raw(),'dispatch',1e9)
    assert not any(row['core']==0 and row['stage']=='COUNT_SEND' for row in result['records'])
    assert any(row['core']==30 and row['stage']=='COUNT_SEND' for row in result['records'])


def test_no_frequency_keeps_ticks_without_inventing_microseconds():
    result=timer.parse_records(raw(),'dispatch')
    assert all('total_us' not in row for row in result['records'])


@pytest.mark.parametrize('fault',('shape','schema','mode','profile','role','reserved','point_bitmap','metric_bitmap',
    'point_outside','point_order','counter_time','duration_calls','duration_too_big','foreign_metric'))
def test_corrupt_or_mismatched_records_are_rejected(fault):
    data=raw()
    if fault=='shape':data[0].pop()
    elif fault=='schema':setword(data,0,1,17)
    elif fault in ('mode','profile','role'):
        value=word(data,0,5)
        shift={'mode':0,'profile':24,'role':16}[fault]
        value=(value&~(255<<shift))|({'mode':6,'profile':8,'role':2}[fault]<<shift)
        setword(data,0,5,value)
    elif fault=='reserved':setword(data,0,7,word(data,0,7)|(1<<17))
    elif fault=='point_bitmap':setword(data,0,6,word(data,0,6)&~1)
    elif fault=='metric_bitmap':setword(data,0,7,word(data,0,7)&~(1<<24))
    elif fault=='point_outside':setword(data,0,8+timer.POINTS.index('MAGIC_DONE'),word(data,0,8)-1)
    elif fault=='point_order':setword(data,0,8+timer.POINTS.index('SOURCE_RETURN'),word(data,0,8)+599)
    elif fault=='counter_time':setword(data,0,48+34*2,1)
    elif fault=='duration_calls':setword(data,0,49,0)
    elif fault=='duration_too_big':setword(data,0,48,2801)
    elif fault=='foreign_metric':
        setword(data,0,7,word(data,0,7)|(1<<(24+22)));setword(data,0,48+22*2,10);setword(data,0,49+22*2,1)
    with pytest.raises(ValueError):timer.parse_records(data,'dispatch',1e9)


def test_different_selected_scopes_may_overlap_and_are_not_summed():
    data=raw()
    # Parent and child overlap, but each fits within the recorded same-core total.
    setword(data,0,48,2000);setword(data,0,50,2000)
    result=timer.parse_records(data,'dispatch')
    assert sum(row['total_ticks'] for row in result['cycle_metrics'] if row['core']==0)>2800


@pytest.mark.parametrize('frequency',(0,-1,True,float('nan'),float('inf')))
def test_bad_frequency_rejected(frequency):
    with pytest.raises(ValueError):timer.parse_records(raw(),'dispatch',frequency)


def test_native_dictionary_and_python_schema_agree():
    source=(ROOT/'csrc/kernels/dispatch/low_latency/common/ll_stage_timer_layout.h').read_text()
    for symbol,expected in (('kPointNames',timer.POINTS),('kCycleMetricNames',timer.METRIC_NAMES)):
        match=re.search(symbol+r'\[\]\s*=\s*\{(.*?)\};',source,re.S)
        assert tuple(re.findall(r'"([A-Z_]+)"',match.group(1)))==expected
    body=re.search(r'kCycleMetricStages\[\]\s*=\s*\{(.*?)\};',source,re.S).group(1)
    assert tuple(re.findall(r'\b[A-Z][A-Z_]+\b',body))==timer.METRIC_NAMES
