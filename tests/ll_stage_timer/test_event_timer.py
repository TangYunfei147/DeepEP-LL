"""Actual occurrence timing, strict validation, compact extraction and trace ordering."""
import copy
import importlib.util
from pathlib import Path
import runpy

import pytest
from event_fixtures import raw, set_events, timer

ROOT=Path(__file__).resolve().parents[2]


def alternating(mode=3):
    data=raw(mode=mode)
    for core in range(16):
        set_events(data,core,'recv_metadata',[[110,120],[200,230]])
        set_events(data,core,'recv_output_write',[[120,160],[230,280]])
    return data


def test_occurrences_produce_real_order_and_offline_totals():
    data=alternating()
    report=timer.parse_records(data,'dispatch',1e9)
    metadata=next(row for row in report['records'] if row['stage']=='recv_metadata')
    assert metadata['calls']==2 and metadata['total_ticks']==40
    assert metadata['last_tick']-metadata['first_tick']==120
    assert report['boundary_mode']=='mega'
    trace=timer.chrome_trace([dict(raw=data,operation='dispatch',frequency_hz=1e9)])
    events=[row for row in trace['traceEvents'] if row['ph']=='X' and row['tid']==0
            and row['name'] in ('recv_metadata','recv_output_write')]
    assert [row['name'] for row in events]==['recv_metadata','recv_output_write']*2
    assert [row['dur'] for row in events]==pytest.approx([.01,.04,.03,.05])
    assert all(a['ts']+a['dur']<=b['ts']+1e-12 for a,b in zip(events,events[1:]))
    assert all(row['args']['interval_kind']=='actual occurrence' for row in events)


@pytest.mark.parametrize('damage',['header','overflow','unclosed','missing_end','order','outside','count','unknown_stage','read','mode'])
def test_invalid_event_data_never_silently_truncates(damage):
    data=alternating();core=data['cores'][0]
    if damage=='header':core['header'][6]=0
    if damage=='overflow':core['errors']=1
    if damage=='unclosed':core['errors']=16
    if damage=='missing_end':core['events']['recv_metadata'][0]=[110]
    if damage=='order':core['events']['recv_metadata'][1]=[115,130]
    if damage=='outside':core['events']['recv_metadata'][1]=[999,1001]
    if damage=='count':core['counts'][timer.STAGES.index('recv_metadata')-1]=3
    if damage=='unknown_stage':core['events']['ghost']=[]
    if damage=='read':core['traffic'][1]=[10,1,11,1]
    if damage=='mode':core['header'][5]=9
    with pytest.raises(ValueError):timer.parse_records(data,'dispatch',1e9)


@pytest.mark.parametrize('schema',[1,2,3])
def test_old_formats_are_removed(schema):
    with pytest.raises(ValueError,match='schema 4'):
        timer.parse_records(dict(schema=schema,cores=[]),'dispatch')
    with pytest.raises(ValueError,match='schema 4'):
        timer.parse_records([[[0]*8]*32]*16,'dispatch')


def test_fixed_and_dynamic_limits_and_wait_policy():
    data=raw()
    set_events(data,0,'magic',[[110,120],[130,140]])
    with pytest.raises(ValueError,match='capacity'):
        timer.parse_records(data,'dispatch')
    assert timer.CAPACITIES[timer.STAGES.index('recv_metadata')-1]==4096
    assert timer.CAPACITIES[timer.STAGES.index('recv_prepare')-1]==4
    assert timer.EVENT_WORDS*8%512==0
    device=(ROOT/'csrc/kernels/dispatch/low_latency/common/ll_stage_timer_device.h').read_text()
    timing=device.split('inline void Start(',1)[1].split('inline void CountData(',1)[0]
    assert 'total_ticks' not in timing and 'row + 2U' not in timing
    assert 'mode_==3U && !DependencyWait(stage)' in device


def test_wait_splits_one_task_without_parent_nesting():
    data=raw(mode=3)
    for core in range(16):
        set_events(data,core,'combine_reduce_task',[[110,120,7],[140,160,7]])
        set_events(data,core,'combine_data_wait',[[120,140,7]])
    report=timer.parse_records(data,'dispatch',1e9)
    task=next(row for row in report['records'] if row['core']==0 and row['stage']=='combine_reduce_task')
    assert task['calls']==1 and task['segments']==2 and task['total_ticks']==30
    trace=timer.chrome_trace([dict(raw=data,operation='dispatch',frequency_hz=1e9)])
    work=[row for row in trace['traceEvents'] if row['ph']=='X' and row['tid']==0]
    assert [row['name'] for row in work]==['combine_reduce_task','combine_data_wait','combine_reduce_task']
    assert all(row['args']['task_id']==7 for row in work)
    assert next(row for row in trace['traceEvents'] if row.get('name')=='kernel' and row['ph']=='X')['tid']==10000


def test_nested_task_events_rejected():
    data=alternating()
    set_events(data,0,'send_route',[[105,290]])
    with pytest.raises(ValueError,match='Overlapping'):
        timer.parse_records(data,'dispatch')


def test_mode_choices_and_wheel_guard():
    script=runpy.run_path(str(ROOT/'tests/elastic/test_ep_low_latency.py'))
    assert script['parse_args']([]).ll_timer_mode=='mega'
    script['validate_stage_timer_capability'](True,dict(stage_timer=True,stage_timer_schema=4))
    with pytest.raises(RuntimeError,match='schema 4'):
        script['validate_stage_timer_capability'](True,dict(stage_timer=True,stage_timer_schema=3))


def test_compact_extraction_reads_only_written_slots():
    # Simulates dense device addressing without allocating any zero-filled event arrays.
    data=alternating(mode=1)
    class Slice:
        def __init__(self,values):self.values=values;self.columns=None
        def view(self,rows,columns):self.columns=columns;return self
        def tolist(self):
            return self.values if self.columns is None else [self.values[n:n+self.columns] for n in range(0,len(self.values),self.columns)]
    class Dense:
        shape=(16,timer.SLOTS,8)
        def view(self,*shape):return self
        def __getitem__(self,key):
            core,part=key;entry=data['cores'][core]
            if part.start is None and part.stop==128:
                base=entry['header']+entry['counts']+[0]*(64-8-len(entry['counts']))
                return Slice(base+[value for row in entry['traffic'] for value in row]+[0]*44)
            stage=timer.EVENT_OFFSETS.index(part.start)+1
            return Slice([value for pair in entry['events'][timer.STAGES[stage]] for value in pair])
    compact=timer.compact_device_tensor(Dense())
    assert compact==data
    assert timer.parse_records(compact,'dispatch')['records']
