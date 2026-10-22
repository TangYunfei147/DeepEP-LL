"""Reference sampling, integrity, late export, native guards and interval attribution."""
import copy
import importlib.util
import json
from pathlib import Path
import re
import struct
import sys
from types import ModuleType, SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[2]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

timer=load('segments','deep_ep/utils/ll_stage_timer.py')


def raw(send=10,operation='dispatch',full=False,schema=7,mode=1):
    result=[]
    for core in range(56):
        intervals=[];metrics={};omitted=0
        def end(stage,begin,finish):
            nonlocal omitted
            calls,total=metrics.get(stage,(0,0));calls+=1;metrics[stage]=(calls,total+finish-begin)
            if calls<=2 and len(intervals)+2<=192-(0 if stage==0 else 2):
                intervals.extend([(2*stage,begin),(2*stage+1,finish)])
            else:omitted+=1
        end(50,10,20)
        if full:
            for stage in range(1,{7:64,9:66,10:70}[schema]):
                if stage in timer.COUNTERS: metrics[stage]=(100,0);continue
                for sample in range(2):end(stage,200+stage*20+sample*5,203+stage*20+sample*5)
        else:
            for begin,finish in [(200,203),(300,307),(500,511)]:end(29,begin,finish)
            metrics[52]=(17,0)
        end(0,100,10000)
        events=intervals+[(0x8000+stage+(calls<<32),total) for stage,(calls,total) in sorted(metrics.items())]
        words=[0]*((timer.SLOTS if schema==timer.SCHEMA else 72)*timer.FIELDS)
        role=(3 if core<send else 2) if operation=='dispatch' else (1 if core<send else 4)
        words[:8]=[timer.MAGIC,schema,1 if operation=='dispatch' else 2,56,core,mode|(send<<8)|(role<<16),len(events),omitted]
        for i,(event,value) in enumerate(events):words[8+i*2:10+i*2]=[event,value]
        result.append([words[i:i+8] for i in range(0,len(words),8)])
    return result


def test_first_two_actual_intervals_and_all_call_metrics():
    report=timer.parse_records(raw(),'dispatch',1e9)
    intervals=[r for r in report['intervals'] if r['core']==0 and r['stage']=='RECV_POLL']
    assert [(r['first_tick'],r['last_tick']) for r in intervals]==[(200,203),(300,307)]
    metric=next(r for r in report['records'] if r['core']==0 and r['stage']=='RECV_POLL')
    assert metric['calls']==3 and metric['total_ticks']==21 and metric['average_ticks']==7
    counter=next(r for r in report['records'] if r['stage']=='POLL_RECV_COUNTS_ITERATIONS')
    assert counter['calls']==17 and counter['total_ticks']==0 and not counter['time_measured']
    assert all(r['omitted_intervals']==1 for r in report['cores'])
    trace=timer.chrome_trace([dict(raw=raw(),operation='dispatch',frequency_hz=1e9)])
    samples=[r for r in trace['traceEvents'] if r['ph']=='X' and r['name']=='RECV_POLL' and r['tid']==0]
    assert [r['dur'] for r in samples]==pytest.approx([.003,.007])
    assert all(r['args']['interval_kind']=='sampled actual interval' for r in samples)


def test_capacity_keeps_kernel_and_never_wraps():
    report=timer.parse_records(raw(full=True),'dispatch')
    assert all(r['record_count']<=256 and r['omitted_intervals']>0 for r in report['cores'])
    assert sum(r['stage']=='KERNEL' for r in report['intervals'])==56
    assert all(len([r for r in report['intervals'] if r['core']==core])<=96 for core in range(56))
    assert len(report['records'])==56*64


@pytest.mark.parametrize('damage',['header','role','missing_core','truncated','omitted','padding','unknown','negative','duplicate_metric','missing_kernel','counter_duration'])
def test_corruption_is_not_silently_accepted(damage):
    data=raw();words=[v for row in data[0] for v in row]
    if damage=='header':words[1]=6
    if damage=='role':words[5]=1|(10<<8)|(4<<16)
    if damage=='missing_core':data.pop()
    if damage=='truncated':words[6]-=1
    if damage=='omitted':words[7]=0
    if damage=='padding':words[-1]=1
    if damage=='unknown':words[8]=128
    if damage=='negative':words[9]=-1
    metrics=[i for i in range(words[6]) if (words[8+i*2]&0xffffffff)>=0x8000]
    if damage=='duplicate_metric':words[8+metrics[-1]*2]=words[8+metrics[0]*2]
    if damage=='missing_kernel':words[8+metrics[0]*2]=(1<<32)|0x8031
    if damage=='counter_duration':words[9+metrics[-1]*2]=1
    data[0]=[words[i:i+8] for i in range(0,len(words),8)]
    with pytest.raises(ValueError):timer.parse_records(data,'dispatch')


def test_reference_binary_fields_and_stage_dictionary():
    data=timer.reference_binary(raw(),'dispatch')
    assert len(data)==64+72*64+72*256*16
    assert struct.unpack_from('<8I',data)==(0x50524f46,0x10005,56,16,256,64,64+72*64,1)
    count,omitted,role=struct.unpack_from('<3I',data,64)
    assert (count,omitted,role)==(12,1,3)
    assert tuple(json.loads((ROOT/'tests/ll_stage_timer/dc_seg_stages.json').read_text()))==timer.STAGES
    header=(ROOT/'csrc/kernels/dispatch/low_latency/common/ll_stage_timer_layout.h').read_text()
    names=re.search(r'kNames\[\] = \{(.*?)\};',header,re.S).group(1)
    assert tuple(re.findall('"([A-Z_]+)"',names))==timer.STAGES


def test_full_capture_is_deferred_and_saves_extended_json(monkeypatch,tmp_path):
    package=ModuleType('deep_ep');package.__path__=[]
    utils=ModuleType('deep_ep.utils');utils.__path__=[]
    monkeypatch.setitem(sys.modules,'deep_ep',package);monkeypatch.setitem(sys.modules,'deep_ep.utils',utils)
    copies=[]
    class Tensor:
        def __init__(self):self.ptr=256;self.device='npu:0';self.raw=raw(schema=timer.SCHEMA)
        def data_ptr(self):return self.ptr
        def narrow(self,dim,offset,count):self.ptr+=offset*8;return self
        def view(self,*shape):self.shape=shape;return self
        def cpu(self):copies.append(self.shape);return self
        def tolist(self):return self.raw
    native=SimpleNamespace(low_latency_stage_timer_enabled=True,
        low_latency_stage_timer_layout=lambda:dict(schema=timer.SCHEMA,slots=timer.SLOTS,fields=8,stages=list(timer.STAGES)),
        low_latency_core_config=lambda:dict(active_aiv=56,send_cores=10,device=0,runtime_device_queried=True))
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(int64=0,zeros=lambda *a,**k:Tensor()))
    monkeypatch.setitem(sys.modules,'deep_ep._native',SimpleNamespace(require_native=lambda _:native))
    module=load('deep_ep.utils.ll_stage_timer','deep_ep/utils/ll_stage_timer.py')
    trace=module.StageTrace(object(),'dispatch');assert trace.tensor.shape==(56,timer.SLOTS,8) and trace.tensor.data_ptr()%512==0
    tensors={'input':SimpleNamespace(device='npu:0')}
    with module.capture(trace.buffer,'dispatch',prepared=trace,defer_complete=True):module.attach(trace.buffer,tensors,'dispatch')
    assert not copies
    trace.complete();trace.save(tmp_path/'sample_0.json',dict(rank=0,iteration=0))
    assert copies==[(56,timer.SLOTS,8)]
    assert not (tmp_path/'dispatch_prof_rank0_sample_0.bin').exists()
    assert json.loads((tmp_path/'sample_0.json').read_text())['schema']==timer.SCHEMA
    with pytest.raises(FileExistsError):trace.save(tmp_path/'sample_0.json')
    with pytest.raises(ValueError):module.StageTrace(object(),'dispatch',mode='drain')


def test_segment_reports_keep_counter_and_omission_semantics(tmp_path):
    analyzer=load('segment_analyze','tests/ll_stage_timer/analyze.py')
    source=tmp_path/'input';source.mkdir()
    saved=dict(operation='dispatch',raw=raw(),frequency_hz=1e9,
        metadata=dict(rank=0,world_size=1,iteration=0,correctness='passed',config=dict(B=1)))
    (source/'sample_0.json').write_text(json.dumps(saved))
    output=tmp_path/'output';analyzer.process_directory(source,output,1e9,0)
    assert all((output/name).is_file() for name in ['metrics.csv','intervals.csv','cores.csv','trace.json'])
    report=json.loads((output/'summary.json').read_text())
    counter=next(r for r in report['per_rank_core_stage'] if not r['time_measured'])
    assert counter['mean'] is None and counter['mean_calls']==17


def test_legacy_and_segment_diagnostic_times_are_not_compared():
    parity=load('segment_parity','tests/ll_stage_timer/test_parity.py')
    left=parity.identity();left.update(core_protocol=8010,send_cores=10,recv_cores=46,stage_timer_schema=6)
    right=dict(left,core_protocol=9010,stage_timer_schema=7)
    with pytest.raises(ValueError,match='diagnostic semantics'):parity.comparison.check_identity(left,right)
    for item in (left,right):item.update(stage_timer=False,stage_capture=0,expert_capture=False)
    parity.comparison.check_identity(left,right)


def test_uint32_counter_survives_signed_int64_tensor_carrier():
    data=raw();words=[v for row in data[0] for v in row]
    for i in range(words[6]):
        if (words[8+i*2]&0xffffffff)==0x8000+52:
            words[8+i*2]=((0x80000005<<32)|(0x8000+52))-(1<<64)
    data[0]=[words[i:i+8] for i in range(0,len(words),8)]
    report=timer.parse_records(data,'dispatch')
    row=next(r for r in report['records'] if r['core']==0 and r['stage']=='POLL_RECV_COUNTS_ITERATIONS')
    assert row['calls']==0x80000005
    assert timer.reference_binary(data,'dispatch')
