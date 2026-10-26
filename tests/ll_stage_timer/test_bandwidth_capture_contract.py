"""Real Python capture contract: carrier alignment, mode and schema identity."""
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[2]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


timer=load('bw_capture_timer','deep_ep/utils/ll_stage_timer.py')
entry=load('bw_capture_entry','tests/elastic/test_ep_low_latency.py')


def layout():
    return dict(schema=17,slots=16,fields=8,stages=list(timer.STAGES),
        points=list(timer.bandwidth.POINTS),recording='stage-bandwidth',
        timer_profile='bandwidth',timer_profile_id=8,
        bandwidth_stages=list(timer.bandwidth.STAGES),bandwidth_fields=list(timer.bandwidth.FIELDS),
        bandwidth_word_offset=32,bandwidth_record_words=16)


def capability():
    data=layout()
    return dict(stage_timer=True,stage_timer_schema=17,timer_profile='bandwidth',
        timer_profile_id=8,overlap_stage=3,
        **{'stage_timer_'+key:data[key] for key in ('slots','fields','points','recording',
           'bandwidth_stages','bandwidth_fields','bandwidth_word_offset','bandwidth_record_words')})


def raw(mode):
    result=[]
    for core in range(56):
        words=[0]*128
        words[:8]=[timer.MAGIC,17,1,56,core,mode|(30<<8)|((3 if core<30 else 2)<<16)|(8<<24),0,0]
        for name,tick in [('ENTRY',100),('INIT_DONE',200),('WORK_DONE',1000)]:
            index=timer.bandwidth.POINTS.index(name);words[6]|=1<<index;words[8+index]=tick
        result.append([words[i:i+8] for i in range(0,128,8)])
    return result


@pytest.mark.parametrize('mode',('observe','bandwidth-complete'))
def test_actual_capture_allocation_attachment_and_deferred_export(monkeypatch,tmp_path,mode):
    for name in ('deep_ep','deep_ep.utils'):
        package=ModuleType(name);package.__path__=[];monkeypatch.setitem(sys.modules,name,package)
    copies=[]
    class Tensor:
        def __init__(self):self.ptr=256;self.device='npu:0'
        def data_ptr(self):return self.ptr
        def narrow(self,dim,offset,count):self.ptr+=offset*8;return self
        def view(self,*shape):self.shape=shape;return self
        def cpu(self):copies.append(self.shape);return self
        def tolist(self):return raw(timer.DIAGNOSTIC_MODES[mode])
    native=SimpleNamespace(low_latency_stage_timer_enabled=True,
        low_latency_stage_timer_layout=layout,low_latency_overlap_stage=3,
        low_latency_core_config=lambda:dict(active_aiv=56,device=0,runtime_device_queried=True))
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(int64=0,zeros=lambda *a,**k:Tensor()))
    monkeypatch.setitem(sys.modules,'deep_ep._native',SimpleNamespace(require_native=lambda _:native))
    module=load('deep_ep.utils.ll_stage_timer','deep_ep/utils/ll_stage_timer.py')
    trace=module.StageTrace(object(),'dispatch',mode)
    assert trace.tensor.shape==(56,16,8) and trace.tensor.data_ptr()%512==0
    tensors={'input':SimpleNamespace(device='npu:0')}
    with module.capture(trace.buffer,'dispatch',mode,prepared=trace,defer_complete=True):
        module.attach(trace.buffer,tensors,'dispatch')
        assert tensors['ll_stage_timer_mode']==timer.DIAGNOSTIC_MODES[mode]
        with pytest.raises(RuntimeError,match='multiple launches'):
            module.attach(trace.buffer,tensors,'dispatch')
    assert not copies
    trace.complete();trace.save(tmp_path/'sample_0000.json',dict(rank=0,iteration=0))
    assert copies==[(56,16,8)]
    saved=json.loads((tmp_path/'sample_0000.json').read_text())
    assert saved['schema']==17 and saved['boundary_mode']==mode
    chrome=json.loads((tmp_path/'trace_sample_0000.json').read_text())
    assert chrome['metadata']['format']=='stage-bandwidth'
    assert all(row.get('cat')!='bandwidth' for row in chrome['traceEvents'])
    native.low_latency_overlap_stage=2
    if mode=='bandwidth-complete':
        with pytest.raises(RuntimeError,match='stage3'):module.StageTrace(object(),'dispatch',mode)


@pytest.mark.parametrize('mode',('observe','bandwidth-complete'))
def test_entry_exact_layout_and_stage_gate(mode):
    entry.validate_stage_timer_capability(capability(),mode)
    for key,value in [('stage_timer_bandwidth_stages',list(reversed(timer.bandwidth.STAGES))),
                      ('stage_timer_bandwidth_fields',['peer_bytes']),
                      ('stage_timer_bandwidth_word_offset',24),('stage_timer_bandwidth_record_words',15),
                      ('stage_timer_slots',8),('timer_profile_id',7)]:
        data=capability();data[key]=value
        with pytest.raises(RuntimeError):entry.validate_stage_timer_capability(data,mode)
    data=capability();data['overlap_stage']=2
    if mode=='bandwidth-complete':
        with pytest.raises(RuntimeError,match='stage3'):entry.validate_stage_timer_capability(data,mode)


def test_completion_requires_actual_dispatch_capture():
    flags=['--ll-stage-timer','--dispatch-only','--skip-perf-test','--ll-profile-capture']
    common=['--num-processes','16','--num-tokens','256','--hidden','3584',
            '--num-topk','8','--num-experts','384','--dtype','fp8',
            '--ll-timer-mode','bandwidth-complete']
    assert entry.validate_args(entry.parse_args(common+flags),16)==256
    for removed in flags:
        with pytest.raises(ValueError,match='Bandwidth completion require'):
            entry.validate_args(entry.parse_args(common+[f for f in flags if f!=removed]),16)
    with pytest.raises(ValueError,match='Invalid LL stage diagnostic'):
        timer.StageTrace(object(),'combine','bandwidth-complete')
    with pytest.raises(RuntimeError,match='schema16'):
        entry.validate_stage_timer_capability(capability(),'mte3-tx')
