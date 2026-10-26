"""Exercise real capture/entry contracts without importing Torch or using an NPU."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[2]


def load(name,relative):
    spec=importlib.util.spec_from_file_location(name,ROOT/relative)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


timer=load('mte3_capture_timer','deep_ep/utils/ll_stage_timer.py')
entry=load('mte3_capture_entry','tests/elastic/test_ep_low_latency.py')


def layout():
    return dict(schema=16,slots=8,fields=8,stages=list(timer.STAGES),
        points=list(timer.mte3.POINTS),recording='mte3-backpressure',timer_profile='mte3',timer_profile_id=7,
        wait_metrics=list(timer.mte3.METRICS),
        wait_metric_fields=['sum_ticks','max_ticks','calls','empty_control_ticks'],
        wait_metric_word_offset=32)


def capability():
    data=layout()
    return dict(stage_timer=True,stage_timer_schema=16,timer_profile='mte3',timer_profile_id=7,
        **{'stage_timer_'+key:data[key] for key in ('slots','fields','points','recording',
           'wait_metrics','wait_metric_fields','wait_metric_word_offset')})


def raw(mode,operation='dispatch'):
    records=[]
    for core in range(56):
        role=(3 if core<30 else 2) if operation=='dispatch' else (1 if core<30 else 4)
        words=[0]*64
        words[:8]=[timer.MAGIC,16,1 if operation=='dispatch' else 2,56,core,
                   mode | (30<<8) | (role<<16) | (7<<24),0,0]
        for point,value in [('ENTRY',100),('INIT_DONE',200),('WORK_DONE',2000)]:
            index=timer.mte3.POINTS.index(point)
            words[6] |= 1<<index;words[8+index]=value+core
        records.append([words[i:i+8] for i in range(0,64,8)])
    return records


@pytest.mark.parametrize('mode',('observe','mte3-tx','mte3-rx'))
def test_schema16_entry_checks_exact_metric_semantics(mode):
    entry.validate_stage_timer_capability(capability(),mode)
    for key,bad in [('stage_timer_wait_metrics',list(reversed(timer.mte3.METRICS))),
                    ('stage_timer_wait_metric_fields',['calls','sum_ticks']),
                    ('stage_timer_wait_metric_word_offset',24),('timer_profile_id',6),
                    ('stage_timer_points',list(timer.SPARSE_POINTS))]:
        data=capability();data[key]=bad
        with pytest.raises(RuntimeError):entry.validate_stage_timer_capability(data,mode)


def test_schema15_cannot_silently_run_intervention():
    data=dict(stage_timer=True,stage_timer_schema=15,timer_profile='sparse',timer_profile_id=6,
        stage_timer_slots=8,stage_timer_fields=8,stage_timer_points=list(timer.SPARSE_POINTS),
        stage_timer_recording='sparse-milestones')
    entry.validate_stage_timer_capability(data)
    for mode in ('mte3-tx','mte3-rx'):
        with pytest.raises(RuntimeError,match='schema16'):entry.validate_stage_timer_capability(data,mode)


@pytest.mark.parametrize('mode',('mte3-tx','mte3-rx'))
def test_interventions_require_real_capture_profile_and_dispatch_only(mode):
    flags=['--ll-stage-timer','--dispatch-only','--skip-perf-test','--ll-profile-capture']
    common=['--num-processes','16','--num-tokens','256','--hidden','3584',
            '--num-topk','8','--num-experts','384','--dtype','fp8','--ll-timer-mode',mode]
    assert entry.validate_args(entry.parse_args(common+flags),16)==256
    for removed in flags:
        with pytest.raises(ValueError,match='MTE3 interventions require'):
            entry.validate_args(entry.parse_args(common+[f for f in flags if f!=removed]),16)


@pytest.mark.parametrize('mode',('observe','mte3-tx','mte3-rx'))
def test_capture_transmits_mode_keeps_one_launch_and_delays_copy(monkeypatch,tmp_path,mode):
    package=ModuleType('deep_ep');package.__path__=[]
    utils=ModuleType('deep_ep.utils');utils.__path__=[]
    monkeypatch.setitem(sys.modules,'deep_ep',package)
    monkeypatch.setitem(sys.modules,'deep_ep.utils',utils)
    copies=[]
    class Tensor:
        def __init__(self):self.ptr=256;self.device='npu:0'
        def data_ptr(self):return self.ptr
        def narrow(self,dim,offset,count):self.ptr+=offset*8;return self
        def view(self,*shape):self.shape=shape;return self
        def cpu(self):copies.append(self.shape);return self
        def tolist(self):return raw(timer.DIAGNOSTIC_MODES[mode])
    native=SimpleNamespace(low_latency_stage_timer_enabled=True,
        low_latency_stage_timer_layout=layout,
        low_latency_core_config=lambda:dict(active_aiv=56,send_cores=30,device=0,runtime_device_queried=True))
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(int64=0,zeros=lambda *a,**k:Tensor()))
    monkeypatch.setitem(sys.modules,'deep_ep._native',SimpleNamespace(require_native=lambda _:native))
    module=load('deep_ep.utils.ll_stage_timer','deep_ep/utils/ll_stage_timer.py')
    trace=module.StageTrace(object(),'dispatch',mode)
    assert trace.tensor.data_ptr()%512==0
    tensors={'input':SimpleNamespace(device='npu:0')}
    with module.capture(trace.buffer,'dispatch',mode,prepared=trace,defer_complete=True):
        module.attach(trace.buffer,tensors,'dispatch')
        assert tensors['ll_stage_timer_mode']==timer.DIAGNOSTIC_MODES[mode]
        with pytest.raises(RuntimeError,match='multiple launches'):
            module.attach(trace.buffer,tensors,'dispatch')
    assert not copies
    trace.complete();trace.save(tmp_path/'sample_0000.json',dict(rank=0,iteration=0))
    assert copies==[(56,8,8)]
    saved=json.loads((tmp_path/'sample_0000.json').read_text())
    assert saved['schema']==16 and saved['boundary_mode']==mode
    assert json.loads((tmp_path/'trace_sample_0000.json').read_text())['metadata']['all_calls_in_metrics'] is False


@pytest.mark.parametrize('mode',('mte3-tx','mte3-rx','drain'))
def test_combine_rejects_intervention_before_native_import(mode):
    with pytest.raises(ValueError,match='Invalid LL stage diagnostic'):
        timer.StageTrace(object(),'combine',mode)


def test_trace_rejects_mixed_modes_instead_of_aligning_wait_statistics():
    samples=[dict(raw=raw(code),operation='dispatch',frequency_hz=1e9,
                  metadata=dict(rank=0,iteration=i)) for i,code in enumerate((1,4))]
    with pytest.raises(ValueError,match='Mixed .* modes'):
        timer.chrome_trace(samples)
