"""Execute the real capture and sampling helpers with CPU-only framework mocks."""
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from deep_ep import _native
from deep_ep.utils import ll_reference_profile as reference
from tests.elastic import test_ep_low_latency as st

ROOT=Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def clean_context():
    reference._state.active={}
    yield
    reference._state.active={}


def prepared(buffer):
    result=reference.ReferenceTrace.__new__(reference.ReferenceTrace)
    result.buffer,result.operation,result.mode=buffer,'dispatch','reference'
    result.frequency_hz,result.launches=1e9,0
    result.tensor=SimpleNamespace(device='npu:0')
    result.raw=result.report=None
    return result


def test_alignment_and_actual_fixed56_layout_without_legacy_import(monkeypatch):
    observed=[]
    class Tensor:
        def __init__(self,address=520):self.address=address
        def data_ptr(self):return self.address
        def narrow(self,axis,offset,length):return Tensor(self.address+offset*8)
        def view(self,*shape):observed.append(shape);return self
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(int64='int64',zeros=lambda *a,**k:Tensor()))
    layout=dict(schema=7,slots=80,fields=8,words=list(reference.WORDS))
    config=dict(active_aiv=56,device=0,runtime_device_queried=True)
    native=SimpleNamespace(low_latency_reference_profile_enabled=True,low_latency_reference_profile_layout=lambda:layout,
                           low_latency_core_config=lambda:config)
    monkeypatch.setattr(_native,'require_native',lambda *a:native)
    trace=reference.ReferenceTrace(object())
    assert trace.tensor.data_ptr()%512==0 and observed==[(56,80,8)]
    layout['schema']=5
    with pytest.raises(RuntimeError,match='layout'):reference.ReferenceTrace(object())
    layout['schema']=7;config['active_aiv']=16
    with pytest.raises(RuntimeError,match='actual NPU'):reference.ReferenceTrace(object())
    config['active_aiv']=56;native.low_latency_reference_profile_enabled=1
    with pytest.raises(RuntimeError,match='Rebuild'):reference.ReferenceTrace(object())


def test_attach_once_deferred_copy_and_exception_cleanup():
    owner=object();trace=prepared(owner);tensors=dict(input=SimpleNamespace(device='npu:0'))
    with reference.capture(owner,prepared=trace,defer_complete=True):
        reference.attach(owner,tensors,'dispatch')
        assert tensors['ll_reference_profile'] is trace.tensor
        with pytest.raises(RuntimeError,match='exactly one'):reference.attach(owner,tensors,'dispatch')
        with pytest.raises(RuntimeError,match='Nested'):
            with reference.capture(owner):pass
    assert trace.raw is None and not reference._state.active
    with pytest.raises(ValueError,match='unused'):
        with reference.capture(owner,prepared=trace):pass
    trace=prepared(owner)
    with pytest.raises(RuntimeError,match='combine'):
        with reference.capture(owner,prepared=trace,defer_complete=True):reference.attach(owner,tensors,'combine')
    assert not reference._state.active and trace.launches==0


def test_capture_wrong_device_and_save_never_overwrite(tmp_path):
    owner=object();trace=prepared(owner)
    with pytest.raises(ValueError,match='same NPU'):
        with reference.capture(owner,prepared=trace):reference.attach(owner,{'input':SimpleNamespace(device='npu:1')},'dispatch')
    assert trace.launches==0 and not reference._state.active
    trace.raw=[[[2**63+17]]];trace.report=dict(schema=7,records=[])
    path=tmp_path/'sample.json';trace.save(path,dict(rank=7))
    assert json.loads(path.read_text())['raw']==trace.raw
    with pytest.raises(FileExistsError):trace.save(path)


@pytest.mark.parametrize('option',['--ll-stage-timer','--ll-expert-timer','--ll-timer-mode=observe','--ll-profile-capture'])
def test_retired_options_are_not_accepted(option):
    with pytest.raises(SystemExit):st.parse_args([option])


def test_reference_cli_validates_operation_and_fixed_batch_protocol():
    args=st.parse_args(['--ll-reference-profile','--dispatch-only'])
    assert args.reference_iters==1
    st.validate_args(args,16)
    assert (args.delay_exp_before,args.delay_exp_after)==(100,0)
    for options in (['--ll-reference-profile'],['--ll-reference-profile','--dispatch-only','--skip-check'],
                    ['--delay-exp-before','1'],['--reference-iters','0']):
        with pytest.raises(ValueError):st.validate_args(st.parse_args(options),16)


@pytest.mark.parametrize('after_perf',[False,True])
def test_reference_capture_matches_batch_prefix_and_delays_all_cpu_work(tmp_path,monkeypatch,after_perf):
    events=[];metadata=[];owner=object()
    class Tensor:device='npu:0'
    class Trace:
        def __init__(self,buffer,operation,frequency_hz):
            self.buffer,self.operation,self.mode=buffer,operation,'reference'
            self.frequency_hz,self.launches=frequency_hz,0;self.tensor=Tensor();events.append('allocate')
        def complete(self):events.append('copy')
        def save(self,path,meta):metadata.append(meta);events.append('save')
    dist=SimpleNamespace(get_world_size=lambda:16,ReduceOp=SimpleNamespace(SUM=0),all_reduce=lambda *a,**k:events.append('reduce'))
    torch=SimpleNamespace(float32=0,zeros=lambda *a,**k:Tensor(),ones=lambda *a,**k:Tensor(),distributed=dist,
        exp=lambda *a,**k:events.append('exp'),npu=SimpleNamespace(synchronize=lambda:events.append('sync')))
    monkeypatch.setitem(sys.modules,'torch',torch);monkeypatch.setitem(sys.modules,'torch.distributed',dist)
    monkeypatch.setattr(reference,'ReferenceTrace',Trace)
    monkeypatch.setattr(_native,'require_native',lambda *a:SimpleNamespace(low_latency_ranks_per_group=4))
    monkeypatch.setattr(st,'completion_boundary',lambda *a:events.append('boundary'))
    monkeypatch.setattr(st,'agree_check',lambda fn,*a:fn())
    monkeypatch.setattr(st,'case_config',lambda *a:dict(B=2))
    monkeypatch.setattr(st,'comparison_metadata',lambda *a:dict(installed_source_sha256='a'*64))
    args=st.parse_args(['--ll-reference-profile','--dispatch-only','--reference-iters','2','--num-tokens','2','--warmup','2','--num-iters','3'])
    args.run_directory=str(tmp_path);args.case_id=0;args.profile_route_identity=dict(profile_seed=0,profile_route_sha256='a'*64)
    def invoke():
        reference.attach(owner,dict(input=Tensor()),'dispatch');events.append('invoke')
        return (SimpleNamespace(current_stream_wait=lambda:events.append('wait')),)
    st.diagnostic_calls(owner,invoke,lambda out:events.append('check'),args,'dispatch',0,32,'normal',
        prelude_buffers=(Tensor(),Tensor()) if after_perf else None)
    expected=['allocate']*2
    if not after_perf:expected+=['boundary','reduce']+['invoke']*2+['sync']
    expected+=['exp']*100
    if not after_perf:expected+=['invoke']*3
    expected+=['invoke','wait']*2+['boundary']+['copy','check','save']*2
    assert events==expected
    assert all(m['sampling']['exp_prefix_repeats']==100 and not m['sampling']['per_iteration_all_reduce'] for m in metadata)
    assert all(m['profile_seed']==0 and m['route_sha256']=='a'*64 for m in metadata)


def test_perf_exp_prefix_is_before_active_step_and_not_between_calls(tmp_path,monkeypatch):
    events=[]
    class Tensor:pass
    class Profiler:
        def __enter__(self):events.append('enter');return self
        def __exit__(self,*a):events.append('exit')
        def step(self):events.append('step')
    dist=SimpleNamespace(ReduceOp=SimpleNamespace(SUM=0),all_reduce=lambda *a,**k:events.append('reduce'),
        barrier=lambda:events.append('barrier'),get_rank=lambda:0,get_world_size=lambda:16,
        all_gather_object=lambda rows,local:[rows.__setitem__(i,list(local)) for i in range(16)])
    torch=SimpleNamespace(float32=0,zeros=lambda *a,**k:Tensor(),ones=lambda *a,**k:Tensor(),distributed=dist,
        exp=lambda *a,**k:events.append('exp'),npu=SimpleNamespace(synchronize=lambda:events.append('sync')))
    profiler=SimpleNamespace(profile=lambda **k:Profiler(),schedule=lambda **k:k,tensorboard_trace_handler=lambda p:p,
        _ExperimentalConfig=lambda **k:k,AiCMetrics=SimpleNamespace(PipeUtilization=0),ProfilerLevel=SimpleNamespace(Level1=1),ExportType=SimpleNamespace(Text=0))
    monkeypatch.setitem(sys.modules,'torch',torch);monkeypatch.setitem(sys.modules,'torch.distributed',dist)
    monkeypatch.setitem(sys.modules,'torch_npu',SimpleNamespace(profiler=profiler))
    monkeypatch.setattr(st,'completion_boundary',lambda *a:events.append('boundary'))
    monkeypatch.setattr(st,'agree_check',lambda fn,*a:fn())
    monkeypatch.setattr(st,'kernel_durations',lambda *a:events.append('parse') or [1.]*100)
    args=st.parse_args([])
    result=st.measure(lambda:events.append('invoke') or object(),args,'dispatch',tmp_path,
                      after_profile=lambda *a:events.append('diagnostic'))
    begin,end=events.index('enter'),events.index('exit')
    assert events[:begin]==['boundary','reduce']
    assert events[begin+1:end]==['invoke']*10+['sync']+['exp']*100+['step']+['invoke']*100+['barrier','sync','step']
    assert events[end+1:]==['diagnostic','parse']
    assert result['perf_tail']==50 and result['kernel_samples']==800
