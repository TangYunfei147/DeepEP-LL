"""Profiler sampling protocol stays separate from event diagnostics."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[2]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

@pytest.mark.parametrize('with_tail', [False, True])
def test_native_st_reference_schedule(monkeypatch,tmp_path,with_tail):
    import sys
    st=load('ll_stage_st',ROOT/'tests/elastic/test_ep_low_latency.py')
    calls=[]
    class Tensor:
        def fill_(self,value):calls.append('fill')
    class Profiler:
        def __enter__(self):return self
        def __exit__(self,*args):calls.append('profile_closed')
        def step(self):calls.append('step')
    def profile(**kwargs):
        assert kwargs['schedule']==dict(wait=0,warmup=1,active=1,repeat=1,skip_first=0)
        return Profiler()
    def gather(out,values):out[:]=[values]
    dist=SimpleNamespace(all_reduce=lambda *a,**k:calls.append('reduce'),barrier=lambda:calls.append('barrier'),
                         ReduceOp=SimpleNamespace(SUM=0),get_rank=lambda:0,get_world_size=lambda:1,all_gather_object=gather)
    torch=SimpleNamespace(float32=0,zeros=lambda *a,**k:Tensor(),ones=lambda *a,**k:Tensor(),
                          npu=SimpleNamespace(synchronize=lambda:calls.append('sync')),distributed=dist,
                          exp=lambda *a,**k:calls.append('exp'))
    npu=SimpleNamespace(profiler=SimpleNamespace(profile=profile,schedule=lambda **kw:kw,
        tensorboard_trace_handler=lambda path:None,_ExperimentalConfig=lambda **kw:kw,
        AiCMetrics=SimpleNamespace(PipeUtilization=0),ProfilerLevel=SimpleNamespace(Level1=0),
        ExportType=SimpleNamespace(Text=0)))
    for name,value in [('torch',torch),('torch.distributed',dist),('torch_npu',npu)]:monkeypatch.setitem(sys.modules,name,value)
    st.completion_boundary=lambda args:None
    st.agree_check=lambda fn,*args:fn()
    st.trace_call=lambda args,label,fn,*a,**kw:fn(*a,**kw)
    st.kernel_durations=lambda *args:calls.append('parse') or [1.0]*100
    args=st.parse_args([])
    assert (args.warmup,args.num_iters,args.perf_tail)==(10,100,50)
    tail=(lambda delay,alignment:calls.append('tail')) if with_tail else None
    st.measure(lambda:calls.append('invoke'),args,'dispatch',tmp_path,after_profile=tail)
    assert calls==['reduce']+['invoke']*10+['sync','step']+['exp','fill','reduce','invoke']*100+['barrier','sync','step','profile_closed']+(['tail'] if with_tail else [])+['parse']
