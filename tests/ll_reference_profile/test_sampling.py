"""Run real ST sampling helpers with CPU event spies, never NPU initialization."""
from contextlib import contextmanager
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('reference_st',ROOT/'tests/elastic/test_ep_low_latency.py')
st=importlib.util.module_from_spec(spec);spec.loader.exec_module(st)


def args(tmp_path):
    result=st.parse_args(['--ll-reference-profile','--dispatch-only','--warmup','2','--num-iters','3',
        '--ll-timer-iters','2','--seed','17','--delay-rows','1','--delay-cols','1'])
    result.run_directory=str(tmp_path/'run');result.case_id=0
    result.reference_profile_seed=17;result.reference_route_sha256='a'*64
    return result


def mocks(monkeypatch,events):
    class Tensor:pass
    dist=ModuleType('torch.distributed')
    dist.ReduceOp=SimpleNamespace(SUM=0)
    dist.get_world_size=lambda:16;dist.get_rank=lambda:0
    dist.all_reduce=lambda *a,**k:events.append('reduce')
    dist.barrier=lambda:events.append('barrier')
    def gather(out,values):
        events.append('gather')
        for i in range(len(out)):out[i]=values
    dist.all_gather_object=gather
    torch=ModuleType('torch');torch.__path__=[];torch.float32=0;torch.distributed=dist
    torch.zeros=lambda *a,**k:Tensor();torch.ones=lambda *a,**k:Tensor()
    torch.exp=lambda *a,**k:events.append('exp')
    torch.npu=SimpleNamespace(synchronize=lambda:events.append('sync'))
    monkeypatch.setitem(sys.modules,'torch',torch);monkeypatch.setitem(sys.modules,'torch.distributed',dist)
    monkeypatch.setattr(st,'control_device',lambda *a:'npu')
    monkeypatch.setattr(st,'completion_boundary',lambda *a:events.append('boundary'))
    monkeypatch.setattr(st,'agree_check',lambda invoke,*a:invoke())
    return torch,dist


@pytest.mark.parametrize('after_perf',[False,True])
def test_reference_batch_prefix_and_delayed_export(monkeypatch,tmp_path,after_perf):
    events=[];saved=[];arg=args(tmp_path);mocks(monkeypatch,events)
    for name in ('deep_ep','deep_ep.utils'):
        package=ModuleType(name);package.__path__=[];monkeypatch.setitem(sys.modules,name,package)
    monkeypatch.setitem(sys.modules,'deep_ep._native',SimpleNamespace(require_native=lambda _:SimpleNamespace(low_latency_ranks_per_group=4)))
    @contextmanager
    def capture(*a,**k):
        events.append('attach');yield;events.append('detach')
    monkeypatch.setitem(sys.modules,'deep_ep.utils.ll_reference_profile',SimpleNamespace(capture=capture))
    class Trace:
        def complete(self):events.append('copy')
        def save(self,path,metadata):saved.append((path,metadata));events.append('save')
    def invoke():
        events.append('invoke');return (SimpleNamespace(current_stream_wait=lambda:events.append('wait')),)
    st.diagnostic_calls(object(),invoke,lambda _:events.append('check'),arg,'dispatch',0,16,'normal',
                        prepared=[Trace(),Trace()],prelude_buffers=(object(),object()) if after_perf else None)
    assert events.count('exp')==100
    assert events.count('reduce')==(0 if after_perf else 1)
    if not after_perf:assert events[:5]==['boundary','reduce','invoke','invoke','sync']
    prefix_end=events.index('exp')+100
    assert events[events.index('exp'):prefix_end]==['exp']*100
    tail=events[prefix_end:]
    assert 'sync' not in tail and 'reduce' not in tail
    assert tail[:3]==(['attach','invoke','wait'] if after_perf else ['invoke']*3)
    assert events.index('copy')>max(i for i,event in enumerate(events) if event=='invoke')
    assert [event for event in events if event in ('copy','check','save')]==['copy','check','save']*2
    assert all('ll_reference_profile' in path.parts for path,_ in saved)
    for _,meta in saved:
        assert meta['profile_seed']==17 and meta['route_sha256']=='a'*64
        assert meta['sampling']['sampling_protocol']==st.SAMPLING_PROTOCOL
        assert meta['sampling']['exp_prefix_repeats']==100
        assert meta['sampling']['per_iteration_exp']==0 and meta['sampling']['per_iteration_all_reduce'] is False


def test_profiler_exp_prefix_before_active_step_and_none_inside(monkeypatch,tmp_path):
    events=[];arg=args(tmp_path);mocks(monkeypatch,events)
    @contextmanager
    def profile(**kw):
        events.append('profile_enter');yield SimpleNamespace(step=lambda:events.append('step'));events.append('profile_exit')
    profiler=SimpleNamespace(profile=profile,schedule=lambda **k:None,tensorboard_trace_handler=lambda *a:None,
        _ExperimentalConfig=lambda **k:None,AiCMetrics=SimpleNamespace(PipeUtilization=0),
        ProfilerLevel=SimpleNamespace(Level1=0),ExportType=SimpleNamespace(Text=0))
    monkeypatch.setitem(sys.modules,'torch_npu',SimpleNamespace(profiler=profiler))
    monkeypatch.setattr(st,'kernel_durations',lambda *a:[1,2,3])
    st.measure(lambda:events.append('invoke'),arg,'dispatch',tmp_path/'profile',
               after_profile=lambda *a:events.append('diagnostic'))
    start=events.index('step');end=events.index('step',start+1)
    assert events[start-100:start]==['exp']*100
    assert events[start+1:end]==['invoke']*3+['barrier','sync']
    assert events.count('reduce')==1 and events.index('diagnostic')>events.index('profile_exit')


def test_declared_route_is_restored_after_correctness_seed_changes():
    source=inspect.getsource(st.run_case)
    # Execute the actual restoration block against pure-CPU route data.
    begin=source.index('    # Restore the declared route')
    end=source.index('    if args.dispatch_only:',begin)
    block='\n'.join(line[4:] for line in source[begin:end].splitlines())
    arg=SimpleNamespace(ll_reference_profile=True,skip_check=False,correctness_repeats=3,seed=17,
                        num_tokens=4,num_topk=8,num_experts=384)
    calls=[]
    class Event:
        def current_stream_wait(self):calls.append('wait')
    torch=SimpleNamespace(int64=0,tensor=lambda value,**k:value,npu=SimpleNamespace(synchronize=lambda:calls.append('sync')))
    def dispatch():calls.append('dispatch');return ('recv',None,None,'matching-handle',Event())
    scope=dict(args=arg,world=16,rank=0,m=4,mode='normal',torch=torch,dispatch=dispatch,
        completion_boundary=lambda _:calls.append('boundary'),make_routes=st.make_routes,
        expert_entries=st.expert_entries,agree_check=lambda f,*a:f(),verify_dispatch=lambda *a:calls.append('check'),
        hashlib=hashlib,json=json)
    exec(compile(block,'actual-route-reset','exec'),scope)
    expected=st.make_routes(16,4,8,384,17)
    assert scope['routes']==expected and scope['handle']=='matching-handle'
    assert arg.sample_route_sha256==hashlib.sha256(json.dumps(expected,separators=(',',':')).encode()).hexdigest()
    assert arg.reference_profile_seed==17 and calls==['boundary','dispatch','wait','sync','check']


def test_only_reference_cli_and_batch_protocol_are_accepted(tmp_path):
    arg=args(tmp_path)
    assert st.validate_args(arg,16)==16
    assert arg.delay_exp_before==100 and arg.delay_exp_after==0
    for flag in ('--ll-stage-timer','--ll-expert-timer','--ll-profile-capture','--ll-timer-mode'):
        with pytest.raises(SystemExit):st.parse_args([flag])
    with pytest.raises(ValueError,match='dispatch-only'):
        st.validate_args(st.parse_args(['--ll-reference-profile']),16)
    with pytest.raises(ValueError,match='correctness'):
        st.validate_args(st.parse_args(['--ll-reference-profile','--dispatch-only','--skip-check']),16)
    with pytest.raises(ValueError,match='100'):
        st.validate_args(st.parse_args(['--delay-exp-before','1']),16)


def test_reference_capability_checks_current_layout():
    parser=st.runpy.run_path(str(ROOT/'deep_ep/utils/ll_reference_profile_parse.py'))
    layout=dict(schema=6,slots=80,fields=8,words=list(parser['WORDS']),implementation='mte-shared-receive',send_cores=30,recv_cores=26)
    valid=dict(reference_profile=True,reference_profile_layout=layout,send_cores=30,recv_cores=26,
               overlap_stage=3,installed_source_sha256='a'*64)
    st.validate_reference_profile_capability(True,valid)
    for field,value in [('reference_profile',1),('overlap_stage',2),('installed_source_sha256','bad')]:
        with pytest.raises(RuntimeError):st.validate_reference_profile_capability(True,dict(valid,**{field:value}))


def test_multi_host_run_name_is_shared_but_paths_remain_local(tmp_path,monkeypatch):
    monkeypatch.setattr(st,'PROJECT_ROOT',tmp_path)
    name=[];events=[]
    def broadcast(value,src):
        assert src==0
        if value[0] is not None:name[:]=value
        else:value[:]=name
    dist=SimpleNamespace(broadcast_object_list=broadcast)
    monkeypatch.setattr(st,'agree_check',lambda callback,args,rank:(callback(),events.append(('agreed',rank))))
    original=st.create_run_directory
    def create(root,leaf):events.append(('create',root,leaf));return original(root,leaf)
    monkeypatch.setattr(st,'create_run_directory',create)
    first=SimpleNamespace(output_dir=str(tmp_path/'host0 checkout/results'),run_name='same-run')
    second=SimpleNamespace(output_dir=str(tmp_path/'other host checkout/results'),run_name='same-run')
    a=Path(st.create_distributed_run_directory(first,0,0,dist))
    b=Path(st.create_distributed_run_directory(second,16,0,dist))
    c=Path(st.create_distributed_run_directory(second,17,1,dist))
    assert a!=b and b==c and a.name==b.name=='same-run'
    assert a.parent==Path(first.output_dir) and b.parent==Path(second.output_dir)
    assert a.is_dir() and b.is_dir() and name==['same-run']
    assert len([e for e in events if e[0]=='create'])==2


@pytest.mark.parametrize('name',('../escape','/absolute','bad\\child','..','',None))
def test_remote_run_name_cannot_escape_local_output(tmp_path,monkeypatch,name):
    monkeypatch.setattr(st,'PROJECT_ROOT',tmp_path)
    calls=[]
    def agree(callback,args,rank):
        try:callback()
        except Exception:calls.append('failure consensus');raise
        calls.append('success consensus')
    monkeypatch.setattr(st,'agree_check',agree)
    dist=SimpleNamespace(broadcast_object_list=lambda value,src:value.__setitem__(0,name))
    config=SimpleNamespace(output_dir=str(tmp_path/'results'),run_name='safe')
    with pytest.raises(ValueError,match='single local'):
        st.create_distributed_run_directory(config,16,0,dist)
    assert calls==['success consensus','failure consensus']
    assert not (tmp_path/'results').exists()


def test_remote_nonempty_run_directory_is_preserved_and_fails_collectively(tmp_path,monkeypatch):
    monkeypatch.setattr(st,'PROJECT_ROOT',tmp_path)
    output=tmp_path/'results/same-run';output.mkdir(parents=True)
    (output/'evidence.json').write_text('keep')
    errors=[]
    def agree(callback,args,rank):
        try:callback()
        except Exception as error:errors.append(type(error));raise
    monkeypatch.setattr(st,'agree_check',agree)
    dist=SimpleNamespace(broadcast_object_list=lambda value,src:value.__setitem__(0,'same-run'))
    config=SimpleNamespace(output_dir=str(output.parent),run_name='same-run')
    with pytest.raises(FileExistsError,match='preserved'):
        st.create_distributed_run_directory(config,16,0,dist)
    assert errors==[FileExistsError] and (output/'evidence.json').read_text()=='keep'


def test_output_root_outside_project_rejected_before_broadcast(tmp_path,monkeypatch):
    monkeypatch.setattr(st,'PROJECT_ROOT',tmp_path/'project')
    monkeypatch.setattr(st,'agree_check',lambda callback,*a:callback())
    dist=SimpleNamespace(broadcast_object_list=lambda *a,**k:pytest.fail('must reject before broadcast'))
    config=SimpleNamespace(output_dir=str(tmp_path/'outside'),run_name='same-run')
    with pytest.raises(ValueError,match='inside the local project'):
        st.create_distributed_run_directory(config,0,0,dist)
