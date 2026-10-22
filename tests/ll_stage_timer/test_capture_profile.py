"""Actual instrumented task profiling: identity, sample mapping and ordering."""
from contextlib import contextmanager
import csv
import json
from types import SimpleNamespace
import sys

import pytest
from tests.elastic import test_ep_low_latency as st


def write_profile(path, rows):
    path.mkdir(parents=True, exist_ok=True)
    with (path/'kernel_details.csv').open('w', newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=['Name','Duration(us)','Start Time(us)'])
        writer.writeheader()
        writer.writerows(rows)


def test_active_task_parser_orders_and_rejects_ambiguous_exports(tmp_path):
    entry='dispatch_server_dedup_kernel'
    write_profile(tmp_path, [{'Name':entry, 'Duration(us)':22, 'Start Time(us)':200},
                             {'Name':entry, 'Duration(us)':18, 'Start Time(us)':100},
                             {'Name':'unrelated_kernel', 'Duration(us)':900, 'Start Time(us)':50}])
    result=st.read_active_capture_profile(tmp_path,'dispatch',2,0)
    assert result['status']=='available' and result['kernel_us']==[18,22]
    assert result['statistics']['mean_us']==20
    write_profile(tmp_path,[{'Name':entry, 'Duration(us)':22, 'Start Time(us)':100},
                            {'Name':entry, 'Duration(us)':18, 'Start Time(us)':100}])
    result=st.read_active_capture_profile(tmp_path,'dispatch',2,0)
    assert result['status']=='unavailable' and 'unique finite' in result['reason']
    assert 'kernel_us' not in result
    assert st.read_active_capture_profile(tmp_path,'dispatch',3,0)['status']=='unavailable'


def test_all_rank_capture_completeness_required():
    rank=lambda n:dict(rank=n, operation='dispatch', status='available', expected_samples=2, kernel_us=[10+n,20+n])
    rows=[rank(0),rank(1)]
    assert st.summarize_active_capture_profiles(rows,'dispatch')['status']=='available'
    rows[1]=dict(rank=1, operation='dispatch', status='unavailable', reason='missing task')
    result=st.summarize_active_capture_profiles(rows,'dispatch')
    assert result['status']=='unavailable' and 'summary' not in result
    assert st.summarize_active_capture_profiles([rank(0),rank(0)],'dispatch')['status']=='unavailable'
    assert st.summarize_active_capture_profiles([],'dispatch')['status']=='unavailable'


def test_profile_flag_requires_active_timer():
    args=st.parse_args(['--ll-profile-capture'])
    with pytest.raises(ValueError,match='requires --ll-stage-timer'):
        st.validate_args(args,16)


def test_installed_source_fingerprint_is_gathered_and_mismatch_rejected(tmp_path):
    package=SimpleNamespace(__file__=str(tmp_path/'__init__.py'))
    (tmp_path/'_build_info.json').write_text(json.dumps(dict(source_sha256='a'*64)))
    assert st.installed_source_fingerprint(package)=='a'*64
    common=dict(implementation='candidate',timer_profile='prefix',source_sha256='a'*64)
    st.validate_matching_capabilities([common,dict(common)])
    with pytest.raises(RuntimeError,match='source build differs'):
        st.validate_matching_capabilities([common,dict(common,source_sha256='b'*64)])
    (tmp_path/'_build_info.json').write_text(json.dumps(dict(source_sha256='missing')))
    with pytest.raises(RuntimeError,match='source_sha256 build identity'):
        st.installed_source_fingerprint(package)


def test_profiler_surrounds_active_capture_without_per_call_cpu_work(monkeypatch,tmp_path):
    from deep_ep.utils import ll_stage_timer as timer
    from deep_ep import _native
    events=[];saved=[]
    class Tensor:
        def fill_(self,value):events.append('fill')
    def gather(out,value):
        events.append('gather_profile')
        for rank in range(len(out)):
            out[rank]=dict(value,rank=rank)
    dist=SimpleNamespace(get_world_size=lambda:16, ReduceOp=SimpleNamespace(SUM=0),
                         all_reduce=lambda *a,**k:events.append('reduce'), all_gather_object=gather)
    torch=SimpleNamespace(exp=lambda *a,**k:events.append('exp'), distributed=dist)
    monkeypatch.setitem(sys.modules,'torch',torch)
    monkeypatch.setitem(sys.modules,'torch.distributed',dist)
    monkeypatch.setattr(_native,'require_native',lambda _:SimpleNamespace(low_latency_ranks_per_group=4))
    monkeypatch.setattr(st,'completion_boundary',lambda *a:events.append('boundary'))
    monkeypatch.setattr(st,'agree_check',lambda fn,*a:fn())
    monkeypatch.setattr(st,'case_config',lambda *a:dict(B=1,M=32))
    class Trace:
        def complete(self):events.append('copy')
        def save(self,path,meta):events.append('save');saved.append(meta)
    @contextmanager
    def capture(*a,prepared,defer_complete,**kw):
        assert defer_complete
        events.append('capture')
        yield prepared
    @contextmanager
    def profile(directory):
        events.append('profile_begin')
        yield SimpleNamespace(step=lambda:events.append('profile_step'))
        events.append('profile_end')
    def parse(directory,operation,count,rank):
        events.append('parse_profile')
        return dict(rank=rank,operation=operation,expected_samples=count,status='available',
                    measurement='instrumented_full_device_kernel_task_us',kernel_us=[20,21])
    monkeypatch.setattr(timer,'capture',capture)
    monkeypatch.setattr(st,'active_capture_profiler',profile)
    monkeypatch.setattr(st,'read_active_capture_profile',parse)
    def invoke():
        events.append('invoke')
        return (None,SimpleNamespace(current_stream_wait=lambda:events.append('wait')))
    args=SimpleNamespace(run_directory=str(tmp_path),case_id=0,ll_timer_iters=2,ll_profile_capture=True,
                         ll_timer_mode='observe',ll_timer_frequency_hz=1e9,seed=0,
                         warmup=10,num_iters=100,delay_rows=1,delay_cols=1,group_backend='hccl')
    st.diagnostic_calls(object(),invoke,lambda out:events.append('check'),args,'dispatch',0,32,'normal',
                         prepared=[Trace(),Trace()],prelude_buffers=(Tensor(),Tensor()))
    assert events==['profile_begin']+['exp','fill','reduce','capture','invoke','wait']*2+[
        'boundary','profile_step','profile_end','parse_profile','gather_profile']+['copy','check','save']*2
    assert [m['active_capture_profile']['kernel_us'] for m in saved]==[20,21]
    result=json.loads((tmp_path/'case_00000/normal/active_capture_dispatch.json').read_text())
    assert result['status']=='available' and result['correctness']=='passed'
    assert result['summary']['max_rank_mean_kernel_us']==20.5
