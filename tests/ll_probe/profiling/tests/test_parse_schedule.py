"""CPU checks for the opt-in diagnostic and unchanged default measurement order."""
from pathlib import Path
import runpy
import sys
import types

REPO=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(REPO/'tests/ll_probe/profiling'))


def test_measure_keeps_capture_before_parse(monkeypatch,tmp_path):
    st=runpy.run_path(str(REPO/'tests/elastic/test_ep_low_latency.py'))
    ns=st['measure'].__globals__
    events=[]
    torch=types.ModuleType('torch'); dist=types.ModuleType('torch.distributed'); torch.distributed=dist
    monkeypatch.setitem(sys.modules,'torch',torch); monkeypatch.setitem(sys.modules,'torch.distributed',dist)
    monkeypatch.setitem(ns,'run_profiled',lambda *a,**k:events.append('capture'))
    def parse(args,operation,path,progress,kernel):
        events.append(('parse',operation,kernel)); return 'parsed'
    monkeypatch.setitem(ns,'read_measurement',parse)
    args=types.SimpleNamespace(warmup=10,num_iters=100,delay_rows=64,delay_cols=64)
    assert st['measure'](lambda:None,args,'dispatch',tmp_path)=='parsed'
    assert events==['capture',('parse','dispatch',None)]


def test_probe_refuses_other_devices(monkeypatch):
    from parse_schedule_probe import require_two_devices
    for key in ('PROBE_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES','ASCEND_RT_VISIBLE_DEVICES','ASCEND_VISIBLE_DEVICES'):
        monkeypatch.setenv(key,'0,1')
    require_two_devices()
    monkeypatch.setenv('ASCEND_RT_VISIBLE_DEVICES','0,2')
    import pytest
    with pytest.raises(ValueError): require_two_devices()
