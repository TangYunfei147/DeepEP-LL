"""Only the current reference capture may attach, with exact native layout."""
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[2]


def load(name,relative):
    spec=importlib.util.spec_from_file_location(name,ROOT/relative)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


@pytest.fixture
def capture_module(monkeypatch):
    for name in ('deep_ep','deep_ep.utils'):
        package=ModuleType(name);package.__path__=[];monkeypatch.setitem(sys.modules,name,package)
    parser=load('deep_ep.utils.ll_reference_profile_parse','deep_ep/utils/ll_reference_profile_parse.py')
    monkeypatch.setitem(sys.modules,'deep_ep.utils.ll_reference_profile_parse',parser)
    module=load('deep_ep.utils.ll_reference_profile','deep_ep/utils/ll_reference_profile.py')
    monkeypatch.setitem(sys.modules,'deep_ep.utils.ll_reference_profile',module)
    return module


def test_current_layout_alignment_and_retained_capture(monkeypatch,tmp_path,capture_module):
    module=capture_module;events=[]
    class Tensor:
        device='npu:3'
        def __init__(self):self.address=520
        def data_ptr(self):return self.address
        def narrow(self,axis,offset,count):self.address+=offset*8;self.count=count;return self
        def view(self,*shape):self.shape=shape;return self
        def cpu(self):events.append('copy');return self
        def tolist(self):return [[[123]]]
    layout=dict(schema=module.SCHEMA,slots=module.SLOTS,fields=module.FIELDS,words=list(module.WORDS),
                implementation='mte-shared-receive',send_cores=30,recv_cores=26)
    native=SimpleNamespace(low_latency_reference_profile_enabled=True,
        low_latency_reference_profile_layout=lambda:layout,
        low_latency_core_config=lambda:dict(active_aiv=56,send_cores=30,recv_cores=26,device=3,runtime_device_queried=True))
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(int64=0,zeros=lambda *a,**k:Tensor()))
    monkeypatch.setitem(sys.modules,'deep_ep._native',SimpleNamespace(require_native=lambda _:native))
    monkeypatch.setattr(module,'parse_records',lambda *a:dict(schema=6,records=[],send_cores=30,recv_cores=26))
    buffer=object();trace=module.ReferenceTrace(buffer)
    assert trace.tensor.shape==(56,80,8) and trace.tensor.count==56*640
    assert trace.tensor.data_ptr()%512==0
    tensors={'input':SimpleNamespace(device='npu:3')}
    with module.capture(buffer,prepared=trace,defer_complete=True):
        module.attach(buffer,tensors,'dispatch')
        assert set(tensors)=={'input','ll_reference_profile'}
        with pytest.raises(RuntimeError,match='exactly one'):module.attach(buffer,tensors,'dispatch')
        with pytest.raises(RuntimeError,match='Nested'):
            with module.capture(buffer,prepared=trace):pass
    assert not events and not module._state.active
    trace.complete();trace.save(tmp_path/'sample_0000.json',dict(rank=3,profile_seed=17))
    assert events==['copy']
    saved=json.loads((tmp_path/'sample_0000.json').read_text())
    assert saved['format']==module.FORMAT and saved['metadata']['profile_seed']==17
    with pytest.raises(FileExistsError):trace.save(tmp_path/'sample_0000.json')
    for key,value in [('schema',5),('slots',8),('implementation','metadata-cores'),('send_cores',28)]:
        old=layout[key];layout[key]=value
        with pytest.raises(RuntimeError,match='layout mismatch'):module.ReferenceTrace(object())
        layout[key]=old
    native.low_latency_reference_profile_enabled=1
    with pytest.raises(RuntimeError,match='Rebuild'):module.ReferenceTrace(object())


@pytest.mark.parametrize('operation,mode',[('combine','reference'),('dispatch','observe'),('dispatch','drain')])
def test_other_operations_and_modes_rejected_before_native_import(capture_module,operation,mode):
    with pytest.raises(ValueError,match='dispatch with reference'):capture_module.ReferenceTrace(object(),operation,mode)


@pytest.mark.parametrize('frequency',[0,-1,True,float('nan'),float('inf')])
def test_frequency_requires_positive_real(capture_module,frequency):
    with pytest.raises(ValueError,match='frequency'):capture_module.ReferenceTrace(object(),frequency_hz=frequency)


def test_capture_rejects_combine_before_attachment(capture_module):
    module=capture_module;buffer=object()
    trace=module.ReferenceTrace.__new__(module.ReferenceTrace)
    trace.buffer=buffer;trace.operation='dispatch';trace.mode='reference';trace.frequency_hz=1e9;trace.launches=0
    trace.tensor=SimpleNamespace(device='npu:0')
    module._state.active={id(buffer):trace}
    with pytest.raises(RuntimeError,match='never combine'):
        module.attach(buffer,{'input':SimpleNamespace(device='npu:0')},'combine')
    assert trace.launches==0


def test_public_buffer_only_attaches_reference_on_dispatch():
    source=(ROOT/'deep_ep/buffers/elastic.py').read_text()
    assert source.count('from ..utils.ll_reference_profile import attach')==1
    assert 'attach(self, tensors, "dispatch")' in source
    assert 'attach(self, tensors, "combine")' not in source
    assert 'll_stage_timer' not in source and 'll_expert_timer' not in source
