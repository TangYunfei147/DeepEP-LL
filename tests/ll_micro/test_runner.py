import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('ll_micro_run',Path(__file__).with_name('run.py'))
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)

@pytest.mark.parametrize('text',[
    'unknown npu-smi format',
    '| 0 | Ascend950DT |\n| 1 | Ascend950DT |\nProcess id in container\nNo running processes found in NPU 0',
    '| 0 | Ascend950DT |\nProcess id in container\n| 0 | 1234 | python |',
])
def test_busy_or_unknown_stops(monkeypatch,text):
    monkeypatch.setattr(runner.subprocess,'check_output',lambda *a,**k:text)
    with pytest.raises(RuntimeError):runner.idle()

def test_all_devices_must_be_idle(monkeypatch):
    text='Process id in container\n| 0 | Ascend950DT |\n| 1 | Ascend950DT |\nNo running processes found in NPU 0\nNo running processes found in NPU 1'
    monkeypatch.setattr(runner.subprocess,'check_output',lambda *a,**k:text)
    assert runner.idle()==text

def test_tail50_remains_per_core(tmp_path):
    path=tmp_path/'scalar.csv'
    path.write_text('sample,core,mode,repeats,count,ticks,checksum,event_us\n'+''.join(
        f'{i},{c},1,32,1,{100000 if i<50 else (c+1)*64},0,2\n'
        for i in range(100) for c in range(2)))
    rows=runner.summarize(path)
    assert [r['samples'] for r in rows]==[50,50]
    assert [r['ticks_per_repeat_p50'] for r in rows]==[2,4]
