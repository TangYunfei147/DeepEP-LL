"""CPU-only checks for timing boundaries, task identity and capture windows."""
import csv
from pathlib import Path
from types import SimpleNamespace as NS
import sys

import pytest

from tests.ll_probe.profiling import profile_common as pc


def test_reference_capture_sequence(monkeypatch, tmp_path):
    calls = []
    class Tensor:
        def fill_(self, x):
            calls.append(('fill', x))
    def tensor(*shape, **kwargs):
        calls.append(('allocate', shape))
        return Tensor()
    class Profiler:
        def __enter__(self): calls.append('enter'); return self
        def __exit__(self, *args): calls.append('exit')
        def step(self): calls.append('step')
    def profile(**kwargs):
        assert kwargs['schedule'] == dict(wait=0, warmup=1, active=1, repeat=1, skip_first=0)
        assert kwargs['profile_memory'] and not kwargs['with_stack']
        return Profiler()
    dist = NS(is_initialized=lambda: True, get_backend=lambda: 'hccl',
              all_reduce=lambda *a, **k: calls.append('reduce'),
              barrier=lambda: calls.append('barrier'), ReduceOp=NS(SUM='sum'))
    torch = NS(zeros=tensor, ones=tensor, float32='fp32', distributed=dist,
               exp=lambda *a, **k: calls.append('exp'), npu=NS(synchronize=lambda: calls.append('sync')))
    npu = NS(profiler=NS(profile=profile, schedule=lambda **kw: kw,
        tensorboard_trace_handler=lambda path: path, _ExperimentalConfig=lambda **kw: kw,
        AiCMetrics=NS(PipeUtilization='pipe'), ProfilerLevel=NS(Level1=1)))
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'torch.distributed', dist)
    monkeypatch.setitem(sys.modules, 'torch_npu', npu)
    pc.run_profiled(lambda: calls.append('invoke'), tmp_path)
    assert calls[:4] == [('allocate', (118260, 30720)), ('allocate', (1,)), 'reduce', 'enter']
    assert calls[4:16] == ['invoke']*10+['sync', 'step']
    assert calls[16:-4] == ['exp', ('fill', 1), 'reduce', 'invoke']*100
    assert calls[-4:] == ['barrier', 'sync', 'step', 'exit']


def write_tasks(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['Kernel Name', 'Task Type', 'Task Duration(us)', 'Task Start Time(us)', 'Task Wait Time(us)'])
        writer.writerows(rows)


def test_tail_per_task_sorting_and_format_preference(tmp_path):
    rows = [('four_metadata', kind, i+1, i, .5) for kind in ('AI_VECTOR_CORE', 'AI_CPU') for i in reversed(range(100))]
    rows += [('four_metadata_check', 'AI_VECTOR_CORE', 900, 1, 0), ('Exp', 'AI_VECTOR_CORE', 800, 1, 0)]
    write_tasks(tmp_path/'op_summary_0.csv', rows)
    write_tasks(tmp_path/'kernel_details.csv', [('four_metadata', 'AI_VECTOR_CORE', 999, 1, 0)])
    groups = pc.read_named_tasks(tmp_path, 'four_metadata', 100)
    assert len(groups) == 2
    for g in groups:
        assert g['kernel_us'] == list(range(1, 101))
        assert g['selected_kernel_us'] == list(range(51, 101))
        assert g['duration']['mean_us'] == 75.5
        assert g['duration']['sample_count'] == 50
        assert g['source_format'] == 'op_summary*.csv'
    with pytest.raises(ValueError, match='expected 101'):
        pc.read_named_tasks(tmp_path, 'four_metadata', 101)


def test_missing_and_nonfinite_tasks_rejected(tmp_path):
    write_tasks(tmp_path/'op_summary.csv', [('four_empty', 'AI_VECTOR_CORE', 'nan', 1, 0)])
    with pytest.raises(ValueError, match='Invalid task timing'):
        pc.read_named_tasks(tmp_path, 'four_empty', 1)
    with pytest.raises(ValueError, match='No named kernel'):
        pc.read_named_tasks(tmp_path, 'four_comm_1', 1)
