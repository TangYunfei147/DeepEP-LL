"""Trace durations, clocks, automatic export and relocatable offline conversion."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('timer_trace_test', ROOT/'deep_ep/utils/ll_stage_timer.py')
timer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(timer)


def sample(offset=10**17, rank=0):
    raw = [[[0]*8 for _ in range(32)] for _ in range(16)]
    for core, rows in enumerate(raw):
        rows[0] = [timer.MAGIC, 1, 1, 16, core, 2, 1, len(timer.STAGES)]
        rows[1] = [offset+100, offset+200, 100, 1, offset+100, 0, 0, 0]
        rows[15] = [offset+110, offset+180, 30, 3, offset+170, 0, 0, 0]
    return dict(raw=raw, operation='dispatch', frequency_hz=1e9,
                metadata=dict(rank=rank, iteration=0, run_id='test', case_id=0))


def test_exact_ticks_envelope_and_independent_clocks():
    trace = timer.chrome_trace([sample(), sample(2*10**17, 1)])
    events = [e for e in trace['traceEvents'] if e['ph']=='X']
    assert len(events) == 64
    for pid in (1, 2):
        kernel = next(e for e in events if e['pid']==pid and e['name']=='kernel')
        assert kernel['ts']==0 and kernel['dur']==pytest.approx(.1)
        wait = next(e for e in events if e['pid']==pid and e['name'].startswith('recv_wait'))
        assert wait['ts']==pytest.approx(.01) and wait['dur']==pytest.approx(.07)
        assert wait['args']['total_us']==pytest.approx(.03)
        assert wait['args']['gap_ticks']==40
        assert wait['args']['interval_kind']=='repeated envelope'
    assert len({(e['pid'], e['tid']) for e in events})==64


@pytest.mark.parametrize('frequency', [None, 0, -1, float('nan'), float('inf')])
def test_explicit_valid_frequency_required(frequency):
    saved=sample(); saved['frequency_hz']=frequency
    with pytest.raises(ValueError): timer.chrome_trace([saved])


def test_duplicate_and_corrupt_records_rejected():
    with pytest.raises(ValueError, match='Duplicate'): timer.chrome_trace([sample(), sample()])
    saved=sample(); saved['raw'][0][15][5]=1
    with pytest.raises(ValueError, match='Unclosed'): timer.chrome_trace([saved])


@pytest.mark.parametrize('frequency', [None, 1e9])
def test_capture_save_auto_export(tmp_path, frequency):
    saved=sample()
    obj=object.__new__(timer.StageTrace)
    obj.raw=saved['raw']; obj.operation='dispatch'; obj.frequency_hz=frequency
    obj.report=timer.parse_records(obj.raw, obj.operation, frequency)
    obj.save(tmp_path/'sample_0000.json', saved['metadata'])
    assert len(list(tmp_path.glob('sample_*.json')))==1
    output=tmp_path/'trace_sample_0000.json'
    assert output.exists() == (frequency is not None)
    if output.exists(): assert json.loads(output.read_text())['traceEvents']


def test_offline_moved_checkout_filters_and_no_overwrite(tmp_path):
    moved=tmp_path/'project with spaces'
    for relative in ('deep_ep/utils/ll_stage_timer.py', 'tests/ll_stage_timer/export_trace.py'):
        target=moved/relative; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/relative, target)
    data=moved/'data'; data.mkdir()
    for rank in (0, 1):
        saved=sample(rank=rank); saved['frequency_hz']=None
        (data/f'sample_{rank}.json').write_text(json.dumps(saved))
    command=[sys.executable, str(moved/'tests/ll_stage_timer/export_trace.py'), 'data',
             '--frequency-hz', '1000000000', '--rank', '1', '--iteration', '0',
             '--operation', 'dispatch', '--output', 'result/trace.json']
    subprocess.run(command, cwd=tmp_path, check=True)
    output=moved/'result/trace.json'
    trace=json.loads(output.read_text())
    assert {e['args']['rank'] for e in trace['traceEvents'] if e['ph']=='X'}=={1}
    before=output.read_bytes()
    assert subprocess.run(command, cwd=tmp_path, capture_output=True).returncode!=0
    assert output.read_bytes()==before
