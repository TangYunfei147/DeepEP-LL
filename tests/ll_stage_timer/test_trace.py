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
    assert 'displayTimeUnit' not in trace  # Chrome rejects 'us'; ts/dur remain us.
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
    assert len({(e['pid'], e['tid']) for e in events})==32
    lanes=[e for e in trace['traceEvents'] if e['name']=='thread_name']
    assert len(lanes)==32 and all('/' not in e['args']['name'] for e in lanes)


def test_core_lanes_preserve_every_interval():
    core=timer.chrome_trace([sample()])
    stage=timer.chrome_trace([sample()], lane_layout='stage')
    def intervals(trace):
        return sorted((e['name'], e['ts'], e['dur'], e['args']['core'], e['args']['total_ticks'])
                      for e in trace['traceEvents'] if e['ph']=='X')
    assert intervals(core)==intervals(stage)
    assert len([e for e in stage['traceEvents'] if e['name']=='thread_name'])==32
    with pytest.raises(ValueError, match='lane layout'):
        timer.chrome_trace([sample()], lane_layout='unknown')


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


def test_unified_analysis_latest_and_all_samples(tmp_path):
    import os
    spec = importlib.util.spec_from_file_location('unified_analyzer', ROOT/'tests/ll_stage_timer/analyze.py')
    analyzer = importlib.util.module_from_spec(spec); spec.loader.exec_module(analyzer)
    for name, stamp in (('old', 100), ('new', 200)):
        run = tmp_path/name; run.mkdir()
        (run/'cases.json').write_text('{}')
        os.utime(run/'cases.json', (stamp, stamp))
    run = tmp_path/'new'
    for case in (0, 1):
        directory=run/f'case_{case:05d}'/'normal'/'ll_stage_timer'; directory.mkdir(parents=True)
        for iteration in (0, 1):
            saved=sample(); saved['frequency_hz']=None
            saved['metadata'].update(world_size=1, correctness='passed', config={'B':32},
                                     case_id=case, iteration=iteration)
            (directory/f'sample_{iteration}.json').write_text(json.dumps(saved))
    selected, directories=analyzer.discover(tmp_path, latest=True)
    assert selected == run and len(directories)==2
    for directory in directories:
        output=run/'stage-analysis'/directory.relative_to(run)
        analyzer.process_directory(directory, output, 1e9, 1)
        report=json.loads((output/'summary.json').read_text())
        assert report['per_rank_core_stage'][0]['samples']==2
        assert report['per_rank_core_stage'][0]['unit']=='us'
        trace=json.loads((output/'trace.json').read_text())
        assert {e['args']['iteration'] for e in trace['traceEvents'] if e['ph']=='X'}=={1}
    assert analyzer.discover(tmp_path, latest=True)[0] == run
    # Partial latest run must fail, never silently fall back to older data.
    victim=directories[0]/'sample_0.json'
    saved=json.loads(victim.read_text()); saved['metadata']['world_size']=2
    victim.write_text(json.dumps(saved))
    with pytest.raises(ValueError): analyzer.process_directory(directories[0], run/'bad', 1e9, 0)


def test_named_run_no_overwrite_or_escape(tmp_path):
    from tests.elastic import test_ep_low_latency as st
    args=st.parse_args(['--run-name', 'drain-001'])
    assert args.ll_timer_frequency_hz==1e9
    output=Path(st.create_run_directory(tmp_path, args.run_name))
    assert output==tmp_path/'drain-001'
    with pytest.raises(FileExistsError): st.create_run_directory(tmp_path, args.run_name)
    for bad in ('', '.', '..', '../escape', '/absolute', 'a\\b'):
        with pytest.raises(ValueError): st.create_run_directory(tmp_path, bad)


def test_declared_diagnostic_tail_must_be_complete(tmp_path):
    spec=importlib.util.spec_from_file_location('tail_analyzer', ROOT/'tests/ll_stage_timer/analyze.py')
    analyzer=importlib.util.module_from_spec(spec); spec.loader.exec_module(analyzer)
    saved=sample()
    saved['metadata'].update(world_size=1, config={'B':32}, correctness='passed',
                             sampling={'diagnostic_iters':2})
    path=tmp_path/'sample_0000.json'; path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match='Incomplete diagnostic tail'):
        analyzer.analyze([path], 1e9)
