"""Synthetic evidence verifies report arithmetic, not real hardware performance."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('report_analyzer', Path(__file__).with_name('analyze.py'))
analyzer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyzer)


def saved(rank=0, iteration=0, operation='dispatch', hidden=3584, schema=3, frequency=1e9, mode=1, flags=0):
    raw = []
    for core in range(16):
        # Clock offsets deliberately differ; only per-core spans are comparable.
        start = rank*1000000 + core*10000
        slow = core == 10 + iteration % 2
        duration = (1000 + rank*10) * (iteration+1) * (4 if slow else 1)
        words = [0]*256
        words[:8] = [analyzer.timer.MAGIC, schema, 1 if operation=='dispatch' else 2,
                     16, core, mode, 1, len(analyzer.timer.STAGES)]
        words[8:12] = [start, start+duration, duration, 1]
        if slow:
            stages = {'recv_wait': .5, 'recv_consume': .6, 'recv_forward': .4} if operation=='dispatch' else {'combine_partial': .6}
            for stage, fraction in stages.items():
                offset = 8 + (analyzer.timer.STAGES.index(stage)-1)*4
                elapsed = int(duration*fraction)
                words[offset:offset+4] = [start+5, start+5+elapsed, elapsed, 1]
        if schema==3:
            words[208:212] = [hidden, 1, 0, 0]
            words[212:216] = [hidden*2, 2, hidden, 1]
            words[216:220] = [hidden, 1, 0, 0]
            words[228] = flags
        raw.append([words[n:n+8] for n in range(0,256,8)])
    return dict(raw=raw, operation=operation, frequency_hz=frequency,
                metadata=dict(rank=rank, world_size=16, iteration=iteration, correctness='passed',
                              run_id='synthetic-test', case_id=0, seed=7,
                              config=dict(hidden=hidden, num_tokens=256, num_topk=8, dtype='fp8'),
                              sampling=dict(diagnostic_iters=3)))


def inputs(directory, **kwargs):
    directory.mkdir(parents=True)
    paths=[]
    for operation in ('dispatch','combine'):
        for rank in range(16):
            for iteration in range(3):
                path=directory/f'sample_{operation}_{rank}_{iteration}.json'
                path.write_text(json.dumps(saved(rank,iteration,operation,**kwargs)),encoding='utf-8')
                paths.append(path)
    return paths


def test_slow_core_ratios_are_same_sample_not_nested_sums(tmp_path):
    report=analyzer.analyze(inputs(tmp_path/'data'))
    facts=report['diagnostics']
    rank=next(row for row in facts['rank_kernel'] if row['rank']==0 and row['operation']=='dispatch')
    assert rank['mean']==8  # (4+8+12) us, not absolute-clock envelope
    share=next(row for row in facts['longest_core_stages']
               if row['rank']==0 and row['operation']=='dispatch' and row['stage']=='recv_wait')
    assert share['fraction']['mean']==pytest.approx(.5)
    assert share['observed_samples']==3
    assert {row['core'] for row in facts['longest_core_samples'] if row['rank']==0}=={10,11}
    document=analyzer.render_bottleneck_report(report)
    assert '## dispatch' in document and '## combine' in document
    assert '3584' in document and 'rank 15' in document
    assert '不能相加' in document and '不能单凭等待证明链路拥塞' in document
    assert '33.33%' in document  # forwarding / (first+forward), reads not double counted


@pytest.mark.parametrize('kwargs,text', [
    ({'schema':2},'counters_unavailable'), ({'frequency':None},'frequency_unavailable'),
    ({'flags':1},'nonstandard_skip_communication'), ({'mode':2},'drain主动串行化'),
])
def test_limitations_not_fabricated_measurements(tmp_path,kwargs,text):
    report=analyzer.analyze(inputs(tmp_path/'data',**kwargs))
    document=analyzer.render_bottleneck_report(report)
    assert text in document
    if kwargs.get('flags'):
        assert '待验证线索（' not in document
    if kwargs.get('frequency','present') is None:
        assert '累计时间均值 (ticks)' in document


def test_outside_kernel_stage_has_no_share():
    sample=saved()
    words=[word for row in sample['raw'][10] for word in row]
    offset=8+(analyzer.timer.STAGES.index('recv_wait')-1)*4
    words[offset:offset+4]=[200000,202000,2000,1]
    sample['raw'][10]=[words[n:n+8] for n in range(0,256,8)]
    report=analyzer.timer.parse_records(sample['raw'],'dispatch',1e9)
    result=analyzer.diagnostic_sample(report,sample['metadata'])
    assert next(row for row in result['stages'] if row['stage']=='recv_wait')['fraction'] is None


def test_fraction_averages_per_round_and_missing_stage_is_not_zero():
    samples=[]
    for iteration in range(3):
        sample=saved(iteration=iteration)
        core=10+iteration%2
        words=[word for slot in sample['raw'][core] for word in slot]
        offset=8+(analyzer.timer.STAGES.index('recv_wait')-1)*4
        if iteration==1:
            words[offset+1]=words[offset]+1600
            words[offset+2]=1600  # 20% of an 8000-tick kernel
        if iteration==2:
            words[offset:offset+4]=[0]*4
        sample['raw'][core]=[words[n:n+8] for n in range(0,256,8)]
        parsed=analyzer.timer.parse_records(sample['raw'],'dispatch',1e9)
        samples.append(analyzer.diagnostic_sample(parsed,sample['metadata']))
    report=analyzer.diagnostic_summary(samples)
    wait=next(row for row in report['longest_core_stages'] if row['stage']=='recv_wait')
    assert wait['observed_samples']==2 and wait['fraction']['samples']==2
    assert wait['fraction']['mean']==pytest.approx(.35)  # (.5 + .2)/2, not 3600/12000 or missing=0


def test_multiple_shapes_cli_documents_and_index(tmp_path):
    parent=tmp_path/'run with spaces'
    for number,hidden in enumerate((3584,7168)):
        inputs(parent/f'case_{number:05d}'/'normal'/'ll_stage_timer',hidden=hidden)
    output=tmp_path/'analysis with spaces'
    subprocess.run([sys.executable,str(ROOT/'tests/ll_stage_timer/analyze.py'),str(parent),
                    '--output-dir',str(output)],cwd=tmp_path,check=True,capture_output=True,text=True)
    index=(output/'REPORT_INDEX.md').read_text(encoding='utf-8')
    assert 'case_00000' in index and 'case_00001' in index
    for number,hidden in enumerate((3584,7168)):
        destination=output/f'case_{number:05d}'/'normal'/'ll_stage_timer'
        document=(destination/'bottleneck_summary.md').read_text(encoding='utf-8')
        assert str(hidden) in document
        assert 'recv_consume专项' in document and 'recv_load_wait' in document
        assert (destination/'recv_core_samples.csv').is_file()
        assert json.loads((destination/'summary.json').read_text())['configurations']['dispatch']['hidden']==hidden
    before=(output/'REPORT_INDEX.md').read_bytes()
    rejected=subprocess.run([sys.executable,str(ROOT/'tests/ll_stage_timer/analyze.py'),str(parent),
                             '--output-dir',str(output)],cwd=tmp_path,capture_output=True,text=True)
    assert rejected.returncode and 'already exists' in rejected.stderr
    assert (output/'REPORT_INDEX.md').read_bytes()==before


def test_receiver_includes_noncritical_and_unobserved_cores():
    sample=saved()
    words=[value for row in sample['raw'][12] for value in row]
    offset=8+(analyzer.timer.STAGES.index('recv_consume')-1)*4
    words[offset:offset+4]=[120005,120705,700,2]
    sample['raw'][12]=[words[n:n+8] for n in range(0,256,8)]
    report=analyzer.timer.parse_records(sample['raw'],'dispatch',1e9)
    rows=analyzer.receiver_samples(report,sample['metadata'])
    assert {row['core'] for row in rows}==set(range(8,16))
    assert next(row for row in rows if row['core']==12 and row['stage']=='recv_consume')['per_call']==pytest.approx(.35)
    absent=next(row for row in rows if row['core']==13 and row['stage']=='recv_consume')
    assert absent['present'] is False and absent['duration'] is None
    assert sum(row['forward_bytes'] or 0 for row in rows)==8*3584  # no repeated count per stage


def test_receiver_call_weighting_missing_and_legacy_counts():
    all_rows=[]
    for iteration,(duration,calls) in enumerate(((10,2),(90,3),(None,None))):
        report=dict(operation='dispatch',active_aiv=16,frequency_hz=1e6,traffic_available=False,traffic=None,
                    records=[] if duration is None else [dict(core=8,stage='recv_consume',first_tick=0,last_tick=100,
                                                              total_ticks=duration,calls=calls)])
        all_rows.extend(analyzer.receiver_samples(report,dict(rank=0,iteration=iteration)))
    entry=next(row for row in analyzer.receiver_summary(all_rows) if row['core']==8 and row['stage']=='recv_consume')
    assert entry['samples']==3 and entry['observed_samples']==2
    assert entry['per_call']['mean']==17.5  # (10/2+90/3)/2
    assert entry['weighted_per_call']==20  # (10+90)/(2+3), separate metric
    assert entry['forward_bytes'] is None
    assert analyzer.receiver_samples(dict(operation='combine'),{})==[]
