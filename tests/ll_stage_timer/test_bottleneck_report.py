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


from event_fixtures import raw as event_raw, set_events


def saved(rank=0, iteration=0, operation='dispatch', hidden=3584, frequency=1e9, mode=1, flags=0):
    data=event_raw(operation,mode)
    for core,entry in enumerate(data['cores']):
        start=rank*1000000+core*10000
        slow=core==10+iteration%2
        duration=(1000+rank*10)*(iteration+1)*(4 if slow else 1)
        set_events(data,core,'kernel',[[start,start+duration]])
        if slow:
            stages={'recv_wait':.5,'recv_metadata':.3,'recv_forward':.1} if operation=='dispatch' else {'combine_output_task':.6}
            cursor=start+5
            for stage,fraction in stages.items():
                elapsed=int(duration*fraction)
                set_events(data,core,stage,[[cursor,cursor+elapsed]])
                cursor+=elapsed
        entry['traffic']=[[hidden,1,0,0],[hidden*2,2,hidden,1],[hidden,1,0,0],[0]*4,[0]*4]
        entry['flags']=flags
    return dict(raw=data,operation=operation,frequency_hz=frequency,
                metadata=dict(rank=rank,world_size=16,iteration=iteration,correctness='passed',
                              run_id='synthetic-test',case_id=0,seed=7,
                              config=dict(hidden=hidden,num_tokens=256,num_topk=8,dtype='fp8'),
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
    assert '片段互斥' in document and '不能单凭等待证明链路拥塞' in document
    assert '33.33%' in document  # forwarding / (first+forward), reads not double counted


@pytest.mark.parametrize('kwargs,text', [
    ({'frequency':None},'frequency_unavailable'),
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


def test_outside_kernel_stage_is_rejected():
    sample=saved()
    set_events(sample['raw'],10,'recv_wait',[[200000,202000]])
    with pytest.raises(ValueError,match='outside'):
        analyzer.timer.parse_records(sample['raw'],'dispatch',1e9)


def test_fraction_averages_per_round_and_missing_stage_is_not_zero():
    samples=[]
    for iteration in range(3):
        sample=saved(iteration=iteration)
        core=10+iteration%2
        if iteration==1:
            start=sample['raw']['cores'][core]['events']['recv_wait'][0][0]
            set_events(sample['raw'],core,'recv_wait',[[start,start+1600]])
        if iteration==2:
            set_events(sample['raw'],core,'recv_wait',[])
        parsed=analyzer.timer.parse_records(sample['raw'],'dispatch',1e9)
        samples.append(analyzer.diagnostic_sample(parsed,sample['metadata']))
    wait=next(row for row in analyzer.diagnostic_summary(samples)['longest_core_stages'] if row['stage']=='recv_wait')
    assert wait['observed_samples']==2 and wait['fraction']['samples']==2
    assert wait['fraction']['mean']==pytest.approx(.35)


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
        assert '接收任务' in document and 'recv_load_wait' in document
        assert (destination/'recv_core_samples.csv').is_file()
        assert json.loads((destination/'summary.json').read_text())['configurations']['dispatch']['hidden']==hidden
    before=(output/'REPORT_INDEX.md').read_bytes()
    rejected=subprocess.run([sys.executable,str(ROOT/'tests/ll_stage_timer/analyze.py'),str(parent),
                             '--output-dir',str(output)],cwd=tmp_path,capture_output=True,text=True)
    assert rejected.returncode and 'already exists' in rejected.stderr
    assert (output/'REPORT_INDEX.md').read_bytes()==before


def test_receiver_includes_noncritical_and_unobserved_cores():
    sample=saved()
    set_events(sample['raw'],12,'recv_metadata',[[120005,120305],[120305,120705]])
    report=analyzer.timer.parse_records(sample['raw'],'dispatch',1e9)
    rows=analyzer.receiver_samples(report,sample['metadata'])
    assert {row['core'] for row in rows}==set(range(8,16))
    assert next(row for row in rows if row['core']==12 and row['stage']=='recv_metadata')['per_call']==pytest.approx(.35)
    absent=next(row for row in rows if row['core']==13 and row['stage']=='recv_metadata')
    assert absent['present'] is False and absent['duration'] is None
    assert sum(row['forward_bytes'] or 0 for row in rows)==8*3584  # no repeated count per stage


def test_receiver_call_weighting_missing_and_zero_counts():
    all_rows=[]
    for iteration,(duration,calls) in enumerate(((10,2),(90,3),(None,None))):
        report=dict(operation='dispatch',active_aiv=16,frequency_hz=1e6,traffic=[],
                    records=[dict(core=core,stage='kernel',total_ticks=100) for core in range(8,16)]+([] if duration is None else [dict(core=8,stage='recv_metadata',first_tick=0,last_tick=100,total_ticks=duration,calls=calls,segments=calls)]))
        all_rows.extend(analyzer.receiver_samples(report,dict(rank=0,iteration=iteration)))
    entry=next(row for row in analyzer.receiver_summary(all_rows) if row['core']==8 and row['stage']=='recv_metadata')
    assert entry['samples']==3 and entry['observed_samples']==2
    assert entry['per_call']['mean']==17.5  # (10/2+90/3)/2
    assert entry['weighted_per_call']==20  # (10+90)/(2+3), separate metric
    assert entry['forward_bytes'] is None
    assert analyzer.receiver_samples(dict(operation='combine'),{})==[]
