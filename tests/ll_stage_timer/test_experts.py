"""CPU attribution, interval validation, export and portable offline pipeline checks."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]
def load(name, relative):
    spec=importlib.util.spec_from_file_location(name,ROOT/relative)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
expert=load('expert_parser_test','deep_ep/utils/ll_expert_timer.py')
timer=load('stage_parser_expert_test','deep_ep/utils/ll_stage_timer.py')
analyzer=load('expert_analyzer_test','tests/ll_stage_timer/analyze.py')


def data(rank=0, world=16, experts=16):
    raw=[[[0]*8 for _ in range(8+experts*4)] for _ in range(16)]
    for core,rows in enumerate(raw):
        rows[0]=[expert.MAGIC,1,rank,core,experts,1,1,4]
    owner=rank*(experts//world)
    for core in (8,9):
        raw[core][8+owner*4]=[100,200,0,2,0,0,0,0]
        raw[core][8+owner*4+1]=[110,190,20,2,180,0,0,0]
        raw[core][8+owner*4+2]=[120,210,40,2,190,0,0,0]
    raw[8][8+(owner+1)*4+3]=[150,180,30,1,150,0,0,0]
    return raw


def stage_raw(rank):
    raw=[[[0]*8 for _ in range(32)] for _ in range(16)]
    for core,rows in enumerate(raw):
        rows[0]=[timer.MAGIC,1,1,16,core,1,1,len(timer.LEGACY_STAGES)]
        rows[1]=[90,220,130,1,90,0,0,0]
        for name in timer.NOISY_SEND_ENVELOPES:
            index=timer.LEGACY_STAGES.index(name) if name in timer.LEGACY_STAGES else None
            if index is not None: rows[index]=[100,210,20,2,200,0,0,0]
    return raw


def saved(rank=0, world=16, experts=16):
    return dict(raw=stage_raw(rank),operation='dispatch',frequency_hz=1e9,
                metadata=dict(rank=rank,world_size=world,iteration=0,case_id=0,run_id='test',
                              correctness='passed',config={'num_experts':experts}),
                expert=dict(raw=data(rank,world,experts),rank=rank,world=world,experts=experts))


def test_attribution_separates_local_work_and_forward_target():
    report=expert.parse_records(data(),0,16,16,1e9)
    local=[r for r in report['records'] if r['stage']!='forward']
    assert all(r['expert_id']==0 and r['expert_rank']==0 for r in local)
    forward=next(r for r in report['records'] if r['stage']=='forward')
    assert forward['expert_id']==1 and forward['executing_rank']==0 and forward['expert_rank']==1
    assert forward['total_us']==pytest.approx(.03)
    assert sum(r['calls'] for r in local if r['stage']=='packets')==4


@pytest.mark.parametrize('damage',['active','wrong_rank','wrong_owner','padding','sender','bad_sum','cross_host','header','missing_core'])
def test_bad_expert_records_fail(damage):
    raw=data(world=32,experts=32)
    if damage=='active': raw[8][9][5]=1
    if damage=='wrong_rank': raw[8][0][2]=1
    if damage=='wrong_owner': raw[8][8+4+1]=raw[8][9];raw[8][9]=[0]*8
    if damage=='padding': raw[0][1][0]=1
    if damage=='sender': raw[0][9]=raw[8][9]
    if damage=='bad_sum': raw[8][9][2]=100
    if damage=='cross_host': raw[8][8+16*4+3]=raw[8][8+4+3];raw[8][8+4+3]=[0]*8
    if damage=='header': raw[8][0][6]=0
    if damage=='missing_core': raw.pop()
    with pytest.raises(ValueError): expert.parse_records(raw,0,32,32)


def test_cross_core_sum_is_distinct_from_span(tmp_path):
    path=tmp_path/'sample_0.json';path.write_text(json.dumps(saved()))
    rows,summary=analyzer.expert_reports([path],1e9)
    result=next(r for r in summary if r['expert_id']==0 and r['stage']=='output_write')
    assert result['mean']==pytest.approx(.08) and result['mean_span']==pytest.approx(.09)
    assert result['calls_total']==4 and len(rows)==7
    packets=next(r for r in summary if r['stage']=='packets')
    assert packets['mean'] is None and packets['duration_kind']=='counter_only'
    assert not any(r['stage']=='shared_prepare' for r in rows)


def test_noisy_envelopes_hidden_only_from_trace():
    sample=saved()
    visible=timer.chrome_trace([sample],show_repeated=True)
    compact=timer.chrome_trace([sample])
    names=lambda trace:{e['args']['stage'] for e in trace['traceEvents'] if e['ph']=='X'}
    assert 'send_issue' in names(visible) and 'send_issue' not in names(compact)
    assert 'send_issue' in {r['stage'] for r in timer.parse_records(sample['raw'],'dispatch')['records']}
    assert 'kernel' in names(compact)


def test_full_offline_export_after_move(tmp_path):
    moved=tmp_path/'project with spaces'
    for name in ('deep_ep/utils/ll_stage_timer.py','deep_ep/utils/ll_stage_timer_legacy.py','deep_ep/utils/ll_stage_timer_sparse.py','deep_ep/utils/ll_expert_timer.py','tests/ll_stage_timer/analyze.py','tests/ll_stage_timer/relay_timing.py','tests/ll_stage_timer/communication_report.py','tests/ll_stage_timer/transfer_report.py','tests/ll_stage_timer/text_codec.py'):
        path=moved/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,path)
    root=moved/'data';root.mkdir()
    # One-rank synthetic parser fixture, not a device launch.
    sample=saved(world=1,experts=16)
    (root/'sample_0.json').write_text(json.dumps(sample))
    subprocess.run([sys.executable,str(moved/'tests/ll_stage_timer/analyze.py'),'data','--output-dir','report'],cwd=tmp_path,check=True)
    assert (moved/'report/experts.csv').is_file() and (moved/'report/expert_cores.csv').is_file()
    assert json.loads((moved/'report/summary.json').read_text())['per_expert']


def test_missing_expert_payload_rejected(tmp_path):
    sample=saved();sample.pop('expert');sample['metadata']['sampling']={'expert_timer':True}
    path=tmp_path/'sample_0.json';path.write_text(json.dumps(sample))
    with pytest.raises(ValueError,match='Missing requested'): analyzer.expert_reports([path],1e9)


def test_reference_capture_rejects_legacy_expert_instrumentation():
    with pytest.raises(ValueError,match='per-expert'):
        timer.StageTrace(object(),'dispatch',expert_config=dict(rank=0,world=16,experts=16))


def test_expert_cli_implies_stage_capture():
    st=load('expert_cli','tests/elastic/test_ep_low_latency.py')
    args=st.parse_args(['--ll-expert-timer'])
    with pytest.raises(ValueError,match='per-expert'): st.validate_args(args,16)
