"""Reproducible single-change shared-pull series and bounded target reports."""
import csv
import importlib.util
import io
import json
from pathlib import Path
import re
import subprocess

import pytest

ROOT=Path(__file__).resolve().parents[2]
DIRECTORY=ROOT/'tests/ll_control_opt'
FIXED=(
    ('baseline','78e9fb4d2686d73e8baf9dbdd633b9bf203fdd4d'),
    ('prefix-only','b2bcdbdcf311a19b2e321710ce084f2dcf9e138e'),
    ('pull-reference','d4c7c0a1669b6b4fda03d4028964852a2f13999e'),
    ('no-fence','2daff37d3b257f65528a892dc9b12f4ec91df8d1'),
    ('prepare-once','fdc4335f75b7aa0d6b9df8e940f55ebcc7ea97e8'),
    ('async-flags','10a437195b16d1c026088ec751527af8fd2e7773'),
    ('logical-role','68fe8d1f5415d38ab41194aac95d451db7ea8e18'),
)


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


suite=load('shared_pull_series_suite',DIRECTORY/'suite.py')
report=load('shared_pull_series_report',DIRECTORY/'export_report.py')


def git(*args):
    return subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()


def entries(kind):
    return suite.commits(DIRECTORY/f'commits-shared-pull-{kind}.json',ROOT)


def plan(kind):
    args=suite.parser().parse_args([
        '--commits',f'tests/ll_control_opt/commits-shared-pull-{kind}.json',
        '--devices',','.join(map(str,range(16))),'--num-nodes','1','--node-rank','0',
        '--master-addr','127.0.0.1','--master-port','29600','--shmem-port','29601',
        '--resume','--send-cores','30','--seeds','0','--rounds','2',
        '--order','alternating','--run-prefix',f'shared-pull-{kind}-001'])
    return suite.make_plan(args)


def test_series_pins_every_step_and_keeps_incremental_references():
    path=DIRECTORY/'commits-shared-pull-series.json'
    raw=json.loads(path.read_text(encoding='utf-8'))
    series=entries('series')
    assert raw['schema']==2 and len(series)==9
    assert raw['comparison_base'] in ('baseline',FIXED[0][1])
    assert [(item['label'],item['revision']) for item in series[:7]]==list(FIXED)
    for index,(source,resolved) in enumerate(zip(raw['commits'],series)):
        assert re.fullmatch(r'[0-9a-f]{40}',source['revision'])
        previous=series[max(0,index-1)]
        assert resolved['comparison_base']==previous['revision']
        assert resolved['comparison_label']==previous['label']
        if index:
            assert source['comparison_base'] in (previous['label'],previous['revision'])
        if index>=2:
            # Baseline lives on the restored branch; the experimental steps
            # from prefix-only onward must be individual cumulative commits.
            assert git('show','--format=%P','--no-patch',resolved['revision'])==previous['revision']


def test_fast_manifest_measures_current_baseline_prefix_fix_and_same_final_source():
    fast=entries('fast')
    series=entries('series')
    assert len(fast)==3
    assert [(item['label'],item['revision']) for item in fast[:2]]==list(FIXED[:2])
    assert fast[-1]['revision']==series[-1]['revision']
    raw=json.loads((DIRECTORY/'commits-shared-pull-fast.json').read_text(encoding='utf-8'))
    assert all(re.fullmatch(r'[0-9a-f]{40}',item['revision']) for item in raw['commits'])


def test_pinned_sources_record_actual_modes_and_prefix_wire_protocols(tmp_path):
    expected=[(0,0,1,19000),(0,1,1,25000)]+[(2,1,1,27000)]*5+[(2,1,2,28000)]*2
    for item,(mode,prefix,wire,base) in zip(entries('series'),expected):
        source=tmp_path/item['label']
        directory=source/'csrc/kernels/dispatch/low_latency/common'
        directory.mkdir(parents=True)
        for name in ('ll_core_config.h','ll_shared_pull_probe_config.h'):
            relative=f'csrc/kernels/dispatch/low_latency/common/{name}'
            if git('ls-tree','--name-only',item['revision'],'--',relative):
                (directory/name).write_text(git('show',item['revision']+':'+relative)+'\n',encoding='utf-8')
        identity=suite.build.source_protocol_identity(source)
        assert identity==dict(shared_pull_probe_mode=mode,prefix_ready_revision=prefix,
                              shared_pull_wire_revision=wire)
        assert suite.build.protocol_base(mode,prefix,wire)==base
        native=git('show',item['revision']+':csrc/ops/low_latency.cpp')
        if prefix:
            assert 'low_latency_prefix_ready_revision' in native
            assert 'low_latency_shared_pull_wire_revision' in native


@pytest.mark.parametrize('kind,count',[('fast',3),('series',9)])
def test_plan_uses_same_sdk_full_off_correctness_and_alternating_complete_calls(kind,count):
    actual=plan(kind)
    assert len(actual['phases']['build'])==count
    assert len({identity['dependency_id'] for identity in actual['dependencies'].values()})==1
    runs=actual['phases']['run']
    assert len(runs)==2*count
    assert [item['label'] for item in runs[:count]]==[item['label'] for item in reversed(runs[count:])]
    for item in runs:
        assert item['measured_route_seed']==2
        command=item['command']
        assert not {'--ll-stage-timer','--skip-check','--dispatch-only'} & set(command)
        assert command[command.index('--dtype')+1]=='fp8'
        assert command[command.index('--num-tokens')+1]=='256'
        assert command[command.index('--hidden')+1]=='3584'
        assert command[command.index('--num-topk')+1]=='8'
        assert command[command.index('--num-experts')+1]=='384'


@pytest.mark.parametrize('kind,seeds,count',[
    ('fast',(0,),12),('series',(0,),36),('series',(0,17,101),108)])
def test_report_preserves_every_incremental_row_within_single_file_limit(tmp_path,kind,seeds,count):
    selected=entries(kind)
    measured={(round_id,seed):{
        item['label']:dict(dispatch=85.41371981243567+index*.21973818342321,
                           combine=138.91598241358216+index*.01469123876543)
        for index,item in enumerate(selected)} for round_id in (1,2) for seed in seeds}
    rows=suite.comparisons(selected,measured,'alternating')
    assert len(rows)==count
    for row in rows:
        row.update(result_status='measured',source_session_id='session-20261026-123456-1234567890abcdef',
                   reference_crosses_sessions=False,shmem_revision='a'*40,
                   reference_shmem_revision='a'*40,sdk_identity_source='verified_manifest',
                   sdk_library_matches_reference=True)
    source=tmp_path/'comparison.csv'
    with source.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    destination=tmp_path/'analysis-report.txt'
    size=report.export(source,destination)
    assert size==len(destination.read_bytes())<=65535
    text=destination.read_text(encoding='utf-8')
    assert f'Rows: {count}\n' in text
    csv_text=text[text.index(','.join(report.FIELDS)+'\n'):]
    exported=list(csv.DictReader(io.StringIO(csv_text)))
    assert len(exported)==count
    for expected,actual in zip(rows,exported):
        assert actual=={field:str(expected.get(field,'')) for field in report.FIELDS}
