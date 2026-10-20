"""Uniform CLI op aliases across cumulative, event and sampled-segment branches."""
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import pytest
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).parent

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def sample(operation,rank):
    if (HERE/'test_segments.py').exists():
        fixture=load('op_segments',HERE/'test_segments.py')
        raw=fixture.raw(operation=operation)
        value=dict(raw=raw,operation=operation,frequency_hz=1e9,metadata={})
    elif (HERE/'test_bottleneck_report.py').exists():
        fixture=load('op_bottleneck',HERE/'test_bottleneck_report.py')
        value=fixture.saved(rank=rank,operation=operation)
        value['metadata']['sampling']=dict(diagnostic_iters=1)
    else:
        fixture=load('op_trace',HERE/'test_trace.py');value=fixture.sample(rank=rank)
        value['operation']=operation
        for core in value['raw']:core[0][2]=1 if operation=='dispatch' else 2
    value['metadata'].update(rank=rank,world_size=16,iteration=0,correctness='passed',config=dict(hidden=3584,num_tokens=8,num_topk=8,dtype='bf16'))
    return value

@pytest.mark.parametrize('flag,operation',[('--op','dispatch'),('--op','combine'),('--operation','dispatch'),('--operation','combine')])
def test_cli_selection_and_rank_validation(tmp_path,flag,operation):
    source=tmp_path/'input with spaces';source.mkdir()
    for rank in range(16):
        (source/f'sample_{operation}_{rank}.json').write_text(json.dumps(sample(operation,rank)))
    other='combine' if operation=='dispatch' else 'dispatch'
    # Unselected incomplete ranks must not contaminate this operation's report.
    (source/f'sample_{other}_0.json').write_text(json.dumps(sample(other,0)))
    command=[sys.executable,'-X','utf8',str(HERE/'analyze.py'),str(source),flag,operation,'--frequency-hz','1000000000']
    subprocess.run(command,cwd=tmp_path,check=True)
    output=source/f'stage-analysis-{operation}'
    summary=json.loads((output/'summary.json').read_text())
    assert set(summary['configurations'])=={operation}
    for path in output.glob('*.csv'):
        with path.open() as stream:
            rows=list(csv.DictReader(stream))
        if rows and 'operation' in rows[0]:assert {r['operation'] for r in rows}=={operation}
    events=json.loads((output/'trace.json').read_text())['traceEvents']
    assert len({r['pid'] for r in events if r['ph']=='X'})==16
    names=[r['args']['name'] for r in events if r.get('name')=='process_name']
    if names:assert all(name.startswith(operation+' ') for name in names)
    else:assert {r['cat'] for r in events if r['ph']=='X'}=={operation}
    # Hide the last selected-rank input; analyzer must still reject missing rank.
    (source/f'sample_{operation}_15.json').rename(source/'excluded.json')
    bad=subprocess.run(command+['--output-dir',str(tmp_path/'bad')],cwd=tmp_path,capture_output=True,text=True)
    assert bad.returncode!=0 and 'Missing ranks' in bad.stderr
    assert not (tmp_path/'bad').exists()

def test_missing_operation_and_invalid_cli_fail_before_output(tmp_path):
    source=tmp_path/'input';source.mkdir()
    (source/'sample_0.json').write_text(json.dumps(sample('dispatch',0)))
    command=[sys.executable,'-X','utf8',str(HERE/'analyze.py'),str(source),'--op']
    missing=subprocess.run(command+['combine'],capture_output=True,text=True)
    assert missing.returncode!=0 and 'No combine diagnostic samples' in missing.stderr
    invalid=subprocess.run(command+['unknown'],capture_output=True,text=True)
    assert invalid.returncode!=0 and 'invalid choice' in invalid.stderr
    assert not (source/'stage-analysis-combine').exists()


def test_default_keeps_both_operations_and_never_overwrites(tmp_path):
    source=tmp_path/'input';source.mkdir()
    for operation in ('dispatch','combine'):
        for rank in range(16):
            (source/f'sample_{operation}_{rank}.json').write_text(json.dumps(sample(operation,rank)))
    command=[sys.executable,'-X','utf8',str(HERE/'analyze.py'),str(source),'--frequency-hz','1000000000']
    subprocess.run(command,cwd=tmp_path,check=True)
    output=source/'stage-analysis'
    before=(output/'summary.json').read_bytes()
    assert set(json.loads(before)['configurations'])=={'dispatch','combine'}
    events=json.loads((output/'trace.json').read_text())['traceEvents']
    assert len({r['pid'] for r in events if r['ph']=='X'})==32
    again=subprocess.run(command,cwd=tmp_path,capture_output=True,text=True)
    assert again.returncode!=0 and 'FileExistsError' in again.stderr
    assert (output/'summary.json').read_bytes()==before
