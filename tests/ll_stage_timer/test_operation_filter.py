"""Operation selection filters every report without weakening selected-rank validation."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('op_fixture',ROOT/'tests/ll_stage_timer/test_segments.py')
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)

@pytest.mark.parametrize('operation',['dispatch','combine'])
def test_cli_operation_filters_statistics_and_trace(tmp_path,operation):
    source=tmp_path/'raw';source.mkdir()
    for op in ('dispatch','combine'):
        saved=dict(raw=fixture.raw(operation=op),operation=op,frequency_hz=1e9,
                   metadata=dict(rank=0,world_size=1 if op==operation else 2,iteration=0,config=dict(B=8),correctness='passed'))
        (source/f'sample_{op}.json').write_text(json.dumps(saved))
    subprocess.run([sys.executable,str(ROOT/'tests/ll_stage_timer/analyze.py'),str(source),'--operation',operation],check=True)
    output=source/f'stage-analysis-{operation}'
    summary=json.loads((output/'summary.json').read_text())
    assert set(summary['configurations'])=={operation}
    assert {row['operation'] for row in summary['per_rank_core_stage']}=={operation}
    trace=json.loads((output/'trace.json').read_text())
    assert {row['cat'] for row in trace['traceEvents'] if row['ph']=='X'}=={operation}
    import csv
    for name in ('metrics.csv','intervals.csv','cores.csv'):
        with (output/name).open() as f: assert {row['operation'] for row in csv.DictReader(f)}=={operation}
    if operation=='combine': assert not (output/'send_phase.csv').exists()
    # The selected operation still fails if its own rank set is incomplete.
    path=source/f'sample_{operation}.json';saved=json.loads(path.read_text());saved['metadata']['world_size']=2
    path.write_text(json.dumps(saved))
    result=subprocess.run([sys.executable,str(ROOT/'tests/ll_stage_timer/analyze.py'),str(source),'--operation',operation,
                           '--output-dir',str(tmp_path/'bad')],capture_output=True,text=True)
    assert result.returncode!=0 and 'Missing ranks' in result.stderr
    assert not (tmp_path/'bad').exists()
