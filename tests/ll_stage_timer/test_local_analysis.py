import json
from pathlib import Path
import pytest
from tests.ll_stage_timer.test_operation_filter import load, sample

HERE=Path(__file__).parent
analyzer=load('local_analysis',HERE/'analyze.py')

def inputs(root, ranks=range(16,32), operations=('dispatch','combine')):
    root.mkdir(parents=True)
    paths=[]
    for op in operations:
        for rank in ranks:
            for iteration in range(2):
                value=sample(op,rank)
                value['metadata'].update(world_size=32,iteration=iteration,sampling={'diagnostic_iters':2})
                path=root/f'sample_{op}_{rank}_{iteration}.json'
                path.write_text(json.dumps(value))
                paths.append(path)
    return paths

def test_second_host_without_rank_zero(tmp_path):
    paths=inputs(tmp_path/'node1')
    report=analyzer.analyze(paths)
    assert report['world_size']==32 and report['analyzed_ranks']==list(range(16,32))
    assert report['analysis_scope']=='local_rank_subset'
    assert report['missing_remote_ranks']==list(range(16))
    output=tmp_path/'analysis'
    analyzer.process_directory(tmp_path/'node1',output,None,0)
    assert 'local_rank_subset' in (output/'bottleneck_summary.md').read_text(encoding='utf-8')
    trace=json.loads((output/'trace.json').read_text())
    assert trace
    with pytest.raises(ValueError,match='Missing ranks'):
        analyzer.analyze(paths,require_all_ranks=True)

def test_local_missing_card_and_iteration_still_rejected(tmp_path):
    paths=inputs(tmp_path/'node1',operations=('dispatch',))
    with pytest.raises(ValueError,match='Missing ranks'):
        analyzer.analyze([p for p in paths if '_23_' not in p.name])
    with pytest.raises(ValueError,match='misaligned'):
        analyzer.analyze(paths[:-1])
    with pytest.raises(ValueError,match='Duplicate'):
        analyzer.analyze(paths+[paths[0]])

def test_missing_local_operation_ranks_rejected(tmp_path):
    paths=inputs(tmp_path/'node1')
    with pytest.raises(ValueError,match='Missing ranks'):
        analyzer.analyze([p for p in paths if 'combine_23_' not in p.name])

def test_complete_global_and_local_latest_without_manifest(tmp_path):
    run=tmp_path/'runs'/'run001'
    paths=inputs(run/'case_00000'/'normal'/'ll_stage_timer',ranks=range(32))
    report=analyzer.analyze(paths,require_all_ranks=True)
    assert report['analysis_scope']=='all_ranks'
    selected, directories=analyzer.discover(tmp_path/'runs',True)
    assert selected==run and directories==[run/'case_00000'/'normal'/'ll_stage_timer']
