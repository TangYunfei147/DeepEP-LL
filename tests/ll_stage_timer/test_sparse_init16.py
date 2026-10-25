"""Schema16 initialization endpoints coexist with exact schema15 decoding."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[2]


def load(name,relative):
    spec=importlib.util.spec_from_file_location(name,ROOT/relative)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


fixture=load('init16_old_fixture','tests/ll_stage_timer/test_sparse_timer.py')
timer=fixture.timer


def raw16(operation='dispatch',minimal=False,send=30):
    raw=fixture.raw(operation=operation,minimal=minimal,send=send)
    for core in range(56):
        raw[core][0][1]=16
        if operation=='dispatch':
            for point,tick in zip(timer.sparse.INIT_POINTS,(120,140,160,180)):
                index=timer.POINTS.index(point)
                raw[core][0][6] |= 1<<index
                fixture.setword(raw,core,8+index,tick+core)
    return raw


def test_schema16_appends_points_and_keeps_schema15_records_exact():
    assert timer.SCHEMA==16 and timer.SLOTS*timer.FIELDS*8==512
    assert len(timer.sparse.POINTS_V15)==14 and len(timer.POINTS)==18
    assert timer.POINTS[:14]==timer.sparse.POINTS_V15
    old=timer.parse_records(fixture.raw(),'dispatch',1e9)
    new=timer.parse_records(raw16(),'dispatch',1e9)
    assert old['schema']==15 and old['points']==list(timer.sparse.POINTS_V15)
    assert new['schema']==16 and new['points']==list(timer.POINTS)
    added={row[0] for row in timer.sparse.INIT_PAIRS}
    assert [row for row in new['records'] if row['stage'] not in added]==old['records']
    assert len(new['milestones'])==len(old['milestones'])+4*56
    assert {row['total_ticks'] for row in new['records'] if row['stage'] in added}=={20}


@pytest.mark.parametrize('point', ['TILING_DONE','PIPE_READY','OBJECT_READY','MAGIC_DONE'])
def test_dispatch16_requires_each_initialization_point(point):
    raw=raw16()
    index=timer.POINTS.index(point)
    raw[0][0][6] &= ~(1<<index)
    fixture.setword(raw,0,8+index,0)
    with pytest.raises(ValueError,match='required sparse initialization'):
        timer.parse_records(raw,'dispatch')


@pytest.mark.parametrize('point,tick', [('TILING_DONE',99),('PIPE_READY',119),('OBJECT_READY',139),
                                      ('MAGIC_DONE',159),('MAGIC_DONE',201)])
def test_dispatch16_checks_time_order_instead_of_appended_point_id(point,tick):
    raw=raw16();fixture.setword(raw,0,8+timer.POINTS.index(point),tick)
    with pytest.raises(ValueError,match='initialization timestamp order'):
        timer.parse_records(raw,'dispatch')


def test_schema15_never_interprets_the_appended_points_and_reserved_padding_stays_checked():
    raw=raw16()
    for core in range(56):raw[core][0][1]=15
    with pytest.raises(ValueError,match='bitmap or reserved'):
        timer.parse_records(raw,'dispatch')
    raw=raw16();fixture.setword(raw,0,26,1)
    with pytest.raises(ValueError,match='bitmap or reserved'):
        timer.parse_records(raw,'dispatch')
    raw=raw16();fixture.setword(raw,0,32,987)
    timer.parse_records(raw,'dispatch')  # Export touches only words 0..31 of each 64-word carrier.


def test_combine16_preserves_optional_initialization_and_does_not_fill_missing_values():
    report=timer.parse_records(raw16('combine',True),'combine')
    assert {row['stage'] for row in report['records']}=={'INIT','KERNEL'}
    assert len(report['milestones'])==56*3
    assert not any(row['point'] in timer.sparse.INIT_POINTS for row in report['milestones'])


def test_mixed_sparse_schemas_are_rejected_in_tensor_trace_and_analysis():
    raw=raw16();raw[20][0][1]=15
    with pytest.raises(ValueError,match='mismatched sparse header'):timer.parse_records(raw,'dispatch')
    saved=[dict(raw=raw,operation='dispatch',frequency_hz=1e9,metadata=fixture.metadata())
           for raw in (fixture.raw(),raw16())]
    with pytest.raises(ValueError,match='Mixed segment schemas'):timer.chrome_trace(saved)
    analyzer=load('init16_analyzer_mixed','tests/ll_stage_timer/analyze.py')
    samples=[(fixture.metadata(iteration=i),timer.parse_records(raw,'dispatch',1e9))
             for i,raw in enumerate((fixture.raw(),raw16()))]
    with pytest.raises(ValueError):analyzer.analyze_samples(samples)


def test_schema16_trace_transfer_analysis_and_initialization_summary(tmp_path):
    decoded=timer.parse_records(raw16(),'dispatch',1e9)
    metadata=fixture.metadata()
    transfer=load('init16_transfer','tests/ll_stage_timer/transfer_report.py')
    _,samples=transfer.decode_samples(transfer.encode_samples(dict(world_size=1),[(metadata,decoded)]))
    assert samples==[(metadata,decoded)]
    trace_module=load('init16_transfer_trace','tests/ll_stage_timer/transfer_trace.py')
    trace=trace_module.trace_from_decoded(samples)
    assert trace['metadata']['all_calls_in_metrics'] is False
    assert any(row['name']=='OBJECT_TO_MAGIC' and row['ph']=='X' for row in trace['traceEvents'])
    analyzer=load('init16_analyzer','tests/ll_stage_timer/analyze.py')
    source=tmp_path/'source';source.mkdir()
    (source/'sample_000.json').write_text(json.dumps(dict(**decoded,raw=raw16(),metadata=metadata)))
    analyzer.process_directory(source,tmp_path/'analysis',1e9,0)
    assert (tmp_path/'analysis/milestones.csv').is_file()
    assert not (tmp_path/'analysis/communication_bottleneck.md').exists()
    metric=load('init16_metric','tests/ll_stage_timer/triage_metrics.py')
    summary=metric.summarize_case({},samples)
    triage=load('init16_triage','tests/ll_stage_timer/triage_report.py')
    cases=[]
    for b in (1,2,4,8,16,24,32):
        row=copy.deepcopy(summary);row['identity']['B']=b
        cases.append(dict(id=f'C{len(cases)+1:03d}',metrics=row,source='analysis/sparse',provenance={}))
    text=triage.render(dict(suite='schema16',cases=cases,matrix={},off=[],performance=[],issues=[],complete=True))
    assert 'DISPATCH INITIALIZATION DETAIL' in text and 'OBJECT->MAGIC' in text
    assert text.count('C001/S |')>=2
    assert len(text.replace('\n','\r\n').encode())<=60000
    # Old data still produces a useful report, without invented initialization costs.
    old_summary=metric.summarize_case({},[(metadata,timer.parse_records(fixture.raw(),'dispatch',1e9))])
    old_text=triage.render(dict(suite='schema15',cases=[dict(id='C001',metrics=old_summary,
        source='analysis/sparse',provenance={})],matrix={},off=[],performance=[],issues=[],complete=True))
    assert 'DISPATCH INITIALIZATION DETAIL' not in old_text
