"""Independent input waits, legacy handling and conservative bottleneck decisions."""
import importlib.util
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[2]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

analysis=load('wait_analysis','tests/ll_stage_timer/analyze.py')
fixtures=load('wait_segments','tests/ll_stage_timer/test_segments.py')
timer=analysis.timer


def sample(wait=250,schema=9,stage='TOKEN_INPUT_WAIT',missing=False,calls=10):
    def row(name,ticks,n=10):
        return dict(core=0,stage=name,total_ticks=ticks,calls=n)
    rows=[row('KERNEL',1000,1),row('SEND_TOKENS',500,1),row('TOKEN_INPUT',30)]
    if stage=='TOKEN_QUANT_WAIT':rows.append(row('TOKEN_QUANT',20))
    if not missing:rows.append(row(stage,wait,calls))
    return dict(operation='dispatch',schema=schema,records=rows)


@pytest.mark.parametrize('stage',['TOKEN_INPUT_WAIT','TOKEN_QUANT_WAIT'])
@pytest.mark.parametrize('wait,expected',[(250,'疑似瓶颈'),(5,'未见明显证据')])
def test_decision_uses_sample_ratios(stage,wait,expected):
    result=analysis.input_wait_report([(dict(rank=0,iteration=i),sample(wait,stage=stage)) for i in range(3)])
    row=result['per_core'][0]
    assert row['status']==expected
    assert row['median_send_share']==wait/500
    assert row['median_kernel_share']==wait/1000
    assert row['weighted_ticks_per_call']==wait/10
    assert row['critical_core_fraction']==1


@pytest.mark.parametrize('kwargs',[dict(schema=7),dict(missing=True),dict(calls=9),dict(wait=501)])
def test_missing_old_or_inconsistent_never_becomes_zero(kwargs):
    result=analysis.input_wait_report([(dict(rank=0,iteration=i),sample(**kwargs)) for i in range(3)])
    row=result['per_core'][0]
    assert row['status']=='数据不足' and row['is_bottleneck_candidate'] is None
    assert row['weighted_ticks_per_call'] is None


def test_few_samples_and_one_missing_sample_are_insufficient():
    data=[(dict(rank=0,iteration=i),sample()) for i in range(2)]
    assert analysis.input_wait_report(data)['per_core'][0]['status']=='数据不足'
    data.append((dict(rank=0,iteration=2),sample(missing=True)))
    assert analysis.input_wait_report(data)['per_core'][0]['status']=='数据不足'


def test_separate_rank_and_core_and_no_combined_ratio():
    data=[]
    for i in range(3):
        data.extend([(dict(rank=0,iteration=i),sample(250)),(dict(rank=1,iteration=i),sample(5))])
    rows=analysis.input_wait_report(data)['per_core']
    assert [r['status'] for r in rows]==['疑似瓶颈','未见明显证据']
    assert analysis.input_wait_report([(dict(rank=0,iteration=0),dict(operation='combine'))])['per_core']==[]


def test_extended_parser_capacity_and_stage_lane_ids():
    raw=fixtures.raw(full=True,schema=9)
    decoded=timer.parse_records(raw,'dispatch')
    assert decoded['schema']==9 and len(decoded['records'])==56*66
    assert {'TOKEN_INPUT_WAIT','TOKEN_QUANT_WAIT'}<=set(row['stage'] for row in decoded['records'])
    with pytest.raises(ValueError,match='schema7'):timer.reference_binary(raw,'dispatch')
    old=timer.parse_records(fixtures.raw(full=True),'dispatch')
    assert old['schema']==7 and len(old['records'])==56*64


def test_analysis_writes_wait_report_and_old_samples_are_insufficient(tmp_path):
    source=tmp_path/'source';source.mkdir()
    for i in range(3):
        saved=dict(raw=fixtures.raw(full=True,schema=9),operation='dispatch',frequency_hz=1e9,
            metadata=dict(world_size=1,rank=0,iteration=i,config={},correctness='passed'))
        (source/f'sample_{i}.json').write_text(json.dumps(saved))
    out=tmp_path/'out'
    analysis.process_directory(source,out,1e9,0)
    assert (out/'input_wait_bottleneck.csv').is_file()
    assert '启发式' in (out/'input_wait_bottleneck.md').read_text(encoding='utf-8')
    assert 'input_wait_bottleneck' in json.loads((out/'summary.json').read_text())


def test_existing_sync_is_wrapped_not_added():
    source=(ROOT/'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h').read_text()
    assert 'LL_SEG_BEGIN(quantWaitBegin, TOKEN_QUANT_WAIT);\n            Sync<HardEvent::V_S>();\n            LL_SEG_END' in source
    assert 'LL_SEG_BEGIN(inputWaitBegin, TOKEN_INPUT_WAIT);\n            Sync<HardEvent::MTE2_S>();\n            LL_SEG_END' in source
    # The existing test_segment_source fingerprint checks *all* non-probe
    # tokens, including event/barrier order, against the prior algorithm.


def test_schema_change_rejects_diagnostic_ratio_but_allows_off_perf():
    parity=load('wait_parity','tests/ll_stage_timer/test_parity.py')
    left=parity.identity()
    left.update(core_protocol=10010,send_cores=10,recv_cores=46,stage_timer_schema=7,
                dispatch_recv_cores=56)
    right=dict(left,stage_timer_schema=9)
    with pytest.raises(ValueError,match='diagnostic semantics'):
        parity.comparison.check_identity(left,right)
    for item in (left,right):
        item.update(stage_timer=False,stage_capture=0,expert_capture=False)
    parity.comparison.check_identity(left,right)
