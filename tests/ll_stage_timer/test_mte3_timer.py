"""CPU validation of schema16 raw layout and the intervention evidence contract."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('mte3_timer_tested', ROOT/'deep_ep/utils/ll_stage_timer_mte3.py')
timer = importlib.util.module_from_spec(spec); spec.loader.exec_module(timer)


def setword(data, core, offset, value):
    data[core][offset//8][offset%8] = value


def raw(mode=1, operation='dispatch', rank=0, iteration=0, minimal=False):
    result = []
    for core in range(56):
        epoch = (rank+1)*1_000_000_000 + core*1_000_000 + iteration*100_000
        points = dict(ENTRY=epoch, INIT_DONE=epoch+1000, WORK_DONE=epoch+100_000)
        role = (3 if core < 30 else 2) if operation == 'dispatch' else (1 if core < 30 else 4)
        words = [0]*64
        words[:8] = [timer.MAGIC,16,1 if operation=='dispatch' else 2,56,core,
                     mode | 30<<8 | role<<16 | 7<<24,0,0]
        if operation=='dispatch' and not minimal:
            # R prepares reception while local source submission is still active.
            prefix = 46000 if core<30 else 5000
            receive = 50000 if core<30 else 12000
            points.update(SEND_PREPARED=epoch+10000,FIRST_SEND=epoch+11000,
                SOURCE_SUBMIT_DONE=epoch+45000,SEND_DONE=epoch+97000,
                PREFIX_WAIT_ENTER=epoch+prefix,PREFIX_WAIT_EXIT=epoch+prefix+1000,
                PREFIX_READY=epoch+prefix+1000,RECV_PREPARED=epoch+receive,
                FIRST_RECV_PROBE=epoch+receive+1000,FIRST_RECV_READY=epoch+receive+2000,
                RECV_PENDING_EMPTY=epoch+90000,RECV_DONE=epoch+95000)
            words[7] = 3 | 3<<8 | 1<<16
            metrics = [[800,100,8,0], [1000,1000,1,0], [2000,200,10,0], [4000,4000,1,0]]+[[0]*4 for _ in range(4)]
            if core<30 and mode==4:
                metrics[1] = [200,200,1,0]
                metrics[4:6] = [[5000,1000,5,50], [1000,1000,1,10]]
            if core<30 and mode==5:
                metrics[3] = [200,200,1,0]
                metrics[6:8] = [[3000,600,7,50], [700,700,1,10]]
            words[32:] = [word for metric in metrics for word in metric]
        for name, tick in points.items():
            index=timer.POINTS.index(name); words[6] |= 1<<index; words[8+index]=tick
        result.append([words[index:index+8] for index in range(0,64,8)])
    return result


@pytest.mark.parametrize('mode',[1,4,5])
def test_modes_preserve_real_milestones_and_separate_wait_statistics(mode):
    parsed=timer.parse_records(raw(mode),'dispatch',1e9)
    assert parsed['schema']==16 and parsed['boundary_mode']==timer.MODES[mode]
    assert parsed['recording']=='mte3-backpressure'
    assert len(parsed['wait_metrics'])==56*8
    assert parsed['cores'][0]['send_lanes']==3
    assert parsed['cores'][0]['recv_lanes']==3
    assert parsed['cores'][0]['overlap_enabled'] is True
    assert not set(timer.METRICS) & {row['stage'] for row in parsed['intervals']}
    assert parsed['wait_metrics'][0]['sum_us']==.8
    assert parsed['wait_metrics'][0]['calls']==8
    assert all(row['empty_control_ticks']==0 for row in parsed['wait_metrics'] if not row['intervention'])


@pytest.mark.parametrize('operation',['dispatch','combine'])
def test_minimal_raw_has_no_fabricated_wait_measurements(operation):
    parsed=timer.parse_records(raw(operation=operation,minimal=True),operation)
    assert {row['stage'] for row in parsed['records']}=={'INIT','KERNEL'}
    assert all(not row['time_measured'] and 'sum_us' not in row for row in parsed['wait_metrics'])
    if operation=='combine':
        assert parsed['wait_metrics']==[]


def test_signed_excess_is_not_clamped_and_never_injected_into_timeline():
    data=raw(4); setword(data,0,32+4*4+3,7000)
    parsed=timer.parse_records(data,'dispatch',1e9)
    row=next(row for row in parsed['wait_metrics'] if row['core']==0 and row['metric']=='TX_REUSE_MTE3')
    assert row['signed_excess_ticks']==-2000 and row['signed_excess_us']==-2
    assert all(row['stage']!='TX_REUSE_MTE3' for row in parsed['intervals'])


@pytest.mark.parametrize('offset,value',[(1,15),(2,2),(3,55),(4,2),(5,0),(6,1<<18),
    (7,1<<17),(7,9),(7,9<<8),(7,(1<<16)|7),(8,-1),(8,True),(8,1<<64),(9,0),(21,0),(26,1),(31,1),
    (32,99),(33,900),(34,0),(35,1),(48,1)])
def test_malformed_words_fail(offset,value):
    data=raw();setword(data,0,offset,value)
    with pytest.raises(ValueError):timer.parse_records(data,'dispatch')


@pytest.mark.parametrize('problem',['shape','mixed_mode','wrong_role','missing_bit','missing_parent',
                                  'dependency_order','outside_work','illegal_R_wait','op_intervention'])
def test_layout_role_and_causal_validation(problem):
    data=raw()
    if problem=='shape':data[0].pop()
    elif problem=='mixed_mode':setword(data,55,5,4 | 30<<8 | 2<<16 | 7<<24)
    elif problem=='wrong_role':setword(data,0,5,1 | 30<<8 | 2<<16 | 7<<24)
    elif problem=='missing_bit':data[0][0][6] &= ~(1<<timer.POINTS.index('FIRST_SEND'))
    elif problem=='missing_parent':
        index=timer.POINTS.index('PREFIX_WAIT_ENTER');data[0][0][6] &= ~(1<<index);setword(data,0,8+index,0)
    elif problem=='dependency_order':setword(data,0,8+timer.POINTS.index('FIRST_SEND'),1_000_001_000)
    elif problem=='outside_work':setword(data,0,8+timer.POINTS.index('SOURCE_SUBMIT_DONE'),1_999_999_999)
    elif problem=='illegal_R_wait':
        data=raw(4)
        for offset,value in enumerate([10,10,1,0]):setword(data,30,48+offset,value)
    elif problem=='op_intervention':data=raw(4,'combine')
    with pytest.raises(ValueError):timer.parse_records(data,'combine' if problem=='op_intervention' else 'dispatch')


@pytest.mark.parametrize('frequency',[True,0,-1,float('nan'),float('inf'),'1e9'])
def test_bad_frequency_is_not_silently_coerced(frequency):
    with pytest.raises(ValueError,match='frequency'):timer.parse_records(raw(),'dispatch',frequency)


def test_explicit_sum_max_calls_consistency_and_control_without_calls():
    for values in ([100,10,1,0],[0,0,0,2],[99,100,2,0]):
        data=raw(4)
        for offset,value in enumerate(values):setword(data,0,48+offset,value)
        with pytest.raises(ValueError,match='sum/max/calls'):timer.parse_records(data,'dispatch')


@pytest.mark.parametrize('problem',['single_wait','disjoint_sum','empty_control'])
def test_disjoint_waits_and_controls_must_fit_same_core_work(problem):
    data=raw(4)
    if problem=='single_wait':
        for offset,value in enumerate([100_000,100_000,1,0]):setword(data,0,44+offset,value)
    elif problem=='disjoint_sum':
        # Each individual sum is below the 99,000-tick work span, but their
        # disjoint total is impossible. No overlapping milestone is included.
        for metric,values in ((0,[48_000,6_000,8,0]),(1,[40_000,40_000,1,0]),(3,[20_000,20_000,1,0])):
            for offset,value in enumerate(values):setword(data,0,32+metric*4+offset,value)
    else:
        setword(data,0,51,100_000)
    with pytest.raises(ValueError,match='Disjoint.*exceed same-core'):
        timer.parse_records(data,'dispatch')


def test_disjoint_total_at_work_boundary_is_valid_without_summing_milestones():
    data=raw()
    for offset,value in enumerate([95_200,95_200,1,0]):setword(data,0,44+offset,value)
    parsed=timer.parse_records(data,'dispatch')
    assert parsed['cores'][0]['measured_wait_ticks']==99_000
    # Several overlapping milestone spans exceed this if incorrectly summed.
    assert sum(row['total_ticks'] for row in parsed['records'] if row['core']==0)>99_000
