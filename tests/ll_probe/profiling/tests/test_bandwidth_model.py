"""Payload, topology bounds and percentile math; no devices or measurement claims."""
import copy
import json
from pathlib import Path
import pytest
from tests.ll_probe.profiling.profile_probe import parser, plan
from tests.ll_probe.profiling.profile_common import describe
from tests.ll_probe.profiling.bandwidth_model import load_hardware, volume, theoretical_time, bandwidth_statistics
from tests.ll_probe.profiling.analyze_profile import analyze, write_table
from tests.ll_probe.profiling.tests.test_deployment_reports import reports

ROOT=Path(__file__).resolve().parents[4]
HARDWARE=ROOT/'tests/ll_probe/configs/hardware/ascend950.yaml'


def shape_args(extra):
    return parser().parse_args(['--kind','comm','--endpoint','tcp://127.0.0.1:29501',
                               '--output-dir','results/shape-plan',*extra])


def test_shape_and_legacy_bytes():
    env=dict(WORLD_SIZE='32',RANK='0',LOCAL_RANK='0')
    args=shape_args(['--bs','32','--hidden-size','7168','--element-bytes','2'])
    result=plan(args,env)
    assert args.bytes==458752 and result['payload']['bytes_per_edge']==458752
    assert result['argv'][5]=='458752'
    assert plan(args,env)['payload']==result['payload']  # idempotent resolution
    args=shape_args([]);plan(args,env);assert args.bytes==15392


@pytest.mark.parametrize('extra',[
    ['--bs','32'],['--hidden-size','7168'],['--bs','0','--hidden-size','32'],
    ['--bs','1','--hidden-size','7'],['--bs','1','--hidden-size','32','--bytes','32']])
def test_bad_shape_rejected(extra):
    with pytest.raises(ValueError):plan(shape_args(extra),dict(WORLD_SIZE='2'))


def test_whitepaper_units_and_fanout_bound():
    hw=load_hardware(HARDWARE)
    assert hw['specification']['whitepaper_reference']['unified_bus_chip_bidirectional_GBps']==2016
    assert hw['specification']['model']['per_rank_tx_GBps']==1008
    cfg=dict(bytes=458752,scenario=3,fanout=3,bs=32,hidden_size=7168,element_bytes=2)
    vol,edges=volume(32,cfg,16)
    assert edges==[(0,1),(0,2),(0,3)]
    assert vol['job_sent_bytes']==1376256 and vol['inter_host_sent_bytes']==0
    bound=theoretical_time(vol,edges,16,hw)
    assert bound['time_us']==pytest.approx(1.3653333333333333)
    assert bound['equivalent_job_peak_GBps']==pytest.approx(1008)


def test_parallel_senders_not_divided_by_one_chip_peak():
    vol,edges=volume(4,dict(bytes=1000,scenario=4),4)
    hw=dict(specification=dict(model=dict(per_rank_tx_GBps=1,per_rank_rx_GBps=1)))
    theory=theoretical_time(vol,edges,4,hw)
    assert vol['job_sent_bytes']==12000
    assert theory['time_us']==3
    assert theory['equivalent_job_peak_GBps']==4


def test_cross_host_shared_bottleneck_and_unknown_topology():
    vol,edges=volume(32,dict(bytes=1000,scenario=3,fanout=17),16)
    hw=dict(specification=dict(model=dict(per_rank_tx_GBps=1008,per_rank_rx_GBps=1008,
                  inter_host_pair_GBps=1,per_host_cross_tx_GBps=.5)))
    assert vol['intra_host_sent_bytes']==15000 and vol['inter_host_sent_bytes']==2000
    assert theoretical_time(vol,edges,16,hw)['time_us']==4
    with pytest.raises(ValueError):theoretical_time(vol,edges,None,hw)


def test_quantiles_and_unclamped_reference_ratio():
    d=describe([1,2,3,4,5])
    assert (d['p50_us'],d['p90_us'],d['p99_us'])==pytest.approx((3,4.6,4.96))
    stats=bandwidth_statistics([1,2,3,4,5],dict(job_sent_bytes=1000),dict(time_us=2))
    assert stats['reference_utilization_percent']['max']==200
    assert stats['exceeds_reference']
    assert stats['bandwidth_GBps']['p90'] != pytest.approx(1/d['p90_us'])
    assert stats['at_latency_percentiles'][1]['effective_GBps']==pytest.approx(1/4.6)


def test_summary_records_volume_samples_and_csv(tmp_path):
    paths=reports(tmp_path,32)
    for path in paths:
        obj=json.loads(path.read_text());obj['configuration'].update(scenario=3,fanout=3)
        obj['hardware_profile']=load_hardware(HARDWARE)
        path.write_text(json.dumps(obj))
    summary=analyze(paths)
    bw=summary['bandwidth_summary']
    assert bw['volume']['job_sent_bytes']==96
    assert bw['captured_sent_bytes']==192 and bw['selected_window_sent_bytes']==96
    table=tmp_path/'summary.csv';write_table(summary,table)
    assert 'P90' in table.read_text() and 'theoretical_time_us' in table.read_text()
    obj=json.loads(paths[-1].read_text());obj['hardware_profile']['specification']['model']['per_rank_tx_GBps']=1
    paths[-1].write_text(json.dumps(obj))
    with pytest.raises(ValueError):analyze(paths)


def test_invalid_hardware_capacity(tmp_path):
    f=tmp_path/'bad.yaml';f.write_text('schema_version: 1\nmodel:\n  per_rank_tx_GBps: -1\n')
    with pytest.raises(ValueError):load_hardware(f)
