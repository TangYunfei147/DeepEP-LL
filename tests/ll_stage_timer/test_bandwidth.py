"""Bandwidth is calculated from event-derived kernel spans and independent byte counts."""
import importlib.util
from pathlib import Path
import pytest
from event_fixtures import raw,set_events,timer

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('event_bandwidth',Path(__file__).with_name('analyze.py'))
analyzer=importlib.util.module_from_spec(spec);spec.loader.exec_module(analyzer)


def sample(iteration=0,frequency=1e9,flags=0,world=16):
    data=raw()
    for core,entry in enumerate(data['cores']):
        set_events(data,core,'kernel',[[core*10000,core*10000+1000*(iteration+1)+core]])
        entry['traffic'][1]=[1000,2,250,1]
        entry['traffic'][2]=[100,1,0,0]
        entry['flags']=flags
    report=timer.parse_records(data,'dispatch',frequency)
    return analyzer.bandwidth_sample(report,dict(rank=0,world_size=world,iteration=iteration))


def test_five_round_division_precedes_statistics():
    rows=[row for iteration in range(5) for row in sample(iteration)]
    summary=analyzer.bandwidth_summary(rows)
    total=next(row for row in summary if row['category']=='intrahost_total')
    assert total['bytes']['mean']==17600 and total['read_bytes']['mean']==4000
    assert total['effective_GBps']['mean']==pytest.approx(sum(17600/(1000*(i+1)+15) for i in range(5))/5)


@pytest.mark.parametrize('kwargs,status', [({'frequency':None},'frequency_unavailable'),
    ({'flags':1},'nonstandard_skip_communication'),({'world':32},'unsupported_ep')])
def test_unsupported_rates_are_empty(kwargs,status):
    total=next(row for row in sample(**kwargs) if row['category']=='intrahost_total')
    assert total['status']==status and total['effective_GBps'] is None
