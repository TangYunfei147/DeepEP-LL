"""Packed schema 2 keeps 2048B/core and preserves detail/legacy trace semantics."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('detail_timer', ROOT/'deep_ep/utils/ll_stage_timer.py')
timer = importlib.util.module_from_spec(spec); spec.loader.exec_module(timer)


def packed():
    result = []
    for core in range(16):
        words = [0]*256
        words[:8] = [timer.MAGIC, 2, 1, 16, core, 1, 1, len(timer.STAGES)]
        for stage in range(1, len(timer.STAGES)):
            offset = 8+(stage-1)*4
            words[offset:offset+4] = [100, 200, 100 if stage==1 else 30, 1 if stage==1 else 3]
        result.append([words[i:i+8] for i in range(0, 256, 8)])
    return result


def test_every_detail_fits_original_buffer_and_exports():
    raw=packed()
    report=timer.parse_records(raw, 'dispatch', 1e9)
    assert report['schema']==2
    assert len(raw[0])*len(raw[0][0])*8==2048
    assert len(report['records'])==16*(len(timer.STAGES)-1)
    assert {r['stage'] for r in report['records']}==set(timer.STAGES[1:])
    saved=dict(raw=raw, operation='dispatch', frequency_hz=1e9)
    trace=timer.chrome_trace([saved], lane_layout='stage')
    lanes=[e for e in trace['traceEvents'] if e['name']=='thread_name']
    assert len({(e['pid'],e['tid']) for e in lanes})==len(lanes)
    restore=next(e for e in trace['traceEvents'] if e['name']=='send_rdma_restore [envelope x3]')
    assert restore['args']['total_us']==pytest.approx(.03)


@pytest.mark.parametrize('damage', ['active', 'unused', 'header', 'negative', 'total', 'empty', 'mixed'])
def test_packed_corruption_rejected(damage):
    raw=packed()
    if damage=='active': raw[0][1][3] |= timer.ACTIVE
    if damage=='unused': raw[0][-1][-1]=1
    if damage=='header': raw[0][0][7]-=1
    if damage=='negative': raw[0][1][1]=-1
    if damage=='total': raw[0][1][2]=101
    if damage=='empty': raw[0][1][3]=0
    if damage=='mixed': raw[0][0][1]=1
    with pytest.raises(ValueError): timer.parse_records(raw, 'dispatch')
