"""Schema13 stage separation, old capture compatibility and business-token parity."""
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('stage13_fixture',Path(__file__).with_name('test_segments.py'))
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
timer=f.timer

def strip_probes(source):
    source=re.sub(r'/\*.*?\*/|//[^\n]*','',source,flags=re.S)
    source=re.sub(r'LL_SEG_(?:BEGIN|END|SCOPE|COUNT)\([^;]*?\);','',source)
    return re.findall(r'\w+|[^\s]',source)

@pytest.mark.parametrize('rel',[
 'csrc/kernels/dispatch/low_latency/dispatch_server_dedup_kernel.h',
 'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_counts.h',
 'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h',
 'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_recv.h',
 'csrc/kernels/combine/low_latency/combine_server_dedup_kernel.h',
 'csrc/kernels/combine/low_latency/combine_server_dedup_partial.h',
 'csrc/kernels/combine/low_latency/combine_server_dedup_source_aggregate.h'])
def test_only_probe_labels_change(rel):
    old=subprocess.run(['git','-C',str(ROOT),'show','9e24a79:'+rel],text=True,encoding='utf-8',capture_output=True,check=True).stdout
    assert strip_probes(old)==strip_probes((ROOT/rel).read_text(encoding='utf-8'))

def capture(stage_id,schema):
    raw=f.raw(schema=schema)
    words=[v for row in raw[0] for v in row]
    # Re-label existing RECV_POLL intervals and metric to exercise arbitrary new ID.
    for i in range(words[6]):
        pos=8+i*2;event=words[pos]&0xffffffff;payload=words[pos]>>32
        if event in (29*2,29*2+1):words[pos]=stage_id*2+(event&1)
        if event==0x8000+29:words[pos]=(payload<<32)|(0x8000+stage_id)
    raw[0]=[words[i:i+8] for i in range(0,len(words),8)]
    return raw

def test_every_new_stage_roundtrips_without_old_name_alias():
    for stage in range(81,len(timer.STAGES)):
        parsed=timer.parse_records(capture(stage,13),'dispatch',1e9)
        metric=next(r for r in parsed['records'] if r['core']==0 and r['stage']==timer.STAGES[stage])
        assert metric['calls']==3 and metric['total_ticks']==21
        assert len([r for r in parsed['intervals'] if r['core']==0 and r['stage']==timer.STAGES[stage]])==2

def test_schema12_retains_old_shape_and_meaning():
    raw=capture(3,12)
    assert len(raw[0])==72
    parsed=timer.parse_records(raw,'dispatch',1e9)
    assert any(r['stage']=='PREPARE_INPUT' for r in parsed['records'])
    assert not any(r['stage']=='SEND_SETUP' for r in parsed['records'])
    with pytest.raises(ValueError):
        timer.parse_records(capture(81,12),'dispatch')

def test_ambiguous_prepare_input_is_no_longer_emitted():
    for folder in ('dispatch/low_latency','combine/low_latency'):
        for p in (ROOT/'csrc/kernels'/folder).rglob('*.h'):
            source=p.read_text(encoding='utf-8')
            assert not re.search(r'LL_SEG_(?:BEGIN|END)\(\w+,\s*PREPARE_INPUT\)|LL_SEG_SCOPE\(PREPARE_INPUT\)',source)
    assert timer.SCHEMA==13 and (timer.SLOTS,timer.FIELDS)==(88,8)
    assert (8+2*timer.RECORDS)*8<=timer.SLOTS*timer.FIELDS*8
    assert json.loads((ROOT/'tests/ll_stage_timer/dc_seg_stages.json').read_text())==list(timer.STAGES)

def test_schema13_cannot_be_compared_as_schema12_diagnostic():
    parity=f.load('stage13_parity','tests/ll_stage_timer/test_parity.py')
    left=parity.identity();left.update(core_protocol=19028,stage_timer_schema=12,stage_timer=True)
    right=dict(left,stage_timer_schema=13)
    with pytest.raises(ValueError,match='different stage schemas'):
        parity.comparison.check_identity(left,right)

def test_reused_names_are_only_identical_operations():
    from collections import Counter
    names=Counter()
    for folder in ('dispatch/low_latency','combine/low_latency'):
        for p in (ROOT/'csrc/kernels'/folder).rglob('*.h'):
            if 'common' in p.parts:continue
            for line in p.read_text(encoding='utf-8').splitlines():
                m=re.search(r'LL_SEG_(?:SCOPE\((\w+)|BEGIN\(\w+,\s*(\w+))',line)
                if m:names[m[1] or m[2]]+=1
    assert {n for n,c in names.items() if c>1}<={'SEND_TOKENS','SEND_SUBMIT_END','SEND_COMPLETE_WAIT'}
