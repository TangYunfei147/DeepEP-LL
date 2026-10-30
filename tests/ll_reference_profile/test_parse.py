"""Synthetic reference-contract tests; none of these values are measurements."""
import copy
import importlib.util
import json
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


timer = load('mte_reference_parser_test', ROOT/'deep_ep/utils/ll_reference_profile_parse.py')
analyzer = load('mte_reference_analyzer_test', Path(__file__).with_name('analyze.py'))


def word(raw, core, index):
    return raw[core][index//8][index%8]


def setword(raw, core, index, value):
    raw[core][index//8][index%8] = value


def sample(rank=0, iteration=0, *, world=16, send=28, selector=0, combined=False, offset=10**17, traced=True):
    raw=[]
    for core in range(56):
        base=offset+core*1000+rank*100000
        points={0:0,1:10,2:12,3:20,4:30,5:40,6:41,7:50,20:170,21:180,57:190}
        counts={}
        sending=core<send
        receiving=core>=send or (combined and core==0)
        if sending:
            points.update({8:51,9:160})
            counts[43]=2 if core==0 else 0
            if counts[43]:points.update({38:55,39:130,40:159})
        if receiving:
            points.update({10:51,24:52,25:60,11:61,12:62,26:63,27:70,13:72,
                           14:73,28:74,29:75,15:80,16:81,17:90,18:91,30:100,31:110,32:120,35:164,36:166,19:168})
            total=2 if core==55 or combined else 0
            counts.update({37:total,33:2 if total else 0,34:4 if total else 0,41:2 if total else 0})
            if total:points[42]=130
        values=[0]*640
        for index,value in points.items():values[index]=base+value
        for index,value in counts.items():values[index]=value
        values[44:50]=[send,56-send,28 if selector else 56,selector,0 if selector else 3,0]
        values[50:57]=[rank,core,world,56,2,timer.SCHEMA,1]
        values[58]=sum(1<<i for i in (*points,*counts))
        values[59:65]=[timer.MAGIC,1,1,0,4|int(sending)|int(receiving)<<1,2]
        if sending and counts[43]:
            values[65]=2;values[67]=base+60;values[68]=base+125
            values[71:80]=[base+80,base+100,base+110,base+125,3,1,0,((rank+1)%world)*24,(rank+1)%world]
            values[84:86]=[1,2];values[88:90]=[2,1]
        if receiving:
            total=counts[37];rows=total if traced else 0
            values[96:109]=[rows,rows,total,total,total,int(total>0),base+181,base+185,0,2,24,int(rows<total),counts[41]]
            for j in range(rows):
                fields=[j+1,j+1,2-j,(rank+1)%world,j,0,1,((rank+(rank+1)%world)%world)*24*2+j]
                for n in range(4):values[128+j*4+n]=fields[n*2]|fields[n*2+1]<<32
        raw.append([values[i:i+8] for i in range(0,640,8)])
    return dict(format=timer.FORMAT,operation='dispatch',frequency_hz=1e9,raw=raw,
        metadata=dict(rank=rank,world_size=world,iteration=iteration,correctness='passed',run_id='synthetic-only',
            case_id=0,route='normal',seed=0,profile_seed=0,route_sha256='1'*64,
            config=dict(B=2,H=3584,dtype='fp8'),sampling=dict(diagnostic_iters=1),
            comparison=dict(installed_source_sha256='2'*64)))


def inputs(path, ranks=range(16), **kwargs):
    paths=[]
    for rank in ranks:
        dest=path/'ll_reference_profile/dispatch'/f'rank_{rank:03d}'/'sample_0000.json'
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(json.dumps(sample(rank,**kwargs)),encoding='utf-8')
        paths.append(dest)
    return paths


@pytest.mark.parametrize('selector', range(5))
def test_29_stages_large_clock_exactness_and_packet_identity(selector):
    saved=sample(selector=selector)
    parsed=timer.parse_records(saved['raw'])
    assert len(timer.STAGES)==29 and parsed['schema']==7 and parsed['metadata_cores']==0
    assert parsed['dispatch_implementation_id']==selector
    kernel=next(row for row in parsed['records'] if row['core']==0 and row['stage']=='kernel')
    assert kernel['total_ticks']==180 and kernel['total_us']==.18
    detail=parsed['cores'][0]['packet_detail']['sender']
    assert detail['ticks']['second_prepare_without_member_q']==17
    received=parsed['cores'][55]['packet_detail']['receiver']
    assert received['recorded_tasks']==2 and [e['process_order'] for e in received['entries']]==[2,1]


@pytest.mark.parametrize('send', [1,20,28,30,55])
def test_actual_split_and_combined_send_receive_are_preserved(send):
    parsed=timer.parse_records(sample(send=send,combined=True)['raw'])
    assert parsed['send_cores']==send and parsed['recv_cores']==56-send
    assert parsed['cores'][0]['role']=='send+receive'
    assert parsed['cores'][0]['packet_detail']['sender'] and parsed['cores'][0]['packet_detail']['receiver']


def test_measured_zero_is_valid_and_absent_stays_absent():
    saved=sample(offset=0);raw=saved['raw']
    setword(raw,0,1,0)
    parsed=timer.parse_records(raw,frequency_hz=None)
    entry=next(r for r in parsed['records'] if r['core']==0 and r['stage']=='entry_config')
    assert entry['total_ticks']==0 and 'total_us' not in entry
    assert not any(r['stage']=='send_packets' and r['core']==1 for r in parsed['records'])
    zero=next(r for r in parsed['records'] if r['stage']=='recv_packet_poll' and r['core']==28)
    assert zero['total_ticks']==0 and zero['calls']==0


@pytest.mark.parametrize('index,value',[(55,5),(56,2),(59,0),(60,2),(61,0),(62,4),(53,72),(50,16),(52,8),
                                     (44,0),(45,25),(46,28),(47,8),(48,9),(49,100),(63,0),(64,1),(80,1),(90,1),(109,1)])
def test_header_role_or_backend_corruption_rejected(index,value):
    raw=sample()['raw'];setword(raw,0,index,value)
    with pytest.raises(ValueError):timer.parse_records(raw)


@pytest.mark.parametrize('damage',['missing_end','unmarked','outside','backwards','sum','shape','precision','duplicate_order','slot','coverage'])
def test_incomplete_or_fabricated_records_are_rejected(damage):
    raw=sample()['raw']
    if damage=='missing_end':setword(raw,0,58,word(raw,0,58)&~(1<<9));setword(raw,0,9,0)
    elif damage=='unmarked':setword(raw,1,38,123)
    elif damage=='outside':setword(raw,0,1,word(raw,0,0)-1)
    elif damage=='backwards':setword(raw,0,39,word(raw,0,38)-1)
    elif damage=='sum':setword(raw,55,33,181)
    elif damage=='shape':raw[0].pop()
    elif damage=='precision':setword(raw,0,0,float(word(raw,0,0)))
    elif damage=='duplicate_order':setword(raw,55,132,word(raw,55,128))
    elif damage=='slot':setword(raw,55,131,1|999999<<32)
    elif damage=='coverage':setword(raw,55,100,1)
    with pytest.raises(ValueError):timer.parse_records(raw)


@pytest.mark.parametrize('frequency',[0,-1,float('nan'),float('inf'),True])
def test_frequency_must_be_real_positive(frequency):
    with pytest.raises(ValueError):timer.parse_records(sample()['raw'],frequency_hz=frequency)


def test_no_combine_or_legacy_record_adapter():
    with pytest.raises(ValueError,match='dispatch'):timer.parse_records(sample()['raw'],'combine')
    with pytest.raises(ValueError):timer.parse_records(sample()['raw'][:16])


def test_layout_header_matches_every_python_word():
    header=(ROOT/'csrc/kernels/dispatch/low_latency/common/ll_reference_profile_layout.h').read_text()
    body=header.split('kNames[kWordsPerCore] = {',1)[1].split('};',1)[0]
    assert tuple(re.findall(r'"([^"]*)"',body))==timer.WORDS


def test_missing_rank_and_changed_build_identity_stop_analysis(tmp_path):
    paths=inputs(tmp_path,ranks=range(15))
    with pytest.raises(ValueError,match='Missing ranks'):analyzer.analyze(paths)
    paths+=inputs(tmp_path,ranks=[15])
    assert analyzer.analyze(paths)['analyzed_ranks']==list(range(16))
    changed=sample(15,send=30)
    paths[-1].write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='Mixed'):analyzer.analyze(paths)


def test_export_preserves_large_integer_cycles_and_combined_packet_tables(tmp_path):
    from zipfile import ZipFile
    import xml.etree.ElementTree as ET
    source=tmp_path/'source';inputs(source,combined=True,traced=False)
    out=tmp_path/'export'
    report=analyzer.process_directory(source,out)
    assert report['packet_trace_coverage']['truncated_cores']>0
    workbook=out/'core_stage_timings.xlsx'
    with ZipFile(workbook) as archive:
        assert len(ET.fromstring(archive.read('xl/workbook.xml')).findall('{*}sheets/{*}sheet'))==8
        for name in archive.namelist():
            if name.endswith('.xml'):ET.fromstring(archive.read(name))
        raw=archive.read('xl/worksheets/sheet4.xml').decode()
        assert '100000000000000000' in raw and 'inlineStr' in raw
        values=archive.read('xl/worksheets/sheet2.xml').decode()
        assert '发送+接收' in values
        assert "'原始Cycle'!BO" in values
    assert (out/'return_report.md').stat().st_size<=60000
    with pytest.raises(FileExistsError):analyzer.process_directory(source,out)
