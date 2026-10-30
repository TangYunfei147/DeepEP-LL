"""Current MTE shared-receive reference contracts; all fixtures are synthetic."""
import importlib.util
import json
from pathlib import Path
import re
import pytest

ROOT=Path(__file__).resolve().parents[2]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
timer=load('reference_parser_test',ROOT/'deep_ep/utils/ll_reference_profile_parse.py')
analyzer=load('reference_analyzer_test',Path(__file__).with_name('analyze.py'))


def sample(rank=0,iteration=0,cores=56,world=16,frequency=1e9,offset=10**17,send=30,receiver=True):
    raw=[];scale=iteration+1
    for core in range(cores):
        base=offset+rank*10000+core*1000
        points={0:0,1:10,2:12,3:20,4:30,5:40,6:41,7:100,20:170,21:180,57:185};counts={}
        if core<send or core==send:
            points.update({8:101,9:120});counts[43]=int(core in (0,send))
            if counts[43]:points.update({38:105,39:115,40:119})
        if core==send:points.update({10:101,11:105,12:106,13:110,24:102,25:104})
        if receiver:
            points.update({14:121,15:125,28:122,29:124,18:126,19:168,30:127,31:130})
            count=2 if core in (0,cores-1) else 0
            counts.update({37:count,33:2*scale if count else 0,34:4*scale if count else 0,41:2 if count else 0})
            if count:points.update({32:140,35:165,36:166,42:157})
        words=[0]*timer.WORD_COUNT
        for index,value in points.items():words[index]=base+value*scale
        for index,value in counts.items():words[index]=value
        words[22:24]=[send,cores-send];words[50:57]=[rank,core,world,cores,1,timer.SCHEMA,1]
        words[58]=sum(1<<index for index in (*points,*counts))
        words[59:65]=[timer.MAGIC,1,1,0,0 if core<send else 1,3]
        if counts.get(43):
            words[88]=1;words[89]=6
            if core==send:words[66]=1;words[69]=base+110*scale;words[70]=base+114*scale;words[86:88]=[1,1]
        if receiver:
            total=counts[37]
            words[96:109]=[0,0,total,total,total,6 if total else 0,
                           base+181*scale,base+183*scale,0,1,24,int(total>0),0]
        raw.append([words[i:i+8] for i in range(0,timer.WORD_COUNT,8)])
    return dict(format=timer.FORMAT,raw=raw,operation='dispatch',frequency_hz=frequency,
        metadata=dict(rank=rank,world_size=world,iteration=iteration,correctness='passed',run_id='synthetic-only',
            case_id=0,route='normal',seed=7,profile_seed=7,route_sha256='1'*64,
            config=dict(num_tokens=1,hidden=3584,dtype='fp8'),
            sampling=dict(diagnostic_iters=2,sampling_protocol='initial-reduce-exp100-batch-v3',
                exp_prefix_repeats=100,per_iteration_exp=0,per_iteration_all_reduce=False,reference_profile=True),
            comparison=dict(source_revision='test-source',source_dirty=False,installed_source_sha256='2'*64)))


def getword(raw,core,index):return raw[core][index//8][index%8]&((1<<64)-1)
def setword(raw,core,index,value):raw[core][index//8][index%8]=value
def omitword(raw,core,index):
    setword(raw,core,index,0);setword(raw,core,58,getword(raw,core,58)&~(1<<index))


def inputs(directory,ranks=range(16),**kwargs):
    paths=[]
    for rank in ranks:
        for iteration in range(2):
            folder=directory/'dispatch'/f'rank_{rank:03d}';folder.mkdir(parents=True,exist_ok=True)
            path=folder/f'sample_{iteration:04d}.json'
            path.write_text(json.dumps(sample(rank,iteration,**kwargs)),encoding='utf-8');paths.append(path)
    return paths


@pytest.mark.parametrize('send',(1,10,28,30,55))
def test_actual_split_exact_integer_subtraction_and_shared_receive(send):
    report=timer.parse_records(sample(send=send)['raw'])
    assert report['active_aiv']==56 and (report['send_cores'],report['recv_cores'])==(send,56-send)
    assert report['metadata_cores']==0
    kernel=next(r for r in report['records'] if r['core']==0 and r['stage']=='kernel')
    assert kernel['total_ticks']==180 and kernel['total_us']==.18
    for core in (0,55):
        info=report['cores'][core]
        assert info['packet_detail']['kind']=='shared' and info['packet_detail']['receiver']['total_tasks']==2
    assert any(r['core']==0 and r['stage']=='sender' for r in report['records'])
    assert any(r['core']==0 and r['stage']=='receiver' for r in report['records'])


def test_signed_torch_carrier_preserves_uint64_precision():
    raw=sample(offset=(1<<63)+100)['raw']
    for slots in raw:
        for row in slots:row[:]=[value-(1<<64) if value>=1<<63 else value for value in row]
    assert timer.parse_records(raw)['records'][0]['total_ticks']==180


def test_valid_zero_is_not_missing_and_empty_receiver_has_no_fake_buffer_time():
    raw=sample(offset=0)['raw'];setword(raw,0,1,0)
    parsed=timer.parse_records(raw,frequency_hz=None)
    first=next(r for r in parsed['records'] if r['core']==0 and r['stage']=='entry_config')
    assert first['total_ticks']==0 and 'total_us' not in first
    assert not any(r['core']==1 and r['stage']=='send_packets' for r in parsed['records'])
    zero=next(r for r in parsed['records'] if r['core']==1 and r['stage']=='recv_packet_poll')
    assert zero['total_ticks']==0 and zero['calls']==0
    assert not any(r['core']==1 and r['stage']=='recv_buffers' for r in parsed['records'])


@pytest.mark.parametrize('index,value',[(55,5),(55,18),(56,2),(59,0),(60,2),(61,0),(62,4),(63,2),
                                       (22,0),(23,30),(53,64),(50,16),(52,8),(44,1),(64,1)])
def test_current_header_rejects_old_or_mismatched_format(index,value):
    raw=sample()['raw'];setword(raw,0,index,value)
    with pytest.raises(ValueError):timer.parse_records(raw)


@pytest.mark.parametrize('index',sorted(timer.COMMON_REQUIRED))
def test_missing_common_point_is_corrupt(index):
    raw=sample()['raw'];omitword(raw,0,index)
    with pytest.raises(ValueError,match='mandatory'):timer.parse_records(raw)


@pytest.mark.parametrize('index',(8,9,43,18,19,30,31,33,34,37,41,32,35,36,42))
def test_required_executed_branch_point_cannot_disappear(index):
    raw=sample()['raw'];omitword(raw,0,index)
    with pytest.raises(ValueError):timer.parse_records(raw)


@pytest.mark.parametrize('index',(38,39,40,32,35,36,42))
def test_empty_core_cannot_contain_task_only_points(index):
    raw=sample()['raw'];setword(raw,1,index,getword(raw,1,0)+155)
    setword(raw,1,58,getword(raw,1,58)|(1<<index))
    with pytest.raises(ValueError,match='Empty|boundary order'):timer.parse_records(raw)


def test_optional_nonexistent_drain_join_are_blank_not_zero():
    parsed=timer.parse_records(sample()['raw'])
    assert not any(r['stage'] in ('count_drain','cumsum_join') for r in parsed['records'])
    raw=sample()['raw'];omitword(raw,30,24)
    with pytest.raises(ValueError,match='Incomplete'):timer.parse_records(raw)


def test_shape_order_and_frequency():
    with pytest.raises(ValueError):timer.parse_records(sample(cores=64)['raw'])
    with pytest.raises(ValueError,match='shape'):timer.parse_records([c[:7] for c in sample()['raw']])
    with pytest.raises(ValueError,match='dispatch'):timer.parse_records(sample()['raw'],'combine')
    raw=sample()['raw'];setword(raw,0,39,getword(raw,0,0)+104)
    with pytest.raises(ValueError,match='order'):timer.parse_records(raw)
    for value in (0,-1,True,float('nan'),float('inf')):
        with pytest.raises(ValueError,match='frequency'):timer.parse_records(sample()['raw'],frequency_hz=value)


def test_role_boundaries_cannot_move_outside_their_parent():
    raw=sample()['raw'];setword(raw,0,40,getword(raw,0,9)+1)
    with pytest.raises(ValueError,match='boundary order'):timer.parse_records(raw)


def test_cpp_word_names_and_current_schema_agree():
    source=(ROOT/'csrc/kernels/dispatch/low_latency/common/ll_reference_profile_layout.h').read_text(encoding='utf-8')
    block=source.split('kNames[kWordsPerCore] = {',1)[1].split('};',1)[0]
    assert tuple(re.findall(r'"([^"]+)"',block))==timer.WORDS
    assert 'kSchema = 6U' in source


@pytest.mark.parametrize('world',(16,32,64,128,256))
def test_existing_ll_domains(world):
    parsed=timer.parse_records(sample(rank=world-1,world=world)['raw'])
    assert parsed['rank']==world-1 and parsed['world_size']==world


def test_analysis_local_scope_round_stats_and_missing_data(tmp_path):
    paths=inputs(tmp_path/'ll_reference_profile',ranks=range(16,32),world=32)
    report=analyzer.analyze(paths)
    assert report['analysis_scope']=='local_rank_subset' and report['analyzed_ranks']==list(range(16,32))
    assert report['rank_kernel'][0]['mean']==pytest.approx(.27)
    with pytest.raises(ValueError,match='Missing ranks'):analyzer.analyze(paths,require_all_ranks=True)
    with pytest.raises(ValueError,match='Missing ranks'):analyzer.analyze(paths[2:])
    with pytest.raises(ValueError,match='misaligned'):analyzer.analyze(paths[:-1])
    with pytest.raises(ValueError,match='Duplicate'):analyzer.analyze(paths+[paths[0]])


@pytest.mark.parametrize('field',('source','config','route','comparison','sampling','seed','profile_seed','route_sha256'))
def test_analysis_rejects_mixed_identity(tmp_path,field):
    paths=inputs(tmp_path/'ll_reference_profile');saved=json.loads(paths[-1].read_text())
    saved['metadata'][field]=123 if field in ('seed','profile_seed') else {'changed':True}
    paths[-1].write_text(json.dumps(saved),encoding='utf-8')
    with pytest.raises(ValueError):analyzer.analyze(paths)


def test_discovery_and_mixed_core_split(tmp_path):
    left=tmp_path/'case_0/normal/ll_reference_profile';right=tmp_path/'case_1/hotspot/ll_reference_profile'
    paths=inputs(left);inputs(right)
    assert analyzer.discover(tmp_path)==[left,right]
    old=json.loads(paths[-1].read_text());new=sample(rank=old['metadata']['rank'],iteration=1,send=28)
    paths[-1].write_text(json.dumps(new),encoding='utf-8')
    with pytest.raises(ValueError,match='Mixed'):analyzer.analyze(paths)
