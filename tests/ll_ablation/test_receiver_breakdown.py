import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
import pytest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from common import ROOT,sha
from receiver_report import CASES,ERRORS,decode_sample

def sample():
    rows=[]
    for core in range(56):
        for name,(duration,flags,output) in CASES.items():
            f=0 if name=='skip-flags' else 3;b=0 if name=='no-payload-output' else 37*3696
            rows.extend([dict(core=core,stage=duration,calls=1,total_ticks=40000),
                         dict(core=core,stage=flags,calls=f+1,total_ticks=0),
                         dict(core=core,stage=output,calls=b+1,total_ticks=0),
                         dict(core=core,stage=ERRORS[name],calls=1,total_ticks=0)])
        rows.append(dict(core=core,stage='REPLAY_ORDER_PLUS1',calls=2,total_ticks=0))
    return dict(schema=15,records=rows)

def test_operation_counters_validate_actual_suppression():
    data=sample();assert len(decode_sample(data))==56
    next(r for r in data['records'] if r['stage']=='REPLAY_SKIP_FLAGS_PLUS1')['calls']=2
    with pytest.raises(ValueError,match='still issued flag'):decode_sample(data)
    data=sample()
    next(r for r in data['records'] if r['stage']=='REPLAY_NOOUT_BYTES_PLUS1')['calls']=2
    with pytest.raises(ValueError,match='still issued payload'):decode_sample(data)
    data=sample()
    next(r for r in data['records'] if r['stage']=='REPLAY_SKIP_ERRORS_PLUS1')['calls']=2
    with pytest.raises(ValueError,match='verification failed'):decode_sample(data)

def test_schema15_raw_record_roundtrip():
    spec=importlib.util.spec_from_file_location('receiver_schema',ROOT/'deep_ep/utils/ll_stage_timer.py')
    timer=importlib.util.module_from_spec(spec);spec.loader.exec_module(timer)
    raw=[];data=sample()
    for core in range(56):
        role=3 if core<30 else 2
        pairs=[(0,100),(1,200),((1<<32)|0x8000,100)]
        for r in data['records']:
            if r['core']==core:pairs.append(((r['calls']<<32)|(0x8000+timer.STAGES.index(r['stage'])),r['total_ticks']))
        words=[timer.MAGIC,15,1,56,core,1|(30<<8)|(role<<16),len(pairs),3]
        for event,tick in pairs:words.extend([event,tick])
        words.extend([0]*(640-len(words)));raw.append([words[i:i+8] for i in range(0,640,8)])
    assert len(decode_sample(timer.parse_records(raw,'dispatch',1e9)))==56

@pytest.mark.parametrize('size',[0,1,15,16,17,37,586])
def test_saved_layout_matches_full_flag_decode_and_priority(size):
    rng=random.Random(size)
    positions=rng.sample(range(384*256),size)
    keep=[bool(rng.getrandbits(1)) for _ in positions]
    encoded=[p|(int(k)<<31) for p,k in zip(positions,keep)]
    assert [p&0x7fffffff for p in encoded]==positions
    pending=list(range(size));consumed=[]
    while pending:
        batch=pending[:16]
        original=[];cached=[]
        for j in batch:
            flags=[1.0]*64;flags[0]=1.5 if keep[j] else 1.0
            original.append(sum(flags)==64.5)
            cached.append(bool(encoded[j]>>31))
        order=lambda values:[i for cls in (True,False) for i in reversed(range(len(values))) if values[i]==cls]
        assert original==cached and order(original)==order(cached)
        consumed.extend(batch[i] for i in order(cached))
        for i in reversed(range(len(batch))):pending[i]=pending[-1];pending.pop()
    assert sorted(consumed)==list(range(size))

def test_cache_advance_preserves_objects_and_noop_header_timestamp(tmp_path):
    from cache_workspace import advance,inspect,HEADER,acquire
    def git(root,*args):
        return subprocess.check_output(['git','-C',str(root),'-c','user.name=codex_remote','-c',
            'user.email=799847028@qq.com',*args],text=True,stderr=subprocess.STDOUT).strip()
    root=tmp_path/'project with spaces';root.mkdir();git(root,'init')
    (root/'.gitignore').write_text('.ablation/\n.tmp/\nbuild/\nthird-party/shmem/install/\n')
    header=root/HEADER;header.parent.mkdir(parents=True);header.write_text('constexpr unsigned kLlAblationMode = 0;\n')
    kernel=root/'csrc/kernel.cpp';kernel.write_text('old\n')
    git(root,'add','.');git(root,'commit','-m','base');old=git(root,'rev-parse','HEAD')
    source=root/'.ablation/sources/ready-replay-ON-s30';source.parent.mkdir(parents=True)
    git(root,'worktree','add','--detach',str(source),old)
    (source/HEADER).write_text('constexpr unsigned kLlAblationMode = 4;\n')
    sdk=source/'third-party/shmem/install/shmem/deepep-sdk.json';sdk.parent.mkdir(parents=True);sdk.write_text('{}')
    obj=source/'build/device.o';obj.parent.mkdir();obj.write_bytes(b'cached object')
    out=root/'.ablation/wheels/ready-replay-ON-s30';out.mkdir(parents=True)
    wheel=out/'old.whl';wheel.write_bytes(b'old wheel')
    info=dict(mode='ready-replay',timer='ON',send_cores=30,recv_cores=26,commit=old,
              source=str(source.relative_to(root)),wheel=wheel.name,wheel_sha256=sha(wheel),header_sha256=sha(source/HEADER))
    manifest=out/'manifest.json';manifest.write_text(json.dumps(info))
    kernel.write_text('new\n');git(root,'add','.');git(root,'commit','-m','new');new=git(root,'rev-parse','HEAD')
    assert advance(root,manifest,new,30,5)==source
    assert obj.read_bytes()==b'cached object' and wheel.read_bytes()==b'old wheel'
    stamp=(source/HEADER).stat().st_mtime_ns
    advance(root,manifest,new,30,5)
    assert (source/HEADER).stat().st_mtime_ns==stamp
    release=acquire(source)
    with pytest.raises(RuntimeError,match='locked'):acquire(source)
    release()
    (source/'csrc/kernel.cpp').write_text('user edit\n')
    with pytest.raises(RuntimeError,match='user changes'):inspect(root,manifest,new,30)
    assert (source/'csrc/kernel.cpp').read_text()=='user edit\n'

def test_one_command_builds_one_wheel_then_runs_and_reports(tmp_path,monkeypatch):
    import argparse
    import receiver_suite as suite
    monkeypatch.setattr(suite,'ROOT',tmp_path)
    monkeypatch.setattr(suite,'git',lambda *args:'current-commit')
    config=tmp_path/'case.json';config.write_text((ROOT/'tests/ll_ablation/b256.json').read_text())
    donor=tmp_path/'.ablation/wheels/ready-replay-ON-s30/manifest.json'
    donor.parent.mkdir(parents=True);donor.write_text('{}')
    a=argparse.Namespace(config=config,send_cores=30,fresh_build=False,offline_deps=None,
        build_only=False,devices=','.join(map(str,range(16))),run_prefix='test')
    cmds=suite.commands(a)
    builds=[c for c in cmds if any(v.endswith('/build.py') or v.endswith('\\build.py') for v in c)]
    assert len(builds)==1 and '--reuse-workspace-manifest' in builds[0]
    assert 'ready-breakdown' in builds[0]
    assert any('receiver_report.py' in v for command in cmds for v in command)
    assert any('receiver_export.py' in v for v in cmds[-1])
    a.build_only=True
    assert len(suite.commands(a))==1

    # A successfully built but stale wheel must remain intact while the same
    # donor cache builds new artifacts in a revision-specific directory.
    old=tmp_path/'.ablation/wheels/ready-breakdown-ON-s30/manifest.json'
    old.parent.mkdir(parents=True);old.write_text(json.dumps({'commit':'old'}))
    before=old.read_bytes()
    cmds=suite.commands(a)
    assert len(cmds)==1 and '--artifact-suffix' in cmds[0]
    assert '--reuse-workspace-manifest' in cmds[0]
    assert old.read_bytes()==before


def test_replay_marker_write_corrupts_compact_payload_regression():
    import struct
    # E4M3 encodings for -7,-6,-5,-4 from the reported input pattern.
    payload=bytearray(3584);payload[480:484]=bytes([0xce,0xcc,0xca,0xc8])
    corrupt=payload.copy();corrupt[480:484]=struct.pack('<f',1.0)
    def fp8(byte):
        return (-1 if byte&128 else 1)*2**(((byte>>3)&15)-7)*(1+(byte&7)/8) if byte&127 else 0
    actual=list(map(fp8,corrupt[480:484]));expected=list(map(fp8,payload[480:484]))
    assert actual==[0,0,0,1.875] and expected==[-7,-6,-5,-4]
    assert abs(actual[0]-expected[0])==7
    assert abs((actual[3]-expected[3])/expected[3])==1.46875
    source=(ROOT/'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_recv.h').read_text()
    assert 'if (forward) {\n            PrepareServerDedupForward(tail);' in source
    assert '|| !ablationReplay_);' in source

@pytest.fixture
def report_directory(tmp_path):
    spec=importlib.util.spec_from_file_location('report_timer',ROOT/'deep_ep/utils/ll_stage_timer.py')
    timer=importlib.util.module_from_spec(spec);spec.loader.exec_module(timer)
    (tmp_path/'ablation_identity.json').write_text(json.dumps(dict(mode='ready-breakdown',timer='ON',send_cores=30,wheel_sha256='wheel')))
    summary=dict(status='completed',world_size=16,result_count=1,config={'ll_timer_iters':3},results=[{'correctness':'passed'}])
    (tmp_path/'summary.json').write_text(json.dumps(summary))
    for iteration in range(3):
        order=(iteration+2)%3  # Real counter phase must not be inferred from iteration.
        data=sample();raw=[]
        for r in data['records']:
            if r['stage'] in ('REPLAY_BASE_BYTES_PLUS1','REPLAY_SKIP_BYTES_PLUS1'):
                r['calls']=(37 if r['core']<32 else 36)*3696+1
            if r['stage']=='REPLAY_ORDER_PLUS1':r['calls']=order+1
            if r['stage']=='ABLATION_SKIP_FLAGS':r['total_ticks']=30000
            if r['stage']=='ABLATION_NO_PAYLOAD':r['total_ticks']=20000
        for core in range(56):
            pairs=[(0,100),(1,200)]
            for step in range(3):
                kind=(order+step)%3;duration=(40000,30000,20000)[kind]
                pairs.extend([(2*(108+kind),250+step*50000),(2*(108+kind)+1,250+step*50000+duration)])
            pairs.append(((1<<32)|0x8000,100))
            for r in data['records']:
                if r['core']==core:pairs.append(((r['calls']<<32)|(0x8000+timer.STAGES.index(r['stage'])),r['total_ticks']))
            words=[timer.MAGIC,15,1,56,core,1|(30<<8)|((3 if core<30 else 2)<<16),len(pairs),0]
            for event,tick in pairs:words.extend([event,tick])
            words.extend([0]*(640-len(words)));raw.append([words[i:i+8] for i in range(0,640,8)])
        for rank in range(16):
            meta=dict(rank=rank,iteration=iteration,correctness='passed',config={'B':256},case_id=0,route='normal',seed=0,
                      comparison={'ablation_mode':5,'ablation_wheel_sha256':'wheel'})
            (tmp_path/f'r{rank}-i{iteration}.json').write_text(json.dumps(dict(raw=raw,operation='dispatch',frequency_hz=1e9,metadata=meta)))
    return tmp_path


def test_complete_report_and_missing_iteration_rejection(report_directory):
    from receiver_report import report
    tmp_path=report_directory
    full=report(tmp_path);result=next(iter(full['summary'].values()))
    assert result['median_skip_flags_delta_us']==10 and result['median_no_output_delta_us']==20
    assert full['report_version']==2
    assert {w['code'] for w in full['audit_warnings']} >= {'position_and_predecessor_confounded','different_maximum_owners','do_not_attribute_or_add_deltas','unrecorded_layout_and_consumption_sequence'}
    first=result['per_iteration'][0]
    assert first['order']==2 and first['execution_order']==['no-payload-output','baseline','skip-flags']
    assert first['positions']=={'baseline':2,'skip-flags':3,'no-payload-output':1}
    assert result['position_strata']['baseline']['2']['iterations']==[0]
    assert result['position_strata']['skip-flags']['3']['predecessor']=='baseline'
    assert result['order_strata']['2']['iterations']==[0]
    assert first['max_duration_owners']['baseline']==dict(us=40,rank=0,core=0,tie_count=16*56,tie_selection='lowest_rank_then_core')
    assert result['same_core_paired_deltas_us']['skip_flags']['median']==10
    assert result['same_core_paired_deltas_us']['active_core_samples']==16*56*3
    assert result['route_identity']['declared_seeds']==[0]
    assert result['route_identity']['effective_route_seeds']==[]
    assert result['route_identity']['effective_route_seed_status']=='not_recorded_in_all_captures'
    captured=next(iter(full['samples'].values()))[0]
    assert captured['frequency_hz']==1e9 and captured['source_file']=='r0-i0.json'
    assert captured['cores'][0]['arms']['no-payload-output']['first_tick']==250
    assert captured['cores'][0]['arms']['baseline']['first_tick']==50250
    summary=json.loads((tmp_path/'summary.json').read_text())
    summary['config']['ll_timer_iters']=9;(tmp_path/'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError,match='Missing diagnostic iterations'):report(tmp_path)


def test_balanced_but_cross_rank_mismatched_order_is_rejected(report_directory):
    from receiver_report import report
    for iteration in range(3):
        file=report_directory/f'r0-i{iteration}.json';obj=json.loads(file.read_text())
        for core in obj['raw']:
            words=[v for row in core for v in row]
            for index in range(8,8+2*words[6],2):
                if words[index]&0xffffffff==0x8000+117:
                    order=(words[index]>>32)-1
                    words[index]=((((order+1)%3)+1)<<32)|(0x8000+117)
            core[:]=[words[i:i+8] for i in range(0,640,8)]
        file.write_text(json.dumps(obj))
    with pytest.raises(ValueError,match='Cross-rank replay order differs'):report(report_directory)


def test_maximum_owners_are_not_same_core_pairing():
    from receiver_report import maximum_owner,paired_core_summary
    def core(index,base,skip):
        return dict(core=index,arms={
            'baseline':dict(us=base,payload_bytes=3696),
            'skip-flags':dict(us=skip),
            'no-payload-output':dict(us=base-1)})
    rows=[dict(rank=2,cores=[core(4,40,10),core(5,20,35)])]
    assert maximum_owner(rows,'baseline')['core']==4
    assert maximum_owner(rows,'skip-flags')['core']==5
    paired=paired_core_summary(rows)['skip_flags']
    assert paired['min']==-15 and paired['max']==30 and paired['mean']==7.5
    assert paired['positive_count']==1  # max(base)-max(skip)=5 is a different statistic.


def test_omitted_replay_intervals_remain_unknown():
    decoded=decode_sample(sample())
    assert all(arm['first_tick'] is None and arm['last_tick'] is None for core in decoded for arm in core['arms'].values())


def test_recorded_route_identity_is_preserved_without_inferring_legacy_seed(report_directory):
    from receiver_report import report
    for file in report_directory.glob('r*-i*.json'):
        obj=json.loads(file.read_text())
        obj['metadata']['comparison'].update(requested_seed=0,actual_route_seed=2,route_digest='route-sha256')
        file.write_text(json.dumps(obj))
    result=next(iter(report(report_directory)['summary'].values()))
    identity=result['route_identity']
    assert identity['declared_seeds']==[0] and identity['effective_route_seeds']==[2]
    assert identity['effective_route_seed_status']=='recorded_not_independently_verified'
    assert identity['route_digest_status']=='recorded' and identity['route_digests']==['route-sha256']


@pytest.mark.parametrize('field,values',[('actual_route_seed',(2,3)),('route_digest',('route-a','route-b'))])
def test_conflicting_recorded_route_identity_is_rejected(report_directory,field,values):
    from receiver_report import report
    for rank,value in enumerate(values):
        file=report_directory/f'r{rank}-i0.json';obj=json.loads(file.read_text())
        obj['metadata'][field]=value
        file.write_text(json.dumps(obj))
    with pytest.raises(ValueError,match='Conflicting recorded .* within the same capture case'):
        report(report_directory)


def test_partially_recorded_route_identity_keeps_missing_values_unknown(report_directory):
    from receiver_report import report
    file=report_directory/'r0-i0.json';obj=json.loads(file.read_text())
    obj['metadata'].update(actual_route_seed=2,route_digest='route-a')
    file.write_text(json.dumps(obj))
    identity=next(iter(report(report_directory)['summary'].values()))['route_identity']
    assert identity['effective_route_seeds']==[2] and identity['route_digests']==['route-a']
    assert identity['effective_route_seed_status']=='not_recorded_in_all_captures'
    assert identity['route_digest_status']=='not_recorded_in_all_captures'
