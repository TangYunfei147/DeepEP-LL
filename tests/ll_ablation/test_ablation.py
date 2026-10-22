import importlib.util
import json
from pathlib import Path
import sys
import pytest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from common import ROOT,MODES
from compare import compare

def fake_run(root,mode,time):
    path=root/mode;path.mkdir()
    (path/'ablation_identity.json').write_text(json.dumps(dict(mode=mode,commit='x',send_cores=28,recv_cores=28,timer='OFF')))
    metadata={'routes':{'source_weights':'w','destination_index':'d','send_counts':'c'}}
    for rank in range(16):(path/f'ablation_metadata_rank{rank:03}.json').write_text(json.dumps(metadata))
    (path/'summary.json').write_text(json.dumps(dict(status='completed',world_size=16,case_count=1,result_count=1,
        config=dict(seed=1,warmup=10,num_iters=100,perf_tail=50,delay_rows=100),
        results=[dict(config={'B':256},route='random',correctness='passed',dispatch={'max_rank_mean_kernel_us':time})])))
    return path

def test_compare_matched_off_and_metadata(tmp_path):
    paths={mode:fake_run(tmp_path,mode,t) for mode,t in [('baseline',88),('fixed-payload',87),('preload',80),('prepack',74)]}
    row=compare(paths)[0]
    assert row['combined_payload_delta_us']==13 and row['preload_delta_us']==7
    p=paths['prepack']/'ablation_metadata_rank000.json';p.write_text('{}')
    with pytest.raises(ValueError,match='metadata mismatch'):compare(paths)

def test_schema_replay_metric_is_separate_from_kernel():
    spec=importlib.util.spec_from_file_location('ablation_timer',ROOT/'deep_ep/utils/ll_stage_timer.py')
    timer=importlib.util.module_from_spec(spec);spec.loader.exec_module(timer)
    raw=[]
    for core in range(56):
        role=3 if core<28 else 2
        records=[(0,100),(1,200),(216,250),(217,280),((1<<32)|0x8000,100),((1<<32)|(0x8000+108),30)]
        words=[timer.MAGIC,14,1,56,core,1|(28<<8)|(role<<16),len(records),0]
        for event,tick in records:words += [event,tick]
        words += [0]*(640-len(words));raw.append([words[i:i+8] for i in range(0,640,8)])
    result=timer.parse_records(raw,'dispatch',1e9)
    assert len([r for r in result['records'] if r['stage']=='ABLATION_READY_RECV' and r['total_ticks']==30])==56

@pytest.mark.parametrize('h',[128,512,3584,4096,7168])
def test_prepacked_constant_payload_matches_copy_layout(h):
    # Independent byte-level mapping: every 480 payload bytes get 32 flag bytes.
    scale_bytes=(h//128)*4
    logical=bytes(h)+bytes([127])*scale_bytes+bytes(128)
    wire=bytearray()
    import struct
    for start in range(0,len(logical),480):
        wire+=logical[start:start+480].ljust(480,b'\0')+struct.pack('<8f',*([1.0]*8))
    for i in range(scale_bytes):assert wire[h+i+(h+i)//480*32]==127
    for i in range(h):assert wire[i+i//480*32]==0

def test_replay_keeps_packet_flags_until_second_consumer():
    recv=(ROOT/'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_recv.h').read_text()
    assert 'if (ablationReplay_) ClearFlags();' in recv
    assert 'const bool forward = firstHop &&' in recv
    assert '((kLlAblationMode!=4U && kLlAblationMode!=5U) || !ablationReplay_);' in recv
    assert recv.count('if (forward) {')==2
    assert 'timer_=nullptr' in recv
    entry=(ROOT/'csrc/kernels/dispatch/low_latency/dispatch_variant_kernel.h').read_text()
    assert entry.index('timer.End(LlStageTimer::KERNEL')<entry.index('kernel.ReplayReadyForAblation')

def test_all_modes_have_distinct_protocol_identity():
    assert len({29000+100*m+28 for m in MODES.values()})==len(MODES)

def test_offline_dependency_escape_rejected(tmp_path):
    from deps import unpack
    bundle=tmp_path/'bundle';bundle.mkdir()
    root=tmp_path/'repo';root.mkdir()
    (bundle/'deps.json').write_text(json.dumps({'entries':[{'path':'../escape','file':'x','sha256':'x','revision':'x'}]}))
    with pytest.raises(ValueError,match='Unsafe dependency path'):unpack(bundle,root)

def test_compare_rejects_on_latency(tmp_path):
    paths={mode:fake_run(tmp_path,mode,88) for mode in ['baseline','fixed-payload','preload','prepack']}
    p=paths['preload']/'ablation_identity.json';d=json.loads(p.read_text());d['timer']='ON';p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='requires OFF'):compare(paths)

def test_offline_dependency_roundtrip(tmp_path):
    import shutil,subprocess
    from deps import pack,unpack
    if not shutil.which('git'):pytest.skip('Git required')
    def git(path,*args):
        return subprocess.check_output(['git','-C',str(path),'-c','user.name=codex_remote',
            '-c','user.email=799847028@qq.com',*args],text=True,stderr=subprocess.STDOUT).strip()
    source=tmp_path/'source';source.mkdir();git(source,'init')
    dep=source/'dep';dep.mkdir();git(dep,'init');(dep/'a.txt').write_text('dependency')
    git(dep,'add','a.txt');git(dep,'commit','-m','dependency');rev=git(dep,'rev-parse','HEAD')
    git(source,'update-index','--add','--cacheinfo','160000',rev,'dep');git(source,'commit','-m','pin')
    bundle=tmp_path/'bundle';pack(source,bundle)
    target=tmp_path/'target';target.mkdir();git(target,'init')
    git(target,'update-index','--add','--cacheinfo','160000',rev,'dep');git(target,'commit','-m','pin')
    unpack(bundle,target)
    assert git(target/'dep','rev-parse','HEAD')==rev
    assert (target/'dep/a.txt').read_text()=='dependency'
    from common import source_checkout
    (target/'notes.txt').write_text('user notes must remain untouched')
    (target/'dep/a.txt').write_text('dependency working-tree experiment')
    assert source_checkout(target)['source_policy']=='committed_HEAD'
    assert (target/'notes.txt').read_text()=='user notes must remain untouched'
    assert (target/'dep/a.txt').read_text()=='dependency working-tree experiment'
    git(target/'dep','add','a.txt');git(target/'dep','commit','-m','different revision')
    with pytest.raises(RuntimeError,match='dep'):source_checkout(target)

@pytest.mark.parametrize('staged',[False,True])
def test_source_guard_rejects_actual_source_edits(tmp_path,staged):
    import subprocess
    from common import source_checkout
    def git(*args):
        return subprocess.check_output(['git','-C',str(tmp_path),'-c','user.name=codex_remote',
            '-c','user.email=799847028@qq.com',*args],text=True,stderr=subprocess.STDOUT).strip()
    git('init');file=tmp_path/'kernel.cpp';file.write_text('original\n')
    git('add','kernel.cpp');git('commit','-m','base')
    file.write_text('changed\n')
    if staged:
        git('add','kernel.cpp')
        file.write_text('original\n') # Index differs even though working tree matches HEAD.
    with pytest.raises(RuntimeError,match='kernel.cpp'):source_checkout(tmp_path)

def test_suite_builds_every_wheel_before_run_and_reuses_baseline(tmp_path,monkeypatch):
    import argparse
    import suite
    from common import sha
    monkeypatch.setattr(suite,'ROOT',tmp_path)
    config=tmp_path/'config.json';config.write_text('{}')
    directory=tmp_path/'.ablation/wheels/baseline-OFF-s30';directory.mkdir(parents=True)
    wheel=directory/'baseline.whl';wheel.write_bytes(b'wheel')
    (directory/'manifest.json').write_text(json.dumps(dict(mode='baseline',timer='OFF',send_cores=30,
        recv_cores=26,commit='old-source',wheel=wheel.name,wheel_sha256=sha(wheel))))
    a=argparse.Namespace(config=config,resume=True,send_cores=30,build=True,build_only=True,
        offline_deps=None,devices='',run_prefix='build-only')
    plan=suite.plan_commands(a)
    assert plan['reused_modes']==['baseline']
    assert len(plan['phases']['build'])==4
    assert not plan['phases']['install'] and not plan['phases']['run']
    assert all(c[c.index('--revision')+1]=='old-source' for c in plan['phases']['build'])
    assert all('--resume' in c for c in plan['phases']['build'])
    a.build_only=False
    plan=suite.plan_commands(a)
    assert list(plan['phases'])==['build','install','run']
    assert len(plan['phases']['run'])==5
    wheel.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='corrupt'):suite.plan_commands(a)

@pytest.mark.parametrize('leftovers',['source','output','both'])
def test_incomplete_build_retry_preserves_files(tmp_path,leftovers):
    from build import build_paths
    tag='fixed-payload-OFF-s30'
    old_source=tmp_path/'.ablation/sources'/tag
    old_output=tmp_path/'.ablation/wheels'/tag
    paths=[]
    if leftovers in ('source','both'):paths.append(old_source)
    if leftovers in ('output','both'):paths.append(old_output)
    for path in paths:path.mkdir(parents=True);(path/'keep.txt').write_text('preserve')
    with pytest.raises(RuntimeError,match='--resume'):build_paths(tmp_path,tag,False)
    source,output,canonical=build_paths(tmp_path,tag,True)
    assert source!=old_source and output.parent==old_output and canonical==old_output
    assert not source.exists() and not output.exists()
    assert all((path/'keep.txt').read_text()=='preserve' for path in paths)
    assert build_paths(tmp_path,tag,True)[0]!=source

def test_completed_build_is_not_overwritten(tmp_path):
    from build import build_paths
    output=tmp_path/'.ablation/wheels/baseline-OFF-s30';output.mkdir(parents=True)
    manifest=output/'manifest.json';manifest.write_text('{}')
    with pytest.raises(RuntimeError,match='Completed'):build_paths(tmp_path,'baseline-OFF-s30',True)
    assert manifest.read_text()=='{}'
