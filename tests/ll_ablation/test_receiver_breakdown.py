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
    config=tmp_path/'case.json';config.write_text('{}')
    donor=tmp_path/'.ablation/wheels/ready-replay-ON-s30/manifest.json'
    donor.parent.mkdir(parents=True);donor.write_text('{}')
    a=argparse.Namespace(config=config,send_cores=30,fresh_build=False,offline_deps=None,
        build_only=False,devices=','.join(map(str,range(16))),run_prefix='test')
    cmds=suite.commands(a)
    builds=[c for c in cmds if any(v.endswith('/build.py') or v.endswith('\\build.py') for v in c)]
    assert len(builds)==1 and '--reuse-workspace-manifest' in builds[0]
    assert 'ready-breakdown' in builds[0]
    assert any('receiver_report.py' in v for v in cmds[-1])
    a.build_only=True
    assert len(suite.commands(a))==1
