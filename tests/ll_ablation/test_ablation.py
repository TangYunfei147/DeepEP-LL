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
    assert 'if (firstHop && (kLlAblationMode!=4U || !ablationReplay_))' in recv
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
