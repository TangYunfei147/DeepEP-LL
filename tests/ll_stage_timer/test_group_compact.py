"""Run the production ownership/collapse helper against an independent route oracle.

These are CPU packet/metadata invariants, not AscendC compile or device validation.
Set CXX to a compiler command (for example 'zig c++') to use an existing toolchain.
"""
from collections import Counter, defaultdict
from pathlib import Path
import os
import random
import shlex
import shutil
import subprocess
import importlib.util
import sys
import pytest

ROOT=Path(__file__).resolve().parents[2]
_spec=importlib.util.spec_from_file_location('compact_report',ROOT/'tests/elastic/inspect_dispatch_routes.py')
report=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=report;_spec.loader.exec_module(report)


@pytest.fixture(scope='module')
def native(tmp_path_factory):
    compiler=shlex.split(os.environ.get('CXX',''))
    if not compiler:
        found=shutil.which('g++') or shutil.which('clang++')
        if not found:
            pytest.skip('C++ compiler needed for production group scheduling helper')
        compiler=[found]
    out=tmp_path_factory.mktemp('group-compact')/('schedule.exe' if os.name=='nt' else 'schedule')
    subprocess.run([*compiler,'-std=c++17','-O2','-iquote',str(ROOT),
                    str(Path(__file__).parent/'fixtures/group_compact.cpp'),'-o',str(out)],check=True)
    return out


def run(native,ids,world,experts,send,rank,dedup):
    b,k=len(ids),len(ids[0])
    text=f'{world} {experts} 4 {send} {rank} {int(dedup)} {b} {k}\n'
    text+=' '.join(str(e) for row in ids for e in row)+'\n'
    groups=world//4
    text+=' '.join(str(report.direct_group_order(groups,rank//4,(step+rank%4)%groups))
                   for step in range(groups))+'\n'
    output=subprocess.run([str(native)],input=text,text=True,capture_output=True,check=True).stdout
    assert output.endswith('END\n')
    return [tuple(map(int,line.split())) for line in output.splitlines()[:-1]]


def verify(packets,ids,world,experts,send,rank,dedup):
    k=len(ids[0]); group_experts=experts//world*4; own=rank//4
    # Independent logical expert routes -> packet oracle, without ownership logic.
    expected={}; remote=set(); original_q={}; histogram=Counter()
    for token,row in enumerate(ids):
        buckets=defaultdict(list)
        for slot,expert in enumerate(row):
            index=token*k+slot
            original_q[index]=histogram[expert];histogram[expert]+=1
            if expert//group_experts!=own:
                remote.add(index);buckets[expert//group_experts].append(index)
        for members in buckets.values():
            if dedup and len(members)>1:
                expected[members[0]]=sum(1<<(i%k) for i in members)
            else:
                expected.update((i,0) for i in members)
    assert len(packets)==len(expected)
    assert {i:mask for _,i,mask in packets}==expected
    counts=[0]*send;by_group=defaultdict(lambda:[0]*send);by_phase=defaultdict(lambda:[0]*send)
    seen=set();direct_seen=set();metadata=defaultdict(dict)
    for core,index,mask in packets:
        assert 0<=core<send
        token,slot=divmod(index,k);expert=ids[token][slot]
        if mask:
            assert core not in direct_seen
            assert mask & (1<<slot) and mask & ((1<<slot)-1)==0
            members=[token*k+j for j in range(k) if mask>>j&1]
        else:
            direct_seen.add(core);members=[index]
        assert not seen.intersection(members);seen.update(members)
        # Restore route/q/weight identities exactly. Deterministic nonuniform
        # weights catch confusing global token IDs with the old owner's slice.
        for i in members:
            t,j=divmod(i,k)
            metadata[t][j]=(ids[t][j],original_q[i],(t*37+j*11+1)/128)
        counts[core]+=1
        by_group[expert//group_experts][core]+=1
        by_phase[bool(mask)][core]+=1
    assert seen==remote
    assert max(counts)-min(counts)<=1
    for values in by_phase.values(): assert max(values)-min(values)<=1
    # Each group's FIRST_HOP and DIRECT runs are individually balanced. Across
    # those two disjoint runs their sum can differ by TWO, not necessarily one.
    for values in by_group.values(): assert max(values)-min(values)<=2
    for i in remote:
        t,j=divmod(i,k)
        assert metadata[t][j]==(ids[t][j],original_q[i],(t*37+j*11+1)/128)


@pytest.mark.parametrize('world',[16,32,64,128])
@pytest.mark.parametrize('send',[1,10,28,55])
@pytest.mark.parametrize('dedup',[False,True])
def test_packet_equivalence_and_balance(native,world,send,dedup):
    experts=world*8
    rng=random.Random(8501+world+send)
    for b,k in [(1,8),(3,3),(8,8),(32,16),(48,32)]:
        rank=(world//2)+3
        ids=[rng.sample(range(experts),k) for _ in range(b)]
        verify(run(native,ids,world,experts,send,rank,dedup),ids,world,experts,send,rank,dedup)


@pytest.mark.parametrize('dedup',[False,True])
def test_skew_empty_groups_high_mask_and_foreign_tokens(native,dedup):
    world,experts,send,rank=64,512,28,3
    cases=[[[i for i in range(32)] for _ in range(3)], # no remote packets
           [[i for i in range(32,64)] for _ in range(8)], # bit31, all one remote group
           [[i*8 for i in range(8,40)]], # B<S, owners beyond metadata token owners
           [[(t*31+j)%experts for j in range(32)] for t in range(48)]]
    for ids in cases:
        packets=run(native,ids,world,experts,send,rank,dedup)
        verify(packets,ids,world,experts,send,rank,dedup)
    ids=cases[2]
    packets=run(native,ids,world,experts,send,rank,dedup)
    assert any(core>0 for core,_,_ in packets)


def test_identity_and_workspace_contract():
    common=ROOT/'csrc/kernels/dispatch/low_latency/common'
    assert '15000U + kLlSendCoreNum' in (common/'ll_core_config.h').read_text(encoding='utf-8')
    assert 'kLlOverlapExperimentStage = 3U' in (common/'ll_overlap_stage.h').read_text(encoding='utf-8')
    sender=(common.parent/'device/dispatch_server_dedup_send.h').read_text(encoding='utf-8')
    assert 'if (enableDedup_ && count_ == 0U) return;' not in sender
    assert 'token -= start_' not in sender
    assert 'LlVisitCompactGroup(' in sender
    assert 'SyncAll' not in sender
    assert 'Allocate(directBuf_, qBytes)' in sender


@pytest.mark.parametrize('world',[16,64])
@pytest.mark.parametrize('capacity',[16,256])
def test_cpu_report_matches_native_ownership_and_baseline_packets(native,world,capacity):
    ids=report.ST.make_routes(world,3,8,384,2)
    before=report.build_plan(ids,384,28,28,4,capacity,send_policy='s3')
    after=report.build_plan(ids,384,28,28,4,capacity,send_policy='group-compact')
    assert [(r.expert,r.q,r.root,r.members,r.kind,r.row,r.receiver) for r in before['routes']]==[
        (r.expert,r.q,r.root,r.members,r.kind,r.row,r.receiver) for r in after['routes']]
    for rank in (0,3,world-1):
        packets=run(native,ids[rank],world,384,28,rank,capacity>32)
        expected=[]
        for core in range(28):
            for i in after['workers'][rank,core].initial:
                item=after['routes'][i]
                mask=sum(1<<after['routes'][j].slot for j in (i,*item.members)) if item.members else 0
                expected.append((core,i-rank*3*8,mask))
        assert packets==expected
