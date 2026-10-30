"""CPU accounting, ownership, causal order, CLI isolation and relocation."""
import csv
import importlib.util
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('cpu_dispatch_routes',ROOT/'tests/elastic/inspect_dispatch_routes.py')
model=importlib.util.module_from_spec(spec);sys.modules[spec.name]=model;spec.loader.exec_module(model)


@pytest.mark.parametrize('send',[1,10,28,55])
@pytest.mark.parametrize('b,m',[(1,1),(3,16),(3,256),(33,256)])
@pytest.mark.parametrize('route',['normal','hotspot'])
def test_conservation_and_exact_static_receive_ownership(send,b,m,route):
    routes=model.ST.make_routes(16,b,8,384,17,route)
    plan=model.build_plan(routes,384,send,56-send,4,m)
    totals,_=model.replay(plan,lambda row:None)
    assert totals['receive_packets']==16*b*8
    assert totals['initial_packets']+totals['forward_packets']==totals['receive_packets']
    if m<=32:assert totals['forward_packets']==0 and totals['initial_packets']==16*b*8
    for rank in range(16):
        incoming=sum(len(v) for v in model.ST.expert_entries(routes,384,16,rank))
        for core in range(56):
            worker=plan['workers'][rank,core]
            assert worker.counts['receive_packets']==(core+1)*incoming//56-core*incoming//56
            assert worker.counts['forward_packets']>=worker.counts['receive_first_hop']
            assert worker.counts['forward_packets']==worker.counts['forward_same_rank']+worker.counts['forward_cross_rank']
            assert worker.counts['send_packets']==len(worker.initial)


def test_counts_are_independent_of_ratio_but_initial_work_changes():
    routes=model.ST.make_routes(16,9,8,384,0)
    snapshots=[]
    for send in (10,28):
        plan=model.build_plan(routes,384,send,56-send,4,256)
        model.replay(plan,lambda row:None)
        snapshots.append(list(model.count_rows(plan,16)))
    assert [r['receive_packets'] for r in snapshots[0]]==[r['receive_packets'] for r in snapshots[1]]
    assert [r['forward_packets'] for r in snapshots[0]]==[r['forward_packets'] for r in snapshots[1]]
    assert [r['send_packets'] for r in snapshots[0]]!=[r['send_packets'] for r in snapshots[1]]


def test_full_requested_shape_matches_independent_packet_formula():
    routes=model.ST.make_routes(16,256,8,384,2)
    plan=model.build_plan(routes,384,28,28,4,256)
    expected=0
    for rank,rows in enumerate(routes):
        for ids in rows:
            local=sum(e//96==rank//4 for e in ids)
            groups={e//96 for e in ids if e//96!=rank//4}
            expected+=local+len(groups)
    totals,_=model.replay(plan,lambda row:None)
    assert totals['initial_packets']==expected
    assert totals['receive_packets']==32768 and totals['forward_packets']==32768-expected


@pytest.mark.parametrize('world',[32,128])
@pytest.mark.parametrize('capacity',[16,256])
def test_multinode_counts_cover_every_rank(world,capacity):
    routes=model.ST.make_routes(world,3,8,384,2)
    plan=model.build_plan(routes,384,10,46,4,capacity)
    totals,_=model.replay(plan,lambda row:None)
    rows=list(model.count_rows(plan,16))
    assert len(rows)==world*56 and {r['node'] for r in rows}==set(range(world//16))
    assert totals['receive_packets']==world*3*8


def test_one_group_has_no_dedup_relays_and_unsafe_group_bitmask_is_rejected():
    routes=model.ST.make_routes(16,3,8,384,2)
    plan=model.build_plan(routes,384,10,46,16,256)
    totals,_=model.replay(plan,lambda row:None)
    assert totals['forward_packets']==0 and totals['initial_packets']==16*3*8
    large=model.ST.make_routes(128,1,3,384,0)
    with pytest.raises(ValueError,match='uint32 group mask'):
        model.build_plan(large,384,28,28,1,256)


def test_no_global_sender_done_barrier_is_introduced():
    routes=[[[((rank//4+1)%4)*96+j for j in range(8)] for token in range(3)] for rank in range(16)]
    plan=model.build_plan(routes,384,1,55,4,256)
    events=[];model.replay(plan,events.append)
    last={rank:max(e['logical_round'] for e in events if e['rank']==rank and e['stage']=='INITIAL_SEND')
          for rank in range(16)}
    assert any(e['stage']=='RECV_INPUT' and e['logical_round']<last[e['rank']] for e in events)


def test_forward_before_output_and_terminal_cannot_precede_parent():
    routes=[]
    # Every origin sends four experts to one card in its next group. That
    # gives both cross-rank FIRST_HOP and SAME-rank terminal publications.
    for rank in range(16):
        begin=((rank//4+1)%4)*16
        routes.append([[begin,begin+1,begin+2,begin+3]])
    plan=model.build_plan(routes,64,28,28,4,256,hidden=128)
    events=[];totals,_=model.replay(plan,events.append)
    assert totals['initial_packets']==16 and totals['forward_packets']==48
    initial={e['route_id']:e for e in events if e['stage']=='INITIAL_SEND'}
    relays={e['route_id']:e for e in events if e['stage']=='RELAY_ISSUE'}
    assert all(e['packet_type']=='TERMINAL' and e['route_id']!=e['parent_route'] for e in relays.values())
    inputs={e['route_id']:e for e in events if e['stage']=='RECV_INPUT'}
    outputs={e['route_id']:e for e in events if e['stage']=='RECV_OUTPUT'}
    for route in plan['routes']:
        published=relays[route.id] if route.kind=='TERMINAL' else initial[route.id]
        assert inputs[route.id]['logical_round']>published['logical_round']
        if route.kind=='FIRST_HOP':
            assert all(relays[i]['sequence']<outputs[route.id]['sequence'] for i in route.members)
    assert sum(w.counts['forward_same_rank'] for w in plan['workers'].values())==48
    assert model.balanced_owner(0,1,46)==0  # Initial expert ownership is remainder-first.
    singleton=[r for r in plan['routes'] if r.destination==0]
    if singleton:
        total=len(singleton)
        assert sorted(r.receiver for r in singleton)==sorted(((row+1)*56-1)//total for row in range(total))


def test_receive_priority_is_within_ready_batch_only():
    for size in range(1,7):
        for ready in range(1<<size):
            for priority in (0,ready,(1<<size)-1,1):
                result=model.priority_order(size,ready,priority)
                expected=sorted((i for i in range(size) if ready&(1<<i)),
                                key=lambda i:(0 if priority&(1<<i) else 1,-i))
                assert result==expected


@pytest.mark.parametrize('groups',[1,4,8,16,17,32,64])
def test_direct_group_order_is_permutation(groups):
    for own in range(groups):
        assert sorted(model.direct_group_order(groups,own,i) for i in range(groups))==list(range(groups))


def test_route_generator_is_same_seed_and_expert_order_as_st():
    rng=random.Random(3)
    expected=[[rng.sample(range(384),8) for _ in range(3)] for rank in range(16)]
    assert model.ST.make_routes(16,3,8,384,3)==expected
    plan=model.build_plan(expected,384,10,46,4,256)
    for rank in range(16):
        ordered=sorted((r for r in plan['routes'] if r.destination==rank),key=lambda r:r.row)
        reference=[(rank*24+e,source,token) for e,entries in enumerate(model.ST.expert_entries(expected,384,16,rank))
                   for source,token in entries]
        assert [(r.expert,r.source,r.token) for r in ordered]==reference


def run_without_accelerator(project,tmp_path,run_name='cpu_test',extra=()):
    wrapper=tmp_path/'no_accelerator.py'
    if not wrapper.exists():wrapper.write_text('''
import importlib.abc,runpy,sys
class BlockAccelerator(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','torch_npu','deep_ep','acl'):
   raise AssertionError('Unexpected accelerator import: '+fullname)
sys.meta_path.insert(0,BlockAccelerator())
entry=sys.argv.pop(1)
runpy.run_path(entry,run_name='__main__')
''')
    entry=project/'tests/elastic/inspect_dispatch_routes.py'
    args=[sys.executable,'-S','-X','utf8',str(wrapper),str(entry),
          '--num-processes','16','--num-nodes','1','--node-rank','0',
          '--master-addr','127.0.0.1','--master-port','29500','--shmem-port','29501',
          '--num-tokens','3','--num-max-tokens-per-rank','256',
          '--hidden','3584','--num-topk','8','--num-experts','384','--dtype','fp8',
          '--ll-reference-profile','--reference-iters','5','--dispatch-only','--do-cpu-sync','0',
          '--send-cores','10','--recv-cores','46','--output-dir',str(tmp_path/'reports'),
          '--run-name',run_name,*extra]
    result=subprocess.run(args,cwd=tmp_path,capture_output=True,text=True,encoding='utf-8')
    return result,tmp_path/'reports'/run_name/'cpu-dispatch-routes'


def test_full_command_no_accelerator_and_no_overwrite(tmp_path):
    result,output=run_without_accelerator(ROOT,tmp_path)
    assert result.returncode==0,result.stderr
    manifest=json.loads((output/'manifest.json').read_text())
    assert manifest['forbidden_modules_loaded']==[] and manifest['world_size']==16
    assert manifest['cases'][0]['logical_routes']==16*3*8  # NOT multiplied by timer-iters=5.
    core=list(csv.DictReader((output/'case_0000/core_counts.csv').open(encoding='utf-8-sig')))
    assert len(core)==16*56 and sum(int(r['receive_packets']) for r in core)==16*3*8
    ranks=list(csv.DictReader((output/'case_0000/rank_counts.csv').open(encoding='utf-8-sig')))
    assert len(ranks)==16 and all(int(r['send_unique_tokens'])==3 for r in ranks)
    routing=list(csv.DictReader((output/'case_0000/routing_counts.csv').open(encoding='utf-8-sig')))
    assert all(int(r['logical_routes'])==int(r['initial_packets'])+int(r['relay_packets']) for r in routing)
    topk=json.loads((output/'case_0000/topk_ids.json').read_text())
    assert topk['topk_ids']==model.ST.make_routes(16,3,8,384,2)
    assert topk['base_seed']==0 and manifest['cases'][0]['effective_seed']==2
    stages=list(csv.DictReader((output/'case_0000/stage_counts.csv').open(encoding='utf-8-sig')))
    keys=[(int(r['rank']),int(r['core']),int(r['logical_round']),int(r['first_core_sequence'])) for r in stages]
    assert keys==sorted(keys)
    totals=manifest['cases'][0]
    for stage,key in [('INITIAL_SEND','initial_packets'),('RECV_INPUT','receive_packets'),
                      ('RECV_OUTPUT','receive_packets'),('RELAY_ISSUE','forward_packets')]:
        assert sum(int(r['packets']) for r in stages if r['stage']==stage)==totals[key]
    before=(output/'manifest.json').read_bytes()
    repeat,_=run_without_accelerator(ROOT,tmp_path)
    assert repeat.returncode!=0 and (output/'manifest.json').read_bytes()==before


def test_relocated_project_with_spaces_and_matrix(tmp_path):
    project=tmp_path/'project with spaces';project.mkdir()
    files=['tests/elastic/test_ep_low_latency.py','tests/elastic/inspect_dispatch_routes.py','scripts/ll_dispatch_implementation.py',
           'csrc/kernels/dispatch/low_latency/common/ll_core_config.h',
           'csrc/kernels/dispatch/low_latency/common/ll_group_local_send.h',
           'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_receive_tasks.h',
           'csrc/kernels/dispatch/low_latency/common/ep_memory_server_dedup_protocol.h',
           'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h',
           'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_recv.h',
           'csrc/kernels/dispatch/low_latency/common/ll_receive_priority.h',
           'csrc/kernels/dispatch/low_latency/common/ll_send_order.h','csrc/kernels/common/ep_server_order.h']
    for name in files:
        destination=project/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,destination)
    result,output=run_without_accelerator(project,tmp_path,'relocated',('--dtype','bf16','fp8'))
    assert result.returncode==0,result.stderr
    manifest=json.loads((output/'manifest.json').read_text())
    assert len(manifest['cases'])==2 and (output/'case_0001/events.csv').is_file()


@pytest.mark.parametrize('phase,seeds',[('initial',[0]),('all',[0,1,2])])
def test_initial_and_all_route_phases_follow_correctness_repeats(tmp_path,phase,seeds):
    result,output=run_without_accelerator(ROOT,tmp_path,phase,('--route-phase',phase))
    assert result.returncode==0,result.stderr
    manifest=json.loads((output/'manifest.json').read_text())
    assert [r['effective_seed'] for r in manifest['cases']]==seeds


def test_explicit_multinode_domain_and_invalid_split():
    args=model.parse_args(['--num-processes','16','--num-nodes','2','--node-rank','1','--send-cores','10'])
    assert (args.num_processes*args.num_nodes,args.send_cores,args.recv_cores)==(32,10,46)
    with pytest.raises(ValueError,match='S/R'):model.parse_args(['--send-cores','10','--recv-cores','28'])
