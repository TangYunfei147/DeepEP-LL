"""CPU-only routing/accounting and a causal replay of relay-priority dispatch.

All ranks are modeled in ONE process. No Torch/native library, network, device,
payload tensor, timer capture or performance measurement is used.
"""
import argparse
from collections import Counter
import csv
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import uuid

ROOT=Path(__file__).resolve().parents[2]
_spec=importlib.util.spec_from_file_location('_cpu_dispatch_st',ROOT/'tests/elastic/test_ep_low_latency.py')
ST=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ST)  # This module's imports are all standard-library.
TOTAL_CORES=56
BATCH=16


def align(n,a=32):
    return (n+a-1)//a*a


def split(n,workers,core):
    return core*(n//workers)+min(core,n%workers),n//workers+(core<n%workers)


def balanced_owner(index,n,workers):
    base,extra=divmod(n,workers)
    boundary=(base+1)*extra
    return index//(base+1) if index<boundary else extra+(index-boundary)//base


def direct_group_order(groups,own,ordinal):
    if groups<16:
        return (own+ordinal)%groups
    half=groups//2
    lower=own<half
    own_base=0 if lower else half
    own_size=half if lower else groups-half
    relative=own-own_base
    if ordinal<2*half:
        ordinal=ordinal//2+(0 if ordinal%2==0 else own_size)
    else:
        ordinal=half if own_size>half else groups-1
    if ordinal<own_size:
        return own_base+(relative+ordinal+1)%own_size
    other_base=half if lower else 0
    return other_base+(relative+ordinal-own_size+1)%(groups-own_size)


def priority_order(size,ready,first_hop):
    ready&=(1<<size)-1
    first=ready&first_hop
    return [i for mask in (first,ready&~first) for i in reversed(range(size)) if mask&(1<<i)]


@dataclass
class Route:
    id:int
    source:int
    token:int
    slot:int
    expert:int
    q:int
    destination:int
    row:int=-1
    receiver:int=-1
    kind:str='DIRECT'
    root:int=-1
    initial_core:int=-1
    members:tuple=()


@dataclass
class Worker:
    rank:int
    core:int
    initial:list=field(default_factory=list)
    pending:list=field(default_factory=list)
    sent:int=0
    scan:int=0
    lanes:int=0
    issued:int=0
    consumed:int=0
    prefix:bool=False
    event_index:int=0
    counts:Counter=field(default_factory=Counter)
    unique:dict=field(default_factory=lambda:{k:set() for k in ('send','receive','forward')})


def packet_layout(h,k,dtype):
    scale=((h+127)//128)*4 if dtype=='fp8' else 0
    header=align(align(h,256)+scale if dtype=='fp8' else h*2)
    index=header+32
    if index%480+k*8>480:
        index=align(index,480)
    blocks=(align(index+k*8)+479)//480
    return blocks,blocks*512


def build_plan(topk,experts,send,recv,group,capacity,hidden=3584,dtype='fp8'):
    if not topk or not topk[0] or not topk[0][0]:raise ValueError('Empty route tensor')
    world=len(topk);b=len(topk[0]);k=len(topk[0][0])
    if not 1<=k<=32 or not b<=capacity or not experts>=world:raise ValueError('Invalid B/M/K/E')
    if send<=0 or recv<=0 or send+recv!=TOTAL_CORES:
        raise ValueError('Require positive S/R and S+R=56')
    if not 1<=group<=16 or world%group or experts%world:
        raise ValueError('Require G=1..16 dividing EP, and E divisible by EP')
    local=experts//world;group_experts=local*group;groups=world//group
    dedup=capacity>32
    if dedup and groups>32:raise ValueError('Current native Classify has a uint32 group mask; dedup above 32 groups is not faithfully modeled')
    workers={(r,c):Worker(r,c) for r in range(world) for c in range(TOTAL_CORES)}
    routes=[];incoming=[[] for _ in range(world)]
    for source,rows in enumerate(topk):
        if len(rows)!=b:raise ValueError('All ranks must have identical B')
        histogram=Counter()
        for token,ids in enumerate(rows):
            if len(ids)!=k or len(set(ids))!=k or any(type(e) is not int or not 0<=e<experts for e in ids):
                raise ValueError('Each token needs K distinct valid expert IDs')
            for slot,e in enumerate(ids):
                item=Route(len(routes),source,token,slot,e,histogram[e],e//local)
                histogram[e]+=1;item.root=item.id
                routes.append(item);incoming[item.destination].append(item.id)
    # Fixed output order: local expert, source rank, original route order/q.
    for rank,ids in enumerate(incoming):
        ids.sort(key=lambda i:(routes[i].expert%local,routes[i].source,routes[i].q))
        total=len(ids)
        for row,i in enumerate(ids):
            item=routes[i];item.row=row
            item.receiver=((row+1)*TOTAL_CORES-1)//total
            workers[rank,item.receiver].pending.append(i)
    blocks,stride=packet_layout(hidden,k,dtype)
    if not 0<blocks<=255:raise ValueError('Packet blockCount outside native bound')
    for worker in workers.values():
        count=len(worker.pending)
        if not count:continue
        fixed=(align(experts*4)+2*align(count*4)+BATCH*blocks*32+256+
               BATCH*align((blocks+7)//8)*4+256+BATCH*36)
        worker.lanes=min(8,(190*1024-fixed)//(stride+64))
        if worker.lanes<1:raise ValueError('Actual receive partition does not fit native UB budget')
    # R source ownership is by local-group expert interval, NOT by token.
    for item in routes:
        own=item.source//group
        if item.expert//group_experts==own:
            owner=balanced_owner(item.expert-own*group_experts,group_experts,recv)
            item.initial_core=send+owner
            workers[item.source,item.initial_core].initial.append(item.id)
    for rank in range(world):
        own=rank//group
        for core in range(send):
            worker=workers[rank,core]
            direct=[]
            if dedup:
                start,count=split(b,send,core)
                for token in range(start,start+count):
                    by_group={};first_hops=[]
                    for slot in range(k):
                        i=(rank*b+token)*k+slot;item=routes[i];g=item.expert//group_experts
                        if g==own:continue
                        slots=by_group.setdefault(g,[]);slots.append(i)
                        if len(slots)==2:first_hops.append(g)  # Classify creates a group on SECOND hit.
                    for members in by_group.values():
                        root=routes[members[0]]
                        root.initial_core=core
                        if len(members)==1:direct.append(root.id);continue
                        root.kind='FIRST_HOP';root.members=tuple(members[1:])
                        for i in members[1:]:
                            routes[i].kind='TERMINAL';routes[i].root=root.id;routes[i].initial_core=core
                    n=len(first_hops)
                    for ordinal in range(n):
                        scheduled=(ordinal+rank%group+core+token)%n
                        worker.initial.append(by_group[first_hops[scheduled]][0])
                for ordinal in range(groups):
                    phase=(ordinal+rank%group+core)%groups
                    g=direct_group_order(groups,own,phase)
                    if g!=own:
                        worker.initial.extend(i for i in direct if routes[i].expert//group_experts==g)
            else:
                start,count=split(b*k,send,core)
                ids=[rank*b*k+i for i in range(start,start+count)
                     if routes[rank*b*k+i].expert//group_experts!=own]
                for i in ids:routes[i].initial_core=core
                if groups<16:worker.initial.extend(ids)
                else:
                    for ordinal in range(groups):
                        g=direct_group_order(groups,own,ordinal)
                        if g!=own:worker.initial.extend(i for i in ids if routes[i].expert//group_experts==g)
    return dict(routes=routes,workers=workers,topk=topk,world=world,b=b,k=k,experts=experts,
                send=send,recv=recv,group=group,capacity=capacity,dedup=dedup,packet_stride=stride)


def replay(plan,emit):
    """One legal causal order, not measured/predicted hardware arrival order.

    Each source worker issues at most one packet per logical round. Each eligible
    receiver consumes one ready snapshot (<=16). New writes become ready NEXT
    round. Prefix counts are assumed immediately available after R initial sends.
    """
    workers=plan['workers'];routes=plan['routes'];ready=set();completed=set()
    stage_counts={};sequence=0;round_no=0
    def event(worker,stage,item=None,target=None,lane=None):
        nonlocal sequence
        sequence+=1;worker.event_index+=1
        target=item if target is None else target
        row=dict(sequence=sequence,logical_round=round_no,rank=worker.rank,core=worker.core,
                 core_sequence=worker.event_index,phase='initial_send' if stage=='INITIAL_SEND' else 'receive',
                 stage=stage,origin_rank=item.source if item else None,token=item.token if item else None,
                 topk_slot=target.slot if target else None,expert=target.expert if target else None,
                 packet_type=target.kind if target else None,route_id=target.id if target else None,
                 parent_route=item.id if stage=='RELAY_ISSUE' else item.root if item and item.kind=='TERMINAL' else None,target_rank=target.destination if target else None,
                 target_core=target.receiver if target else None,output_row=target.row if target else None,
                 lane=lane,tokens=1 if item else 0)
        aggregate=stage_counts.setdefault((worker.rank,worker.core,round_no,stage),
            dict(packets=0,events=0,first_core_sequence=worker.event_index,last_core_sequence=worker.event_index))
        aggregate['packets']+=row['tokens'];aggregate['events']+=1
        aggregate['last_core_sequence']=worker.event_index
        emit(row)
    while any(w.sent<len(w.initial) or w.pending for w in workers.values()):
        progress=False;arrivals=[]
        prefix_ready={rank:all(workers[rank,c].sent==len(workers[rank,c].initial)
                               for c in range(plan['send'],TOTAL_CORES)) for rank in range(plan['world'])}
        just_sent=set()
        for key,w in workers.items():
            if w.sent<len(w.initial):
                item=routes[w.initial[w.sent]];w.sent+=1
                event(w,'INITIAL_SEND',item);arrivals.append(item.id);just_sent.add(key);progress=True
                w.counts['send_packets']+=1;w.counts['send_'+item.kind.lower()]+=1
                w.counts['send_same_rank' if item.destination==w.rank else 'send_cross_rank']+=1
                w.unique['send'].add((item.source,item.token))
        for key,w in workers.items():
            if key in just_sent or w.sent<len(w.initial) or not prefix_ready[w.rank]:continue
            if not w.prefix:event(w,'PREFIX_READY');w.prefix=True
            if not w.pending:continue
            tested=0;batch=[];bits=0
            while tested<len(w.pending):
                if w.scan>=len(w.pending):w.scan=0
                batch=w.pending[w.scan:w.scan+BATCH];tested+=len(batch)
                bits=sum(1<<j for j,i in enumerate(batch) if i in ready)
                if bits:break
                w.scan+=len(batch)
            if not bits:continue
            priority=sum(1<<j for j,i in enumerate(batch) if routes[i].kind=='FIRST_HOP')
            order=priority_order(len(batch),bits,priority);loaded=0
            for current,index in enumerate(order):
                while w.issued-w.consumed<w.lanes and loaded<len(order):
                    item=routes[batch[order[loaded]]]
                    event(w,'RECV_INPUT',item,lane=w.issued%w.lanes)
                    w.issued+=1;loaded+=1
                item=routes[batch[index]]
                if item.id in completed:raise RuntimeError('Duplicate consumption')
                w.counts['receive_packets']+=1;w.counts['receive_'+item.kind.lower()]+=1
                w.unique['receive'].add((item.source,item.token))
                # This branch issues ALL relays before local payload output.
                for member in item.members:
                    target=routes[member];event(w,'RELAY_ISSUE',item,target,w.consumed%w.lanes)
                    arrivals.append(member);w.counts['forward_packets']+=1
                    w.counts['forward_same_rank' if target.destination==w.rank else 'forward_cross_rank']+=1
                    w.unique['forward'].add((item.source,item.token))
                event(w,'RECV_OUTPUT',item,lane=w.consumed%w.lanes)
                completed.add(item.id);ready.remove(item.id);w.consumed+=1;progress=True
            # Snapshot priority must NOT be used to remove swap-with-last slots.
            for index in reversed(range(len(batch))):
                if bits&(1<<index):
                    pos=w.scan+index;w.pending[pos]=w.pending[-1];w.pending.pop()
        if len(set(arrivals))!=len(arrivals) or any(i in ready or i in completed for i in arrivals):
            raise RuntimeError('Duplicate packet publication')
        ready.update(arrivals)
        if not progress:raise RuntimeError('Causal replay stalled; no ready FIRST_HOP/source progress')
        round_no+=1
    if len(completed)!=len(routes) or ready:raise RuntimeError('Incomplete replay')
    sent=sum(w.counts['send_packets'] for w in workers.values())
    relayed=sum(w.counts['forward_packets'] for w in workers.values())
    if sent+relayed!=len(routes):raise RuntimeError('Send/relay accounting mismatch')
    return dict(logical_rounds=round_no,events=sequence,initial_packets=sent,forward_packets=relayed,
                receive_packets=len(completed),logical_routes=len(routes)),stage_counts


def source_defaults():
    root=ROOT/'csrc/kernels/dispatch/low_latency/common'
    def macro(path,name):
        match=re.search(r'^#define\s+'+name+r'\s+(\d+)\s*$',path.read_text(encoding='utf-8'),re.M)
        if not match:raise ValueError('Cannot find source default '+name)
        return int(match[1])
    core=root/'ll_core_config.h'
    return (macro(core,'DEEPEP_LL_SEND_CORES'),macro(core,'DEEPEP_LL_RECV_CORES'),
            macro(root/'ep_memory_server_dedup_protocol.h','DEEPEP_LL_RANKS_PER_GROUP'))


def parse_args(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    def add_cpu_options(parser):
        parser.add_argument('--send-cores',type=int,help='Source S workers; infer R=56-S when R is omitted')
        parser.add_argument('--recv-cores',type=int,help='Source R workers; infer S=56-R when S is omitted')
        parser.add_argument('--ranks-per-group',type=int,default=source_defaults()[2],help='Assumed compiled logical G')
        parser.add_argument('--route-phase',choices=('timed','initial','all'),default='timed',
                            help='Match final ST routes after correctness repeats (default), initial check, or every seed')
    extra=argparse.ArgumentParser(add_help=False)
    add_cpu_options(extra)
    if '--help' in argv or '-h' in argv:
        parser=ST.make_parser();parser.description=__doc__
        add_cpu_options(parser)
        parser.parse_args(argv)
    options,remaining=extra.parse_known_args(argv)
    args=ST.parse_args(remaining)
    s,r,_=source_defaults()
    if options.send_cores is not None:s=options.send_cores;r=TOTAL_CORES-s
    if options.recv_cores is not None:r=options.recv_cores;s=TOTAL_CORES-r if options.send_cores is None else s
    args.send_cores,args.recv_cores,args.ranks_per_group=s,r,options.ranks_per_group
    args.route_phase=options.route_phase
    if min(s,r)<=0 or s+r!=TOTAL_CORES:raise ValueError('S/R must be positive and S+R=56')
    if args.num_processes<1 or args.num_nodes<1 or not 0<=args.node_rank<args.num_nodes:
        raise ValueError('Invalid num-processes/num-nodes/node-rank')
    if args.backend not in (None,'mte'):raise ValueError('This source branch uses MTE, not RDMA')
    return args


def write_csv(path,rows,fieldnames=None):
    iterator=iter(rows);first=next(iterator,None)
    if first is None and fieldnames is None:return
    with path.open('x',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=fieldnames or list(first))
        writer.writeheader()
        if first is not None:writer.writerow(first)
        writer.writerows(iterator)


COUNT_FIELDS=('send_packets','send_unique_tokens','send_direct','send_first_hop','send_same_rank','send_cross_rank',
              'receive_packets','receive_unique_tokens','receive_direct','receive_first_hop','receive_terminal',
              'forward_packets','forward_unique_tokens','forward_same_rank','forward_cross_rank')


def count_rows(plan,local_processes):
    for (rank,core),w in plan['workers'].items():
        row=dict(rank=rank,node=rank//local_processes,local_rank=rank%local_processes,core=core,
                 initial_role='S_remote_send' if core<plan['send'] else 'R_group_local_send',receive_cores=TOTAL_CORES,
                 receive_lanes=w.lanes)
        for name in COUNT_FIELDS:
            if name.endswith('_unique_tokens'):
                row[name]=len(w.unique[name.split('_')[0]])
            else:row[name]=w.counts[name]
        yield row


def validate_source_contract():
    base=ROOT/'csrc/kernels/dispatch/low_latency'
    sender=(base/'device/dispatch_server_dedup_send.h').read_text(encoding='utf-8')
    receiver=(base/'device/dispatch_server_dedup_recv.h').read_text(encoding='utf-8')
    if ('while ((mask & (uint32_t(1) << firstK)) == 0U) ++firstK;' not in sender or
            'LlMakeReceivePriority' not in receiver or 'tasks_.PendingCount()' not in receiver):
        raise ValueError('CPU model requires earliest-topk FIRST_HOP and static relay-priority receive; inspect changed source')


def provenance():
    paths=['tests/elastic/test_ep_low_latency.py','tests/elastic/inspect_dispatch_routes.py',
           'csrc/kernels/dispatch/low_latency/common/ll_core_config.h',
           'csrc/kernels/dispatch/low_latency/common/ep_memory_server_dedup_protocol.h',
           'csrc/kernels/dispatch/low_latency/common/ll_group_local_send.h',
           'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_receive_tasks.h',
           'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h',
           'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_recv.h',
           'csrc/kernels/dispatch/low_latency/common/ll_receive_priority.h',
           'csrc/kernels/dispatch/low_latency/common/ll_send_order.h',
           'csrc/kernels/common/ep_server_order.h']
    hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
    try:
        top=subprocess.run(['git','-C',str(ROOT),'rev-parse','--show-toplevel'],capture_output=True,text=True,check=True).stdout.strip()
        revision=(subprocess.run(['git','-C',str(ROOT),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
                  if Path(top).resolve()==ROOT.resolve() else None)
    except (OSError,subprocess.SubprocessError):revision=None
    return dict(revision=revision,source_sha256=hashes)


def main(argv=None):
    args=parse_args(argv);validate_source_contract();world=args.num_processes*args.num_nodes
    cases=list(ST.matrix_cases(args))
    for case in cases:ST.validate_args(case,world)
    if not 1<=args.ranks_per_group<=16 or world%args.ranks_per_group:
        raise ValueError('G must be in 1..16 and divide EP')
    expanded=[]
    for case_id,case in enumerate(cases):
        repeats=1 if case.skip_check else case.correctness_repeats
        if repeats<1:raise ValueError('correctness-repeats must be positive')
        offsets=range(repeats) if args.route_phase=='all' else (0 if args.route_phase=='initial' else repeats-1,)
        expanded.extend((case_id,case,offset,'timed' if offset==repeats-1 and args.route_phase!='initial' else 'correctness_'+str(offset))
                        for offset in offsets)
    name=args.run_name or 'cpu_routes_'+datetime.now().strftime('%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]
    if name in ('.','..') or Path(name).name!=name or any(c in name for c in ('/','\\',':')):raise ValueError('run-name must be one directory name')
    output=Path(args.output_dir)/name/'cpu-dispatch-routes'
    # Never overwrite an earlier CPU report or original NPU result files.
    output.mkdir(parents=True,exist_ok=False)
    manifest=dict(schema=1,mode='cpu_only',world_size=world,num_nodes=args.num_nodes,local_processes=args.num_processes,
        send_cores=args.send_cores,recv_cores=args.recv_cores,dispatch_receive_cores=TOTAL_CORES,
        group=args.ranks_per_group,seed=args.seed,source=provenance(),arguments=vars(args),cases=[],
        ordering='A deterministic legal causal replay; NOT measured/predicted arrival order or time. '
                 'One initial packet/core/round, one ready snapshot/receiver/round; publications ready next round. '
                 'R-prefix preparation is instantaneous once all local R initial sends finish. '
                 'No global S-send-complete barrier. Timer/perf/network options are metadata only. '
                 'Counts are per ONE dispatch, not multiplied by warmup/iters/timer-iters.',
        forbidden_modules_loaded=[n for n in ('torch','torch_npu','deep_ep') if n in sys.modules])
    sections=['# CPU dispatch 路由与分核统计','',manifest['ordering'],'',
              'packet数按一次投递计数，同一原始token可计入多个packet；unique_tokens按(origin_rank, token)去重。',
              'S/R仅表示初发分工，全部56核静态分片接收。轮次是模型步，不是ticks或耗时。','']
    for index,(st_case_id,case,offset,phase) in enumerate(expanded):
        effective_seed=case.seed+offset
        m=case.num_max_tokens_per_rank if case.num_max_tokens_per_rank is not None else case.num_tokens
        topk=ST.make_routes(world,case.num_tokens,case.num_topk,case.num_experts,effective_seed,case.route)
        plan=build_plan(topk,case.num_experts,args.send_cores,args.recv_cores,args.ranks_per_group,m,case.hidden,case.dtype)
        directory=output/f'case_{index:04d}';directory.mkdir()
        (directory/'topk_ids.json').write_text(json.dumps(dict(seed=effective_seed,base_seed=case.seed,route_phase=phase,shape=[world,case.num_tokens,case.num_topk],
            topk_ids=topk),indent=2),encoding='utf-8')
        event_fields=('sequence','logical_round','rank','core','core_sequence','phase','stage','origin_rank','token',
                      'topk_slot','expert','packet_type','route_id','parent_route','target_rank','target_core','output_row','lane','tokens')
        with (directory/'events.csv').open('x',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fieldnames=event_fields);writer.writeheader()
            totals,stages=replay(plan,writer.writerow)
        ordered_stages=sorted(stages.items(),key=lambda kv:(kv[0][0],kv[0][1],kv[0][2],kv[1]['first_core_sequence']))
        write_csv(directory/'stage_counts.csv',(dict(rank=r,core=c,logical_round=t,phase='initial_send' if s=='INITIAL_SEND' else 'receive',stage=s,**value)
                    for (r,c,t,s),value in ordered_stages))
        rows=list(count_rows(plan,args.num_processes));write_csv(directory/'core_counts.csv',rows)
        ranks=[]
        for rank in range(world):
            rank_rows=[row for row in rows if row['rank']==rank]
            ranks.append(dict(rank=rank,node=rank//args.num_processes,local_rank=rank%args.num_processes,
                              **{key:(len(set().union(*(plan['workers'][rank,c].unique[key.split('_')[0]] for c in range(TOTAL_CORES))))
                                      if key.endswith('_unique_tokens') else sum(row[key] for row in rank_rows))
                                 for key in COUNT_FIELDS}))
        write_csv(directory/'rank_counts.csv',ranks)
        local=case.num_experts//world
        def route_rows():
            for item in plan['routes']:
                root=plan['routes'][item.root]
                terminal=item.kind=='TERMINAL'
                yield dict(route_id=item.id,initial_route_id=root.id,origin_rank=item.source,token=item.token,
                    source_group=item.source//args.ranks_per_group,topk_slot=item.slot,expert=item.expert,
                    target_rank=item.destination,target_group=item.destination//args.ranks_per_group,
                    q=item.q,output_row=item.row,receive_core=item.receiver,packet_type=item.kind,
                    first_hop_slot=root.slot if root.kind=='FIRST_HOP' else None,
                    first_hop_rank=root.destination if root.kind=='FIRST_HOP' else None,
                    initial_sender_rank=item.source,initial_sender_core=item.initial_core,
                    wire_sender_rank=root.destination if terminal else item.source,
                    wire_sender_core=root.receiver if terminal else item.initial_core,
                    slot_index=((item.source+item.destination)%world*local+item.expert%local)*case.num_tokens+item.q)
        write_csv(directory/'routes.csv',route_rows())
        hits=Counter((item.source,item.destination) for item in plan['routes'])
        initial=Counter((item.source,item.destination) for item in plan['routes'] if item.kind!='TERMINAL')
        relays=Counter((item.source,item.destination) for item in plan['routes'] if item.kind=='TERMINAL')
        write_csv(directory/'routing_counts.csv',(dict(origin_rank=src,target_rank=dst,
            target_group=dst//args.ranks_per_group,logical_routes=n,
            initial_packets=initial[src,dst],relay_packets=relays[src,dst]) for (src,dst),n in sorted(hits.items())))
        expert_counts=Counter(item.expert for item in plan['routes'])
        write_csv(directory/'expert_counts.csv',(dict(expert=e,target_rank=e//local,receive_packets=expert_counts[e])
                  for e in range(case.num_experts)))
        record=dict(case_id=index,st_case_id=st_case_id,effective_seed=effective_seed,route_phase=phase,packet_stride=plan['packet_stride'],config=ST.case_config(case,m),dedup=plan['dedup'],route=case.route,**totals)
        manifest['cases'].append(record)
        sections.extend([f'## Case {index}: seed={effective_seed}, {phase}, {record["config"]}', '',str(totals),'',
            '| rank | core | 初始角色 | 初发packet | 接收packet | 转发packet |', '|---|---|---|---|---|---|'])
        sections.extend(f'| {r["rank"]} | {r["core"]} | {r["initial_role"]} | {r["send_packets"]} | {r["receive_packets"]} | {r["forward_packets"]} |' for r in rows)
        sections.append('')
        print(f'CPU case {index}: seed={effective_seed} phase={phase} EP={world} S/R={args.send_cores}/{args.recv_cores} {totals}')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (output/'summary.md').write_text('\n'.join(sections),encoding='utf-8')
    print(f'CPU reports: {output}')
    return output


if __name__=='__main__':
    main()
