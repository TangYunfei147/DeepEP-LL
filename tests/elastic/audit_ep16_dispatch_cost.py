"""Logical EP16 dispatch counts only; no NPU, network, or latency model."""
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
def main():
    spec=importlib.util.spec_from_file_location('_audit_routes',ROOT/'tests/elastic/inspect_dispatch_routes.py')
    model=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=model
    spec.loader.exec_module(model)
    topk=model.ST.make_routes(16,256,8,384,2)
    plan=model.build_plan(topk,384,28,28,4,256)
    routes=plan['routes']
    initial=[r for r in routes if r.kind!='TERMINAL']
    result=dict(kind='CPU logical counts; NOT hardware time or network TX',
        EP=16,B=256,M=256,H=3584,K=8,E=384,G=4,S=28,R=28,seed=2,route='normal',
        routes=len(routes),initial=len(initial),
        relay=sum(r.kind=='TERMINAL' for r in routes),
        first_hop=sum(r.kind=='FIRST_HOP' for r in routes),
        direct=sum(r.kind=='DIRECT' for r in routes),
        original_cross_group=sum(r.source//4!=r.destination//4 for r in routes),
        source_cross_group=sum(r.source//4!=r.destination//4 for r in initial),
        relay_same_rank=sum(r.kind=='TERMINAL' and r.destination==routes[r.root].destination for r in routes),
        same_rank_initial=sum(r.source==r.destination for r in initial),
        q_prefix_elements=sum(r.token*8+r.slot for r in routes
            if r.source//4!=r.destination//4 and r.token>0),
        seed_plus_local_entries_per_rank=sum(model.split(256,28,c)[0]*8 for c in range(28))+256*8)
    assert result['initial']+result['relay']==result['routes']==16*256*8
    assert result['source_cross_group']+result['relay']==result['original_cross_group']
    assert result['first_hop']+result['direct']==result['initial']
    print(json.dumps(result,indent=2))
if __name__=='__main__':
    main()
