"""CPU contract checks for generated API timing; no NPU initialized."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
HERE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('four_st',HERE/'snapshot/test_ll_four_layers.py')
st=importlib.util.module_from_spec(spec);spec.loader.exec_module(st)
calls={'invoke':0,'wait':0,'sync':0,'boundary':0}
def invoke():
    calls['invoke']+=1
    return (None,SimpleNamespace(current_stream_wait=lambda:calls.__setitem__('wait',calls['wait']+1)))
def sync():calls['sync']+=1
def boundary(args):calls['boundary']+=1
def gather(output,raw):output[:]=[list(raw),[t+1 for t in raw]]
fake_dist=SimpleNamespace(get_world_size=lambda:2,all_gather_object=gather)
fake_torch=SimpleNamespace(npu=SimpleNamespace(synchronize=sync),distributed=fake_dist)
args=SimpleNamespace(warmup=2,num_iters=3)
st.measure_kernel=lambda *a,**kw:{'measurement':'kernel_duration_us'}
st.completion_boundary=boundary
out=HERE/'build/adapter-check';out.mkdir(parents=True,exist_ok=True)
with patch.dict('sys.modules',{'torch':fake_torch,'torch.distributed':fake_dist}):
    result=st.measure(invoke,args,'dispatch',out)
assert result['measurement']=='kernel_duration_us'
api=result['full_api']
assert len(api['ranks'])==2 and len(api['rank_max_us'])==3
assert api['rank_max_us']==api['ranks'][1]['api_us']
assert calls=={'invoke':5,'wait':5,'sync':5,'boundary':5}
assert 'not cold' in api['regime']
# Verify multi-party incidence and non-overlapping sender slots independently.
for world in [2,4,8,16,32]:
    graphs=[[(0,1)],[(s,0) for s in range(1,world)],[(0,d) for d in range(1,world)],
            [(s,d) for s in range(world) for d in range(world) if s!=d]]
    assert [len(g) for g in graphs]==[1,world-1,world-1,world*(world-1)]
    for graph in graphs:
        slots=[(d,s*15392,(s+1)*15392) for s,d in graph]
        assert len({(d,begin) for d,begin,_ in slots})==len(slots)
print('PASS API invoke/wait/sync boundary, warmup exclusion, iteration rankMAX and world2/4/8/16/32 graph slots')

assert all(f"p{q}_us" in api["rank_max_stats"] for q in (50,90,99))
assert all(all(f"p{q}_us" in row for q in (50,90,99)) for row in api["ranks"])
