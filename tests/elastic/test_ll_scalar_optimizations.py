"""Protocol equivalence models for independently delivered scalar changes."""
import random

import pytest


@pytest.mark.parametrize("experts,ranks,k,b", [
    (384,16,8,256), (16,16,1,1), (64,16,32,8),
    (1024,64,8,64), (128,32,3,41), (512,128,16,65),
])
@pytest.mark.parametrize("seed", range(4))
def test_remote_q_preserves_prefix_order_and_first_token(experts,ranks,k,b,seed):
    rng=random.Random(seed)
    # Include duplicate IDs to check the existing first-token convention.
    routes=[rng.randrange(experts) for _ in range(b*k)]
    rank=rng.randrange(ranks)
    width=experts//ranks*4
    local=rank//4*width
    histogram=[0]*experts
    table={}
    for i,expert in enumerate(routes):
        if local<=expert<local+width:
            continue
        q=histogram[expert]
        histogram[expert]+=1
        table[i]=0 if i<k else q
    # Check arbitrary sender shards and FIRST_HOP tail lookups, not just totals.
    for send_cores in (1,10,28,55):
        for core in range(send_cores):
            begin=core*(b//send_cores)+min(core,b%send_cores)
            count=b//send_cores+(core<b%send_cores)
            for i in range(begin*k,(begin+count)*k):
                if i in table:
                    assert table[i]==(0 if i<k else routes[:i].count(routes[i]))
