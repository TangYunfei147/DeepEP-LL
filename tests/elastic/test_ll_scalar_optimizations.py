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


@pytest.mark.parametrize("groups", [4,8,16,32,64])
@pytest.mark.parametrize("tasks", [0,1,7,57,2048])
def test_stable_direct_buckets_match_old_group_scan(groups,tasks):
    rng=random.Random(groups*10000+tasks)
    records=[(rng.randrange(257),rng.randrange(32),rng.randrange(groups*24),i)
             for i in range(tasks)]
    heads=[tasks]*groups
    linked=list(records)
    for task in reversed(range(tasks)):
        token,k,expert,q=records[task]
        group=expert//24
        linked[task]=(token*32+k,heads[group],expert,q)
        heads[group]=task
    for rotation in (0,1,groups-1):
        order=[(rotation+i)%groups for i in range(groups)]
        expected=[row for group in order for row in records if row[2]//24==group]
        actual=[]
        for group in order:
            task=heads[group]
            while task<tasks:
                index,next_task,expert,q=linked[task]
                actual.append((index//32,index%32,expert,q))
                task=next_task
                assert len(actual)<=tasks
        assert actual==expected


def test_non_dedup_links_fit_existing_host_worst_case():
    align=lambda n:(n+31)//32*32
    # Covers zero-token senders that can still own flattened routes.
    for b in (1,2,7,28,29,55,56,257):
        for k in (1,3,8,16,32):
            for senders in range(1,56):
                max_tokens=(b+senders-1)//senders
                max_routes=(b*k+senders-1)//senders
                assert align(max_routes*8)<=align(max_tokens*k*16)


def test_position_cache_resets_between_launches_and_keeps_partial_batch_size():
    cached=None
    loads=0
    # Same begin in a different launch is not a cache hit: tables can change.
    for epoch in range(3):
        cached=None
        positions=[epoch*1000+i for i in range(33)]
        tile=[]
        for begin,count in [(0,16),(0,16),(16,16),(16,16),(32,1),(32,1),(0,16)]:
            if cached!=(begin,count):
                tile=positions[begin:begin+count]
                loads+=1
                cached=(begin,count)
            assert tile==positions[begin:begin+count]
    assert loads==12  # 21 probes, with 9 descriptor loads eliminated.
