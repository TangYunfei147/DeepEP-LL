"""Shared source/receive packet detail and bounded trace coverage."""
import pytest
from tests.ll_reference_profile.test_parse import sample,timer,getword,setword


def with_log():
    raw=sample()['raw'];core=0
    setword(raw,core,96,2);setword(raw,core,97,2);setword(raw,core,107,0)
    fields=((1,1,2,0,0,0,2|(1<<8),0),(2,2,1,1,0,0,3,24))
    for j,entry in enumerate(fields):
        for i in range(4):setword(raw,core,128+4*j+i,entry[i*2]|(entry[i*2+1]<<32))
    return raw


def test_same_core_has_source_counts_and_reordered_receiver_trace():
    raw=with_log();core=timer.parse_records(raw)['cores'][0];detail=core['packet_detail']
    assert detail['kind']=='shared' and detail['sender']['packet_count']==1
    receive=detail['receiver'];assert receive['recorded_tasks']==2 and not receive['truncated']
    assert [row['process_order'] for row in receive['entries']]==[2,1]
    assert [row['source_rank'] for row in receive['entries']]==[0,1]
    assert [row['packet_role'] for row in receive['entries']]==['FIRST_HOP','TERMINAL']
    assert receive['entries'][0]['forward_members']==1
    assert receive['entries'][1]['destination_expert']==0


def test_zero_capacity_is_explicitly_truncated_not_missing_data():
    parsed=timer.parse_records(sample()['raw'])
    core=parsed['cores'][0]['packet_detail']['receiver']
    assert core['total_tasks']==2 and core['recorded_tasks']==0 and core['truncated']
    assert core['entries']==[] and core['ready_batches']==2
    empty=parsed['cores'][1]['packet_detail']['receiver']
    assert empty['total_tasks']==0 and empty['recorded_tasks']==0 and not empty['truncated']


def test_same_group_time_only_present_when_group_packet_sent():
    parsed=timer.parse_records(sample()['raw'])
    assert parsed['cores'][0]['packet_detail']['sender']['durations']=={}
    group=parsed['cores'][30]['packet_detail']['sender']
    assert group['group_packets']==1 and group['durations']['group_submit_span']==.004
    assert all('second_put_call' not in core['packet_detail']['sender']['durations'] for core in parsed['cores'])


def test_no_receiver_execution_has_no_fake_trace():
    raw=sample(receiver=False)['raw'];parsed=timer.parse_records(raw)
    assert all(core['packet_detail']['receiver'] is None and core['task_count'] is None for core in parsed['cores'])
    setword(raw,0,105,1)
    with pytest.raises(ValueError,match='Receiver detail'):timer.parse_records(raw)


@pytest.mark.parametrize('word,value',[(65,1),(71,1),(80,1),(90,1),(109,1),(66,2),(89,9),
    (96,1),(97,129),(99,1),(100,1),(101,0),(105,2),(106,0),(107,1),(108,3),(136,1)])
def test_bad_source_or_trace_metadata_rejected(word,value):
    raw=with_log();setword(raw,0,word,value)
    with pytest.raises(ValueError):timer.parse_records(raw)


def test_trace_export_cannot_precede_business_completion():
    raw=with_log();setword(raw,0,102,getword(raw,0,21)-1)
    with pytest.raises(ValueError,match='before business'):timer.parse_records(raw)


def test_nonempty_source_route_partition_can_have_no_remote_packet():
    # S may own input tokens whose destinations are all sent locally by R.
    raw=sample()['raw'];setword(raw,0,88,0)
    parsed=timer.parse_records(raw)
    assert parsed['cores'][0]['source_task_count']==1
    assert parsed['cores'][0]['packet_detail']['sender']['packet_count']==0


@pytest.mark.parametrize('field,value',[(0,0),(1,3),(2,0),(3,16),(4,1),(5,32),(6,4),(7,48)])
def test_bad_packet_identity_rejected(field,value):
    raw=with_log();index=128+field//2;old=getword(raw,0,index)
    masked=(old&0xffffffff00000000)|value if field%2==0 else (old&0xffffffff)|(value<<32)
    setword(raw,0,index,masked)
    with pytest.raises(ValueError):timer.parse_records(raw)


def test_duplicate_ready_or_process_order_rejected():
    raw=with_log();setword(raw,0,132,(getword(raw,0,132)&0xffffffff00000000)|1)
    with pytest.raises(ValueError,match='Duplicate'):timer.parse_records(raw)
