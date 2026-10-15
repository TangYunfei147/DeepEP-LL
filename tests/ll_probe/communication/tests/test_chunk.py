import pytest
from tests.ll_probe.profiling.profile_probe import parser, plan


def make(extra):
    args=parser().parse_args(['--kind','comm','--endpoint','tcp://127.0.0.1:1','--output-dir','results/chunk-plan',*extra])
    return plan(args,{'WORLD_SIZE':'16','RANK':'0','LOCAL_RANK':'0'})


@pytest.mark.parametrize('lanes,chunk',[(1,253952),(2,126976),(3,84640),(4,63488),(4,32),(4,32768)])
def test_chunk_forwarding(lanes,chunk):
    job=make(['--lanes',str(lanes),'--chunk',str(chunk)])
    assert job['argv'][-1]==str(chunk)
    assert job['argv'][7]==str(lanes)


@pytest.mark.parametrize('chunk',[0,-32,33,63520,253984,4294967296])
def test_invalid_chunk(chunk):
    with pytest.raises(ValueError):make(['--chunk',str(chunk)])


def test_ndr_cannot_silently_ignore_chunk():
    with pytest.raises(ValueError,match='only to ubmem'):make(['--transport','ndr','--chunk','32768'])
    assert make(['--transport','ndr'])['argv'][-1]=='4096'
