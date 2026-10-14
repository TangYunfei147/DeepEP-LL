"""Actual sampling domain is independent of metadata's logical expert partition."""
import pytest
from tests.ll_probe.profiling import profile_probe as probe


@pytest.mark.parametrize('world',[1,2,4,8,16,32])
def test_metadata_local_device_and_logical_partition(world):
    local = min(world,16)-1
    args=probe.parser().parse_args(['--kind','metadata','--partitions','16','--output-dir','results/plan'])
    plan=probe.plan(args,dict(WORLD_SIZE=str(world),RANK=str(world-1),LOCAL_RANK=str(local)))
    assert plan['world']==world
    assert plan['device']==local
    assert plan['argv'][-1]==str(local)
    assert plan['argv'][5]=='16'
    assert plan['kernel']=='four_metadata'


def test_legacy_ipc_is_not_multihost_bandwidth():
    args=probe.parser().parse_args(['--kind','peer','--output-dir','results/plan'])
    with pytest.raises(ValueError):
        probe.plan(args,dict(WORLD_SIZE='32',RANK='31',LOCAL_RANK='15'))


def test_partial_deployment_identity_rejected():
    args=probe.parser().parse_args(['--kind','metadata','--output-dir','results/plan'])
    with pytest.raises(ValueError):
        probe.plan(args,dict(DEEPEP_PROBE_RUN_ID='partial'))
