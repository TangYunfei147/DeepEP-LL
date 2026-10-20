"""Fixed-total split validation, role decoding and partition models; no NPU use."""
import importlib.util
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[2]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
timer=load('split_timer','deep_ep/utils/ll_stage_timer.py')


def raw(send):
    result=[]
    for core in range(56):
        words=[0]*256
        words[:8]=[timer.MAGIC,5,1,56,core,1|(send<<8),1,len(timer.STAGES)]
        words[8:12]=[10,20,10,1]
        result.append([words[n:n+8] for n in range(0,256,8)])
    return result


@pytest.mark.parametrize('send',range(1,56))
def test_all_positive_56_splits_cover_roles_and_notify_targets(send):
    recv=56-send
    report=timer.parse_records(raw(send),'dispatch',1e9)
    assert (report['send_cores'],report['recv_cores'])==(send,recv)
    assert sum(row['role']=='send' for row in report['records'])==send
    assert sum(row['role']=='receive' for row in report['records'])==recv
    assert [core-send for core in range(send,56)]==list(range(recv))
    notify=min(8,recv)
    assert 56-notify>=send
    for group in (4,16):
        owners=[peer for core in range(notify) for peer in range(core,group,notify)]
        assert sorted(owners)==list(range(group))
    for workers in (send,recv):
        for count in (1,32,256,10000):
            covered=[i for core in range(workers) for i in range(core*count//workers,(core+1)*count//workers)]
            assert covered==list(range(count))
    # Dispatch control tables must remain below Combine's fixed 1MiB boundary.
    for experts in (16,384,1024):
        matrix=recv*recv*32
        prefix_stride=(experts*4+31)//32*32
        end=72*512+experts*512+2*matrix+recv*prefix_stride
        reserved=(end+1023)//1024*1024+1024
        assert reserved<=1024*1024


@pytest.mark.parametrize('send',[0,56,59])
def test_invalid_recorded_split_rejected(send):
    with pytest.raises(ValueError,match='split'):
        timer.parse_records(raw(send),'dispatch')


def test_mixed_device_role_boundaries_rejected():
    values=raw(20);values[1][0][5]=1|(21<<8)
    with pytest.raises(ValueError,match='Mixed core splits'):
        timer.parse_records(values,'dispatch')


@pytest.mark.parametrize('send,recv',[('27','28'),('30','28'),('0','56'),('-1','59'),('x','28')])
def test_invalid_build_config_fails_before_toolchain_lookup(monkeypatch,send,recv):
    build=load('split_build','build_support.py')
    monkeypatch.setattr(build.platform,'system',lambda:'Linux')
    with pytest.raises(ValueError):
        build.build_configuration(dict(DEEPEP_NPU_ARCH='Ascend950',DEEPEP_LL_SEND_CORES=send,DEEPEP_LL_RECV_CORES=recv))


def test_split_changes_build_identity_and_wires_both_compilers():
    build=load('split_build','build_support.py')
    assert build.build_cache_key(dict(ll_send_cores=20,ll_recv_cores=36))!=build.build_cache_key(dict(ll_send_cores=28,ll_recv_cores=28))
    host=(ROOT/'cmake/Dependencies.cmake').read_text()
    child=(ROOT/'cmake/Device.cmake').read_text()
    device=(ROOT/'cmake/device/CMakeLists.txt').read_text()
    assert 'CoreSplit.cmake' in host and 'Dependencies.cmake' in device
    for name in ('DEEPEP_LL_SEND_CORES','DEEPEP_LL_RECV_CORES'):
        assert f'-D{name}=${{{name}}}' in child
    receiver=(ROOT/'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_recv.h').read_text()
    assert 'receiver_ = GetBlockIdx() - kLlSendCoreNum;' in receiver
    partial=(ROOT/'csrc/kernels/combine/low_latency/combine_server_dedup_partial.h').read_text()
    assert 'GetBlockIdx() - DispatchDedup::kLlSendCoreNum' in partial


def test_compare_treats_split_as_explicit_experiment_variable():
    parity=load('split_parity','tests/ll_stage_timer/test_parity.py')
    left,right=parity.identity(),parity.identity()
    left.update(active_aiv=56,send_cores=20,recv_cores=36,core_protocol=7020,stage_timer_schema=5)
    right.update(active_aiv=56,send_cores=28,recv_cores=28,core_protocol=7028,stage_timer_schema=5)
    parity.comparison.check_identity(left,right)
    right['recv_cores']=27
    with pytest.raises(ValueError,match='split'):
        parity.comparison.check_identity(left,right)
    right['recv_cores']=28
    def summary(identity,duration):
        return dict(comparison=identity,per_rank_core_stage=[dict(operation='dispatch',rank=0,core=20,
                    stage='kernel',unit='us',mean=duration)])
    row=parity.comparison.compare(summary(left,10),summary(right,8),'stage')[0]
    assert row['status']=='role_changed' and row['candidate_over_baseline'] is None


def test_expert_decoder_uses_supplied_receive_boundary():
    fixture=load('split_expert_fixture','tests/ll_stage_timer/test_experts.py')
    data=fixture.data()
    # Legacy fixture records are on core 8; changing the boundary to 9 makes it a sender.
    with pytest.raises(ValueError,match='sender core'):
        fixture.expert.parse_records(data,rank=0,world=16,experts=16,send_cores=9)
    assert fixture.expert.parse_records(data,rank=0,world=16,experts=16,send_cores=1)['records']


def test_perf_entry_rejects_stale_wheel_without_requesting_timer():
    st=load('split_entry','tests/elastic/test_ep_low_latency.py')
    st.validate_core_split_capability(dict(active_aiv=56,send_cores=20,recv_cores=36,core_protocol=7020))
    for identity in (dict(active_aiv=58,send_cores=10,recv_cores=48,core_protocol=6010),
                     dict(active_aiv=56,send_cores=28,recv_cores=28,core_protocol=5),
                     dict(active_aiv=56,send_cores=20,recv_cores=36,core_protocol=7028)):
        with pytest.raises(RuntimeError,match='56-core'):
            st.validate_core_split_capability(identity)
