"""Fixed-total split validation, role decoding and partition models; no NPU use."""
import importlib.util
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[2]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

fixture=load('split_reference_fixture','tests/ll_reference_profile/test_parse.py')
timer=fixture.timer

def raw(send):
    return fixture.sample(send=send)['raw']


@pytest.mark.parametrize('send',range(1,56))
def test_all_positive_56_splits_cover_roles_and_notify_targets(send):
    recv=56-send
    report=timer.parse_records(raw(send),'dispatch',1e9)
    assert (report['send_cores'],report['recv_cores'])==(send,recv)
    assert sum(row['role']=='send' for row in report['cores'])==send
    assert sum(row['role']=='receive' for row in report['cores'])==recv
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
        end=72*512+experts*512+matrix+56*prefix_stride+56*recv*32
        reserved=(end+1023)//1024*1024+1024
        assert reserved<=1024*1024


@pytest.mark.parametrize('send',[0,56,59])
def test_invalid_recorded_split_rejected(send):
    with pytest.raises(ValueError):
        timer.parse_records(raw(send),'dispatch')


def test_mixed_device_role_boundaries_rejected():
    values=raw(20);fixture.setword(values,1,44,21);fixture.setword(values,1,45,35)
    with pytest.raises(ValueError,match='Mixed'):
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
    assert 'receiver_ = enableDedup ? GetBlockIdx()' in receiver
    partial=(ROOT/'csrc/kernels/combine/low_latency/combine_server_dedup_partial.h').read_text()
    assert 'GetBlockIdx() - DispatchDedup::kLlSendCoreNum' in partial






def test_perf_entry_rejects_stale_wheel_without_requesting_timer():
    st=load('split_entry','tests/elastic/test_ep_low_latency.py')
    st.validate_core_split_capability(dict(active_aiv=56,send_cores=20,recv_cores=36,core_protocol=19020))
    for identity in (dict(active_aiv=58,send_cores=10,recv_cores=48,core_protocol=6010),
                     dict(active_aiv=56,send_cores=28,recv_cores=28,core_protocol=5),
                     dict(active_aiv=56,send_cores=20,recv_cores=36,core_protocol=7028)):
        with pytest.raises(RuntimeError,match='56-core'):
            st.validate_core_split_capability(identity)
