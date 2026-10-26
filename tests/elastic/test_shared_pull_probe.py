"""Prototype bytes/lifetime checks; CPU models are not device validation."""
from pathlib import Path
import shutil
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[2]
RECV=ROOT/'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_recv.h'


def method(source,name):
    start=source.index('__aicore__ inline void '+name+'(')
    first=source.index('{',start);last=first+1;depth=1
    while depth:
        depth+=(source[last]=='{')-(source[last]=='}');last+=1
    return source[start:last]


@pytest.mark.parametrize('mode',(0,1,2))
def test_real_descriptor_publish_and_payload_redirect(tmp_path,mode):
    cc=shutil.which('g++') or shutil.which('clang++')
    if not cc:pytest.skip('CPU C++ compiler unavailable')
    source=RECV.read_text(encoding='utf-8')
    methods=method(source,'IssueLoad')+'\n'+method(source,'PublishSharedReference')
    cpp=Path(__file__).with_name('shared_pull_probe_harness.cpp').read_text(encoding='utf-8')
    cpp=cpp.replace('// ACTUAL_RECEIVER_METHODS',methods)
    fence=(ROOT/'csrc/kernels/dispatch/low_latency/device/ll_shared_pull_fence.h').read_text()
    cpp=cpp.replace('// ACTUAL_FENCE_METHOD',method(fence,'LlSharedPullFinish'))
    packet=(ROOT/'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_receive_packet.h').read_text()
    cpp=cpp.replace('// ACTUAL_METADATA_METHOD',method(packet,'BuildServerDedupMetadata'))
    file=tmp_path/'pull.cpp';file.write_text(cpp,encoding='utf-8')
    exe=tmp_path/'pull.exe'
    result=subprocess.run([cc,'-std=c++17','-O2','-UNDEBUG',f'-DDEEPEP_LL_SHARED_PULL_PROBE_MODE={mode}',
        '-I',str(ROOT),str(file),'-o',str(exe)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    result=subprocess.run([str(exe)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr


def test_end_fence_uses_monotonic_local_completion_not_parity_or_early_flag():
    source=(ROOT/'csrc/kernels/dispatch/low_latency/device/ll_shared_pull_fence.h').read_text()
    assert source.index('SyncAll<true>()')<source.index('GetBlockIdx()!=0U')<source.index('previous+1U')
    assert source.index('DataCopy(counter,state')<source.index('state.GetValue(0U)<epoch')
    assert 'previous==0xffffffffU' in source and source.count('CACHE_MODE_DISABLE')==2
    # Independently simulate arbitrary rank completion/publication order. A
    # rank can finish t only if all ranks' reads(t) have completed. Monotonic
    # counters tolerate observing t+1 while another rank is still leaving t.
    import random
    for seed in range(20):
        rng=random.Random(seed);published=[0]*16;exited=[0]*16
        for _ in range(20000):
            r=rng.randrange(16)
            if published[r]==exited[r] and exited[r]<100:
                published[r]+=1 # publication is after all local reads finish
            if published[r]>exited[r] and min(published)>=published[r]:
                exited[r]=published[r]
            for rank in range(16):
                assert exited[rank]<=min(published)
            if min(exited)==100:break
        assert min(exited)==100


def test_candidate_preserves_payload_and_output_handle_and_rejects_old_diagnostics():
    source=RECV.read_text(encoding='utf-8')
    pub=method(source,'PublishSharedReference')
    assert 'DataCopy(output, wire' not in pub
    assert pub.count('ServerDedupReceiveSync<HardEvent::MTE3_S>()')==2
    assert 'wire[uint32_t(data_.tokenStride)+32U]' in pub
    assert 'type.SetValue(0U,oldType)' in pub
    assert source.index('if(pulledTerminal_[lane])')<source.index('const uint32_t role =')
    assert 'header.SetValue(3U,uint32_t(EpServerDedupTokenType::Terminal))' in source
    host=(ROOT/'csrc/ops/low_latency.cpp').read_text(encoding='utf-8')
    assert 'Shared-pull prototype supports OFF performance only' in host
    assert 'p!=16 || b!=256 || h!=3584 || k!=8 || e!=384 || !fp8' in host
