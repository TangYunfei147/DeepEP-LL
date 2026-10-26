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
    methods='\n'.join(method(source,name) for name in
        ('IssueLoad','PrepareSharedReference','PublishSharedReference','RestoreSharedReference'))
    cpp=Path(__file__).with_name('shared_pull_probe_harness.cpp').read_text(encoding='utf-8')
    cpp=cpp.replace('// ACTUAL_RECEIVER_METHODS',methods)
    packet=(ROOT/'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_receive_packet.h').read_text()
    cpp=cpp.replace('// ACTUAL_METADATA_METHOD',method(packet,'BuildServerDedupMetadata'))
    file=tmp_path/'pull.cpp';file.write_text(cpp,encoding='utf-8')
    exe=tmp_path/'pull.exe'
    result=subprocess.run([cc,'-std=c++17','-O2','-UNDEBUG',f'-DDEEPEP_LL_SHARED_PULL_PROBE_MODE={mode}',
        '-I',str(ROOT),str(file),'-o',str(exe)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    result=subprocess.run([str(exe)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr


def test_candidate_preserves_payload_and_output_handle_and_rejects_old_diagnostics():
    source=RECV.read_text(encoding='utf-8')
    pub=method(source,'PublishSharedReference')
    assert 'DataCopy(output, wire' not in pub
    assert pub.count('ServerDedupReceiveSync<HardEvent::MTE3_S>()')==2
    assert 'wire[uint32_t(data_.tokenStride)+32U]' in pub
    assert 'type.SetValue(0U,role)' in method(source,'RestoreSharedReference')
    prepare=method(source,'PrepareSharedReference')
    assert 'Slot(task)' in prepare and 'Slot(task)' not in pub
    assert source.index('if(pulledTerminal_[lane])')<source.index('const uint32_t role =')
    assert 'header.SetValue(3U,uint32_t(EpServerDedupTokenType::Terminal))' in source
    host=(ROOT/'csrc/ops/low_latency.cpp').read_text(encoding='utf-8')
    assert 'Shared-pull prototype supports OFF performance only' in host
    assert 'p!=16 || b!=256 || h!=3584 || k!=8 || e!=384 || !fp8' in host
