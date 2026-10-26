"""Execute actual receive task construction with delayed vector stores on CPU."""
from pathlib import Path
import shutil
import subprocess
import pytest
ROOT=Path(__file__).resolve().parents[2]

def test_native_vector_receive_tables(tmp_path):
    cc=shutil.which('g++') or shutil.which('clang++')
    if not cc:pytest.skip('C++ compiler needed for production receive task mock')
    fixtures=Path(__file__).parent/'fixtures'
    exe=tmp_path/('receive.exe' if __import__('os').name=='nt' else 'receive')
    subprocess.run([cc,'-std=c++17','-O2','-I',str(fixtures/'receive_stub'),'-iquote',str(ROOT),
                    str(fixtures/'receive_tasks.cpp'),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)

def test_single_prefix_publication_contract():
    source=(ROOT/'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_counts.h').read_text(encoding='utf-8')
    build=source[source.index('ServerDedupCounts::CalRecvAndSetFlag()'):source.index('ServerDedupCounts::WaitCumSumFlag()')]
    assert 'if (aivId_!=kLlSendCoreNum) return;' in build
    assert build.count('DataCopyPad(workspace,prefix,copy)')==1
    assert build.index('DataCopyPad(workspace,prefix,copy)')<build.index('SyncFunc<HardEvent::MTE3_S>()')<build.index('PublishPrefixReady();')
    assert 'GetCumSum(' not in source and 'PublishRecvCntSum(' not in source
    assert 'DataCopyPad(prefix, workspace, copy, pad)' in source
    assert 'CACHE_MODE_DISABLE' in source
    receive=(ROOT/'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_recv.h').read_text(encoding='utf-8')
    assert 'uint64_t(receiver_) * counts.prefixStride' not in receive
    # Incoming count total can exceed 255: no truncated strided reduction repeat.
    assert 'ReduceSum(sum,gather,sumContinueBuf_.Get<float>(),rscvStatusNum_)' in source
    assert 'prefixEpoch_' not in source
    complete=source[source.index('ServerDedupCounts::CompleteCumsum()'):source.index('ServerDedupCounts::CalTokenSendExpertCnt')]
    assert complete.index('DataCopyPad(prefix, workspace, copy, pad)')<complete.index('SyncFunc<HardEvent::MTE2_S>()')<complete.index('ClearPrefixReady();')
    assert 'uint64_t(aivId_)*kEpServerDedupMagicStride+32U' in source
