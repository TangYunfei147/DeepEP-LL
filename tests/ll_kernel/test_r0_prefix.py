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
    assert build.index('DataCopyPad(workspace,prefix,copy)')<build.index('SyncFunc<HardEvent::MTE3_S>()')<build.index('DataCopy(prefixReadyGm_')
    assert 'GetCumSum(' not in source and 'PublishRecvCntSum(' not in source
    assert 'DataCopyPad(prefix, workspace, copy, pad)' in source
    assert 'CACHE_MODE_DISABLE' in source
    receive=(ROOT/'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_recv.h').read_text(encoding='utf-8')
    assert 'uint64_t(receiver_) * counts.prefixStride' not in receive
    # Incoming count total can exceed 255: no truncated strided reduction repeat.
    assert 'ReduceSum(sum,gather,sumContinueBuf_.Get<float>(),rscvStatusNum_)' in source
    assert 'Duplicate' not in source[source.index('ServerDedupCounts::WaitCumSumFlag()'):]

def test_epoch_and_full_prefix_shape_changes():
    from itertools import accumulate
    import random
    rng=random.Random(71)
    epoch=0
    for experts in (16,384,1024,32,384):
        next_epoch=1-epoch
        records=[rng.randrange(257) for _ in range(experts)]
        arrived=[False]*experts
        order=list(range(experts));rng.shuffle(order)
        for n,i in enumerate(order):
            arrived[i]=True
            if n+1<experts:
                assert not all(arrived) and epoch!=next_epoch
        prefix=list(accumulate(records))
        assert all(prefix[i]-(prefix[i-1] if i else 0)==records[i] for i in range(experts))
        # Publication release follows the full table; no per-consumer flag clear.
        epoch=next_epoch
        for _ in range(56):
            assert epoch==next_epoch
            assert prefix[-1]==sum(records)
