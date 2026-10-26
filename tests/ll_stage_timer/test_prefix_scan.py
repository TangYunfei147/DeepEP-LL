"""Production VF integer scan executed with checked CPU register operations."""
from pathlib import Path
import shutil
import subprocess
import pytest
ROOT=Path(__file__).resolve().parents[2]

def test_actual_integer_prefix_vf(tmp_path):
    cc=shutil.which('g++') or shutil.which('clang++')
    if not cc:pytest.skip('Native compiler needed for VF semantic mock')
    fixtures=Path(__file__).parent/'fixtures'
    exe=tmp_path/('scan.exe' if __import__('os').name=='nt' else 'scan')
    subprocess.run([cc,'-std=c++17','-O2','-UNDEBUG','-I',str(fixtures/'prefix_stub'),
                    '-iquote',str(ROOT),str(fixtures/'prefix_scan.cpp'),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)

def test_scan_storage_and_publication():
    base=ROOT/'csrc/kernels/dispatch/low_latency'
    source=(base/'device/dispatch_server_dedup_counts.h').read_text(encoding='utf-8')
    body=source[source.index('ServerDedupCounts::CalRecvAndSetFlag()'):source.index('ServerDedupCounts::WaitCumSumFlag()')]
    assert 'prefix.SetValue' not in body and 'prefix.GetValue' not in body
    assert body.index('asc_vf_call<R0PrefixScan>')<body.index('SyncFunc<HardEvent::V_MTE3>()')<body.index('DataCopyPad(workspace')
    assert body.index('SyncFunc<HardEvent::MTE3_S>()')<body.index('PublishPrefixReady();')
    assert 'Ceil(rscvStatusNum_*4U,256U)*256U' in source
    header=(base/'device/dispatch_prefix_scan.h').read_text(encoding='utf-8')
    assert 'float' not in header and 'GetValue' not in header
    for step in (1,2,4,8,16,32):assert f'PrefixScanStep<{step}U>' in header
