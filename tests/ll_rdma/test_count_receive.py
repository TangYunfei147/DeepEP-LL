from pathlib import Path
import os
import subprocess
import pytest
ROOT=Path(__file__).resolve().parents[2]

def test_real_strided_reader(tmp_path):
    zig=os.environ.get('DEEPEP_TEST_ZIG')
    if not zig: pytest.skip('Set DEEPEP_TEST_ZIG for C++ CPU validation')
    exe=tmp_path/'receive.exe'
    subprocess.run([zig,'c++','-std=c++17','-O1','-UNDEBUG',
        '-I'+str(ROOT/'tests/ll_rdma/stubs'),'-I'+str(ROOT/'csrc'),
        str(Path(__file__).with_name('count_receive_test.cpp')),'-o',str(exe)],check=True,timeout=180)
    result=subprocess.run([str(exe)],check=True,capture_output=True,text=True,timeout=60)
    assert 'PASS' in result.stdout
    print(result.stdout)

def test_read_plan_outside_poll_and_body_after_commit():
    source=(ROOT/'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_counts.h').read_text(encoding='utf-8')
    start=source.index('void ServerDedupCounts::WaitDispatch()')
    body=source[start:source.index('void ServerDedupCounts::CalRecvAndSetFlag()',start)]
    assert body.index('BuildCountReadPlan') < body.index('while (sumOfFlag')
    poll=body[body.index('while (sumOfFlag'):body.index('// Commit acquired')]
    assert 'CountFrame(' not in poll
    assert 'readPlan, false' in poll and 'readPlan, true' not in poll
    assert 'readPlan, true' in body
    assert 'if (!packedFrames_) WaitDispatchClearStatus();' in body
