from pathlib import Path
import os
import subprocess
import pytest
ROOT=Path(__file__).resolve().parents[2]

def test_frame_mapping_and_transport(tmp_path):
    zig=os.environ.get('DEEPEP_TEST_ZIG')
    if not zig: pytest.skip('Set DEEPEP_TEST_ZIG for C++ CPU validation')
    exe=tmp_path/'counts.exe'
    subprocess.run([zig,'c++','-std=c++17','-O1','-UNDEBUG',
        '-I'+str(ROOT/'tests/ll_rdma/stubs'),'-I'+str(ROOT/'csrc'),
        str(Path(__file__).with_name('count_frame_test.cpp')),'-o',str(exe)],check=True,timeout=180)
    run=subprocess.run([str(exe)],check=True,capture_output=True,text=True,timeout=60)
    assert 'PASS packed counts' in run.stdout
    print(run.stdout)

def test_commit_clear_follows_all_receiver_snapshots():
    s=(ROOT/'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_counts.h').read_text(encoding='utf-8')
    assert 'if (!packedFrames_) WaitDispatchClearStatus();' in s
    tail = s[s.index('const uint32_t offset = GetCumSum(newAivId);'):]
    assert tail.index('profile_->Mark<27>()') < tail.index('if (packedFrames_) WaitDispatchClearStatus();')
    assert 'if (expert == frame.firstExpert)' in s
