from pathlib import Path
import os
import subprocess
import pytest
ROOT=Path(__file__).resolve().parents[2]

def test_isolated_sources_and_cq_merge(tmp_path):
    zig=os.environ.get('DEEPEP_TEST_ZIG')
    if not zig: pytest.skip('Set DEEPEP_TEST_ZIG for CPU C++ checks')
    exe=tmp_path/'retirement.exe'
    subprocess.run([zig,'c++','-std=c++17','-O1','-UNDEBUG','-I'+str(ROOT/'tests/ll_rdma/stubs'),
        '-I'+str(ROOT/'csrc'),str(Path(__file__).with_name('count_retirement_test.cpp')),'-o',str(exe)],check=True,timeout=180)
    result=subprocess.run([str(exe)],capture_output=True,text=True,check=True,timeout=60)
    assert 'PASS isolated count sources' in result.stdout

def test_retirement_after_role_completion_before_final_timestamp():
    code=(ROOT/'csrc/kernels/dispatch/low_latency/dispatch_server_dedup_kernel.h').read_text(encoding='utf-8')
    assert code.index('counts_.Finish();') < code.index('counts_.Retirement()->Pending()') < code.index('profile_->Mark<21>()')
    assert 'receiver_.FinalScratch()' in code and 'counts_.RetireOwnScratch()' in code
    receive=(ROOT/'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_recv.h').read_text(encoding='utf-8')
    assert 'return prefixBuf_.Get<uint8_t>();' in receive
