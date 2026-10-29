from pathlib import Path
import os
import subprocess
import pytest
ROOT=Path(__file__).resolve().parents[2]

def test_return_transport_with_delayed_consumers(tmp_path):
    zig=os.environ.get('DEEPEP_TEST_ZIG')
    if not zig: pytest.skip('Set DEEPEP_TEST_ZIG for CPU C++ checks')
    exe=tmp_path/'returns.exe'
    subprocess.run([zig,'c++','-std=c++17','-O1','-UNDEBUG','-I'+str(ROOT/'tests/ll_rdma/stubs'),
        '-I'+str(ROOT/'csrc'),str(Path(__file__).with_name('combine_return_batch_test.cpp')),'-o',str(exe)],check=True,timeout=180)
    result=subprocess.run([str(exe)],capture_output=True,text=True,check=True,timeout=90)
    assert 'PASS combine return batching 450 cases' in result.stdout

def test_m4_posts_before_m5_and_kernel_owns_retirement():
    root=ROOT/'csrc/kernels/combine/low_latency'
    partial=(root/'combine_server_dedup_partial.h').read_text(encoding='utf-8')
    process=partial.split('uint32_t Process()',1)[1].split('private:',1)[0]
    assert 'rdmaBatch_->SubmitPending();' in process and 'Flush(' not in process
    kernel=(root/'combine_server_dedup_kernel.h').read_text(encoding='utf-8')
    process=kernel.split('void Process()',1)[1].split('private:',1)[0]
    assert process.index('AggregateSourceTokens();') < process.index('PipeBarrier<PIPE_ALL>();') < process.index('returnBatch_.Flush')
    assert 'workspace_, &returnBatch_' in kernel
    assert 'SendReturnPacket(returnBatch_' in kernel and 'SendReturnPacket(*rdmaBatch_' in partial
