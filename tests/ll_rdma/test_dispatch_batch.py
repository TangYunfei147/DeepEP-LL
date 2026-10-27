"""Compile the production sender batch with adversarial asynchronous transports."""
from pathlib import Path
import os
import shutil
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_batch_transport(tmp_path):
    zig = os.environ.get('DEEPEP_TEST_ZIG')
    compiler = shutil.which('g++') or shutil.which('clang++')
    if not compiler and not zig:
        pytest.skip('C++17 compiler required')
    command = [zig, 'c++'] if zig else [compiler]
    binary = tmp_path / 'dispatch-batch.exe'
    subprocess.run(command + ['-std=c++17', '-O1', '-UNDEBUG', '-Wall', '-Wextra',
        '-I'+str(ROOT/'tests/ll_rdma/stubs'), '-I'+str(ROOT/'csrc'),
        str(Path(__file__).with_name('dispatch_batch_test.cpp')), '-o', str(binary)], check=True, timeout=180)
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True, timeout=60)
    assert 'PASS 75 batch cases' in result.stdout


def test_sender_drains_before_metadata_and_preserves_no_comm():
    source = (ROOT/'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h').read_text(encoding='utf-8')
    assert source.index('rdmaBatch_.Flush') < source.index('profile->Mark<40>')
    assert '#ifndef DEEPEP_LL_NO_COMM\n        if (LlTransport::RemoteWrite' in source
    assert 'rdmaBatch_.Enqueue' in source
