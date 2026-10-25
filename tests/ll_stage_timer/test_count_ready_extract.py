"""Execute the production count VF and actual polling body with CPU API mocks."""
from pathlib import Path
import os
import shlex
import shutil
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[2]
DEVICE=ROOT/'csrc/kernels/dispatch/low_latency/device'
FIXTURES=Path(__file__).with_name('fixtures')


def compiler():
    configured=shlex.split(os.environ.get('CXX',''))
    if configured:return configured
    found=shutil.which('g++') or shutil.which('clang++')
    if not found:pytest.skip('Native compiler required for production count VF')
    return [found]


def build_and_run(tmp_path,fixture):
    exe=tmp_path/(fixture+('.exe' if os.name=='nt' else ''))
    subprocess.run([*compiler(),'-std=c++17','-O2','-UNDEBUG',
                    '-I',str(FIXTURES/'count_ready_stub'),'-I',str(tmp_path),
                    '-iquote',str(ROOT),str(FIXTURES/(fixture+'.cpp')),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)


def test_actual_count_ready_extract_vf(tmp_path):
    build_and_run(tmp_path,'count_ready_extract')


def test_actual_p1_polling_body(tmp_path):
    source=(DEVICE/'dispatch_parallel_count_prefix.h').read_text(encoding='utf-8')
    start=source.index('__aicore__ inline bool ServerDedupCounts::TryParallelRecvCnt()')
    end=source.index('__aicore__ inline void ServerDedupCounts::CalcParallelCumsum()',start)
    (tmp_path/'actual_count_poll.h').write_text(source[start:end],encoding='utf-8')
    build_and_run(tmp_path,'count_ready_poll')


def test_fused_count_scope_and_retirement():
    source=(DEVICE/'dispatch_parallel_count_prefix.h').read_text(encoding='utf-8')
    start=source.index('__aicore__ inline bool ServerDedupCounts::TryParallelRecvCnt()')
    body=source[start:source.index('__aicore__ inline void ServerDedupCounts::CalcParallelCumsum()',start)]
    assert 'if(epWorldSize_==16U && LlSmallBPrefixProducerCount(epWorldSize_)==1U)' in body
    begin=body.index('asc_vf_call<P1CountReadyAndExtract>')
    assert body.index('SyncFunc<HardEvent::MTE2_V>()')<begin
    assert begin<body.index('SyncFunc<HardEvent::V_S>()',begin)<body.index('if(!ready)return false;')
    assert body.index('if(!ready)return false;')<body.index('WaitDispatchClearStatus()')
    assert body.count('asc_vf_call<P1CountReadyAndExtract>')==1
    assert 'InitBuffer' not in body # Existing 96B scalar/gather scratch only.
    main=(DEVICE/'dispatch_server_dedup_counts.h').read_text(encoding='utf-8')
    assert 'return LlUseSmallDirect(axisBS_,buildLocalTasks_);' in main
