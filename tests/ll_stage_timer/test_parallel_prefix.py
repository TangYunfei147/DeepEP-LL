from pathlib import Path
import os
import shutil
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[2]

def test_actual_scan_partition_protocol_and_budget(tmp_path):
    cc=shutil.which('g++') or shutil.which('clang++')
    if not cc:pytest.skip('C++ compiler required for production scan/control fixture')
    f=Path(__file__).with_name('fixtures')
    exe=tmp_path/('parallel.exe' if os.name=='nt' else 'parallel')
    subprocess.run([cc,'-std=c++17','-O2','-UNDEBUG','-I',str(f/'prefix_stub'),'-iquote',str(ROOT),
        str(f/'parallel_prefix.cpp'),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)

def test_guard_generation_publication_and_lifetime():
    base=ROOT/'csrc/kernels/dispatch/low_latency'
    main=(base/'device/dispatch_server_dedup_counts.h').read_text(encoding='utf-8')
    body=(base/'device/dispatch_parallel_count_prefix.h').read_text(encoding='utf-8')
    assert 'return LlUseSmallDirect(axisBS_,buildLocalTasks_);' in main
    assert 'if(UseParallelPrefix()){PrepareParallelRecvCnt();return;}' in main
    assert 'if(UseParallelPrefix()){CompleteParallelCumsum();return;}' in main
    assert 'if(UseParallelPrefix()){CalcParallelCumsum();return;}' in main
    assert 'prefixGeneration_=LlNextPrefixGeneration(publication.GetValue(1U));' in body
    assert body.index('DataCopyPad(workspace[startStatusIndex_]')<body.index('SyncFunc<HardEvent::MTE3_S>()')<body.index('LlPrefixReadyControlByte(aivId_)')
    assert 'ready.SetValue(0U,prefixEpoch_)' in body # Keep old epoch current across modes.
    assert 'if(!overlap_.enabled){LL_SEG_SCOPE(TRANSITION_DRAIN);PipeBarrier<PIPE_ALL>();}' in body
    assert 'SyncAll' not in body
    assert 'matrix' not in body


def test_parallel_budget_only_applies_to_receivers():
    header=(ROOT/'csrc/kernels/dispatch/low_latency/common/ll_overlap_layout.h').read_text()
    assert 'const bool parallelPrefix=!enableDedup && bs<=32U && core>=kLlSendCoreNum;' in header
