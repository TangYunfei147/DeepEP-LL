from pathlib import Path
import os
import shlex
import shutil
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[2]

def compiler():
    cc=shlex.split(os.environ.get('CXX',''))
    if cc:return cc
    found=shutil.which('g++') or shutil.which('clang++')
    if not found:pytest.skip('C++ compiler required for production scan/control fixture')
    return [found]


def test_actual_scan_partition_protocol_and_budget(tmp_path):
    cc=compiler()
    f=Path(__file__).with_name('fixtures')
    outputs=[]
    for send,recv,prefix in [(28,28,0),*((30,26,p) for p in (0,1,4,6,8,26))]:
        exe=tmp_path/(f'parallel_{send}_{prefix}.exe' if os.name=='nt' else f'parallel_{send}_{prefix}')
        subprocess.run([*cc,'-std=c++17','-O2','-UNDEBUG','-I',str(f/'prefix_stub'),'-iquote',str(ROOT),
            f'-DDEEPEP_LL_SEND_CORES={send}',f'-DDEEPEP_LL_RECV_CORES={recv}',
            f'-DDEEPEP_LL_SMALL_B_PREFIX_CORES={prefix}',str(f/'parallel_prefix.cpp'),'-o',str(exe)],check=True)
        result=subprocess.check_output([str(exe)],text=True)
        if send==30:outputs.append(result)
    assert len(set(outputs))==1 # Configurable P cannot affect non-target plans.


@pytest.mark.parametrize('send,recv,prefix',[(55,1,0),(55,1,1),(1,55,55),(28,28,28)])
def test_configured_producer_boundaries(tmp_path,send,recv,prefix):
    cc=compiler()
    source=tmp_path/'config.cpp'
    source.write_text('#include "csrc/kernels/dispatch/low_latency/common/ll_parallel_prefix.h"\n'
        '#include <cassert>\nint main(){using namespace DispatchDedup;'
        f'assert(kLlSmallBPrefixCores=={prefix});'
        f'assert(LlSmallBPrefixProducerCount(16)=={prefix or recv});'
        f'assert(LlSmallBPrefixProducerCount(32)=={recv});}}\n')
    exe=tmp_path/('config.exe' if os.name=='nt' else 'config')
    subprocess.run([*cc,'-std=c++17','-iquote',str(ROOT),f'-DDEEPEP_LL_SEND_CORES={send}',
        f'-DDEEPEP_LL_RECV_CORES={recv}',f'-DDEEPEP_LL_SMALL_B_PREFIX_CORES={prefix}',
        str(source),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)


@pytest.mark.parametrize('prefix',[-1,27,56])
def test_invalid_native_producer_count_rejected(tmp_path,prefix):
    source=tmp_path/'bad.cpp'
    source.write_text('#include "csrc/kernels/dispatch/low_latency/common/ll_small_b_prefix_config.h"\nint main(){}\n')
    result=subprocess.run([*compiler(),'-std=c++17','-iquote',str(ROOT),'-DDEEPEP_LL_SEND_CORES=30',
        '-DDEEPEP_LL_RECV_CORES=26',f'-DDEEPEP_LL_SMALL_B_PREFIX_CORES={prefix}',str(source),
        '-c','-o',str(tmp_path/'bad.o')],capture_output=True,text=True)
    assert result.returncode!=0 and 'prefix producers' in result.stderr

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
    assert 'core-kLlSendCoreNum<prefixProducers ? DispatchParallelPrefixStageUbBytes(e,prefixProducers):32U' in header


def test_consumers_advance_generation_without_becoming_ready_dependencies():
    base=ROOT/'csrc/kernels/dispatch/low_latency'
    body=(base/'device/dispatch_parallel_count_prefix.h').read_text(encoding='utf-8')
    calc=body[body.index('void ServerDedupCounts::CalcParallelCumsum()'):body.index('void ServerDedupCounts::CompleteParallelCumsum()')]
    assert calc.index('DataCopy(sums[uint64_t(receiver)*128U]')<calc.index('if(receiver>=producers)return;')
    assert calc.index('if(receiver>=producers)return;')<calc.index('LlPrefixReadyControlByte(aivId_)')
    complete=body[body.index('void ServerDedupCounts::CompleteParallelCumsum()'):]
    assert 'for(uint32_t p=0;p<producers;++p)' in complete
    assert 'DataCopyParams{uint16_t(producers),1U,15U,0U}' in complete
    assert 'tpipe_->InitBuffer(prefixBuf_,Ceil(rscvStatusNum_*4U,32U)*32U)' in complete
    host=(ROOT/'csrc/ops/low_latency.cpp').read_text(encoding='utf-8')
    assert 'deepep_ll_small_b_prefix_cores()!=DispatchDedup::kLlSmallBPrefixCores' in host
    assert 'module.attr("low_latency_small_b_prefix_cores") = DispatchDedup::kLlSmallBPrefixCores' in host
