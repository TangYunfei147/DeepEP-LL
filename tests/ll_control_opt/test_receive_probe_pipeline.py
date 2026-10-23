"""Actual pair scheduler + production pending table under delayed readiness.

The CPU model checks ownership/progress and bounded submission, not AscendC
instruction support, physical DMA completion or target performance.
"""
from pathlib import Path
import os
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
COMMON = ROOT / 'csrc/kernels/dispatch/low_latency/common'
RECEIVER = ROOT / 'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_recv.h'


@pytest.mark.parametrize('send_cores', (28, 30))
def test_actual_pair_scheduler_and_pending_table(tmp_path, send_cores):
    compiler = shutil.which('g++') or shutil.which('clang++')
    if not compiler:
        pytest.skip('C++ compiler needed for production probe/pending CPU model')
    fixtures = ROOT / 'tests/ll_stage_timer/fixtures'
    binary = tmp_path / ('probe-pair.exe' if os.name == 'nt' else 'probe-pair')
    subprocess.run([compiler, '-std=c++17', '-O2', f'-DDEEPEP_LL_SEND_CORES={send_cores}',
                    f'-DDEEPEP_LL_RECV_CORES={56-send_cores}', '-I', str(fixtures / 'receive_stub'),
                    '-I', str(ROOT), str(Path(__file__).with_name('receive_probe_pipeline.cpp')),
                    '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_device_event_and_spare_storage_contract():
    source = RECEIVER.read_text(encoding='utf-8')
    start = source[source.index('__aicore__ inline void StartProbe(uint32_t start,uint32_t count,uint32_t offset) {',
                                source.index('__aicore__ inline void RemoveProbe(uint32_t start,uint32_t count,uint32_t ready) {')):
                   source.index('__aicore__ inline LlReceiveProbeResult FinishProbe(uint32_t count) {',
                                source.index('__aicore__ inline void RemoveProbe(uint32_t start,uint32_t count,uint32_t ready) {'))]
    finish = source[source.index('// This is the residual scalar join'):source.index('// 摘要：将稳定任务编号')]
    assert start.index('SetFlag<HardEvent::MTE2_V>(6U)') < start.index('WaitFlag<HardEvent::MTE2_V>(6U)')
    assert start.index('WaitFlag<HardEvent::MTE2_V>(6U)') < start.index('BuildReceiveFlagMasks(')
    assert start.index('BuildReceiveFlagMasks(') < start.index('SetFlag<HardEvent::V_S>(6U)')
    assert 'WaitFlag<HardEvent::V_S>' not in start
    assert finish.index('WaitFlag<HardEvent::V_S>(6U)') < finish.index('masks.GetValue(0U)')
    assert 'SetFlag<HardEvent::V_S>' not in finish
    assert 'if (!probePrefetch_) return TrySerialBatch();' in source
    assert 'kProbeBatch * 4U + (probePrefetch_ ? kLlReceiveProbeExtraBytes : 0U)' in source
    layout = (COMMON / 'll_overlap_layout.h').read_text(encoding='utf-8')
    assert layout.index('if (lanes>6) lanes=6;') < layout.index('LlReceiveProbePrefetchFits(p.totalBytes')
    assert 'p.totalBytes+=kLlReceiveProbeExtraBytes;' in layout
    assert 'ServerDedupReceiveSync<HardEvent::S_MTE3>();' in source
    # Probe scopes end before A consumes. ON has the same waits as OFF; its
    # submit/join fragments must not count the intervening payload work.
    helper = (COMMON / 'll_receive_probe_pipeline.h').read_text(encoding='utf-8')
    assert 'LL_SEG' not in helper and 'DEEPEP_LL_STAGE_TIMER' not in helper

