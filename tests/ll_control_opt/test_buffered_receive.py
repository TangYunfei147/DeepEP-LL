"""Buffered-RX protocol/CPU tests; these do not compile or execute AscendC."""
from pathlib import Path
import os
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
LOW = ROOT/'csrc/kernels/dispatch/low_latency'


@pytest.mark.parametrize('send_cores', (1, 28, 30, 55))
def test_production_cadence_window_and_pending_states(tmp_path, send_cores):
    compiler = shutil.which('g++') or shutil.which('clang++')
    if not compiler:
        pytest.skip('CPU C++ compiler required; no device claim from this fixture')
    binary = tmp_path/('buffered.exe' if os.name == 'nt' else 'buffered')
    subprocess.run([compiler, '-std=c++17', '-O2', f'-DDEEPEP_LL_SEND_CORES={send_cores}',
                    f'-DDEEPEP_LL_RECV_CORES={56-send_cores}',
                    '-I', str(ROOT/'tests/ll_stage_timer/fixtures/receive_stub'), '-I', str(ROOT),
                    str(Path(__file__).with_name('buffered_receive.cpp')), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_source_event_leases_and_once_preparation():
    kernel = (LOW/'dispatch_server_dedup_kernel.h').read_text(encoding='utf-8')
    counts = (LOW/'device/dispatch_server_dedup_counts.h').read_text(encoding='utf-8')
    receiver = (LOW/'device/dispatch_server_dedup_recv.h').read_text(encoding='utf-8')
    early = kernel[kernel.index('struct EarlyProgress'):kernel.index('// S remote submission')]
    assert early.count('AllocEventID<HardEvent::MTE2_V>') == 1
    assert early.count('AllocEventID<HardEvent::V_S>') == 1
    assert early.count('ReleaseEventID<HardEvent::MTE2_V>') == 1
    assert early.count('ReleaseEventID<HardEvent::V_S>') == 1
    assert early.index('sender_.Process(') < early.index('progress.Finish();')
    assert 'SetCtrlSpr' not in early and 'GetCtrlSpr' not in early
    # Counts.Init selects the established mode before source sending. Early
    # prefix uses raw integers and packet flags are ordinary0/1/1.5 floats;
    # neither requires a new control-bit transition while input V is active.
    assert 'SetCtrlSpr<60, 60>(0);' in counts
    quant = (LOW/'device/dispatch_dedup_mxfp8_quant.h').read_text(encoding='utf-8')
    assert 'SetCtrlSpr' not in quant and 'GetCtrlSpr' not in quant
    prefix_start = counts.index('__aicore__ inline bool ServerDedupCounts::TryLoadPrefixOnce')
    prefix = counts[prefix_start:counts.index('// 摘要：对原始路由执行逐专家匹配', prefix_start)]
    assert 'WaitCumSumFlag' not in prefix and 'while' not in prefix
    assert 'SetCtrlSpr' not in prefix and 'GetCtrlSpr' not in prefix
    assert 'MTE2_S' not in prefix.replace("sender's pending-input MTE2_S event", '')
    assert prefix.count('SetFlag<HardEvent::MTE2_V>(mte2Vector)') == 2
    assert prefix.count('WaitFlag<HardEvent::MTE2_V>(mte2Vector)') == 2
    assert prefix.count('SetFlag<HardEvent::V_S>(vectorScalar)') == 2
    assert prefix.count('WaitFlag<HardEvent::V_S>(vectorScalar)') == 2
    assert 'if (prefixReaderAllocated_) return;' in counts
    assert 'if (!receiverPrepared_)' in kernel
    run = receiver[receiver.index('__aicore__ inline void Run()'):receiver.index('\nprivate:')]
    assert run.index('stagedCount_') < run.index('while (tasks_.PendingCount()) TryBatch();') < run.index('ClearFlags();')
    acceptance = receiver[receiver.index('__aicore__ inline void AcceptEarly('):receiver.index('struct ProbePipelineOps')]
    buffered, immediate = acceptance.split('} else {', 1)
    assert 'IssueLoad(' in buffered and 'staged.SetValue' in buffered
    assert 'ConsumeReady(' not in buffered and 'ConsumeLoaded(' not in buffered
    assert 'PrimeReady(0U,count,LlReceiveProbeResult{selected,selected})' in immediate
    assert 'ConsumeReady(batch);' in immediate and 'staged.SetValue' not in immediate
    consumer_start = receiver.index('__aicore__ inline void ConsumeReady(ReadyBatch& batch)',
                                    receiver.index('__aicore__ inline void FillReadyLanes('))
    consumer = receiver[consumer_start:receiver.index('__aicore__ inline void RemoveProbe(uint32_t start,uint32_t count,uint32_t ready) {', consumer_start)]
    assert 'ConsumeLoaded(' in consumer and '++consumed_;' in consumer
    prepare = receiver[receiver.index('__aicore__ inline void Prepare('):receiver.index('__aicore__ inline bool CanEarlyReceive')]
    assert 'if (senderActive)' in prepare and 'WaitFlag<HardEvent::V_S>(vectorScalar)' in prepare


def test_sender_order_metadata_and_payload_consumer_remain_original():
    sender = (LOW/'device/dispatch_server_dedup_send.h').read_text(encoding='utf-8')
    receiver = (LOW/'device/dispatch_server_dedup_recv.h').read_text(encoding='utf-8')
    assert 'cadence.Reset(packets_,inputPrefetch_.HasPending());' in sender
    direct = sender[sender.index('LlSubmittedDirectCadence cadence;'):sender.index('// 摘要：非去重发送流程')]
    assert 'EndInputPrefetch' not in direct and 'BeginInputPrefetch' not in direct
    assert direct.index('SendPacket(input, start_ + token, k, expert, q, 0U);') < direct.index('cadence.Visit(packets_)')
    assert direct.index('timer_->End(LlStageTimer::SEND_TOKENS,sendBegin)') < direct.index('progress();')
    assert direct.index('progress();') < direct.index('sendBegin=timer_ ? timer_->Begin(LlStageTimer::SEND_TOKENS)')
    assert 'kLlSendCoreNum' in (LOW/'common/ll_overlap_layout.h').read_text(encoding='utf-8')
    # The candidate intentionally changes discovery/scheduling only. Compare
    # intact packet generation/consumption and retirement against c1e4faa.
    for path, names in ((LOW/'device/dispatch_server_dedup_send.h', ('FillTail', 'ConsumePacket')),
                        (LOW/'device/dispatch_server_dedup_recv.h', ('IssueLoad', 'ConsumeLoaded', 'ClearFlags'))):
        current = path.read_text(encoding='utf-8')
        previous = subprocess.check_output(['git', '-C', str(ROOT), 'show',
            'c1e4faaf32806da84f8818e1ac3eb34864b7958d:'+path.relative_to(ROOT).as_posix()], text=True, encoding='utf-8')
        for name in names:
            def body(text):
                begin = text.index('__aicore__ inline void '+name+'(')
                brace = text.index('{', begin); level=1; end=brace+1
                while level:
                    level += (text[end]=='{')-(text[end]=='}'); end+=1
                return text[brace:end]
            assert body(current) == body(previous), name
    assert sender.index('EndInputPrefetch(input);') < sender.index('LL_SEG_BEGIN(metaPublish,TX_METADATA_PUBLISH);')

