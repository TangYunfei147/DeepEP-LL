"""Preserve cb59 packet semantics while replacing its receive ownership barrier."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'csrc/kernels/dispatch/low_latency'


def function_body(source, name):
    match = re.search(r'\b' + re.escape(name) + r'\s*\([^;]*?\)\s*(?:const\s*)?\{', source)
    assert match, name
    begin = source.index('{', match.start())
    depth = 1
    end = begin + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[begin:end]


def test_stream_keeps_original_wire_probe_copy_metadata_and_cleanup():
    original = (BASE / 'cb59/device/dispatch_server_dedup_recv.h').read_text(encoding='utf-8')
    candidate = (BASE / 'cb59_stream/receiver.h').read_text(encoding='utf-8')
    for name in ('Slot', 'Probe', 'IssueLoad', 'ConsumeLoaded', 'ClearFlags'):
        assert function_body(candidate, name) == function_body(original, name), name
    # Scheduling inside the owned packet slice remains exactly the established
    # ready bitmap / bounded lane pipeline. Only slice ownership is redesigned.
    begin = '        uint32_t scan = 0U, issued = 0U, consumed = 0U;'
    end = '        LL_SPARSE_MARK(RECV_DONE);'
    assert candidate.split(begin, 1)[1].split(end, 1)[0] == original.split(begin, 1)[1].split(end, 1)[0]


def test_stream_does_not_modify_source_count_or_sender_algorithm():
    source = (BASE / 'cb59_stream/kernel.h').read_text(encoding='utf-8')
    assert 'Cb59Replay::ServerDedupCounts counts_' in source
    assert 'Cb59Replay::ServerDedupSender<' in source
    assert 'counts_.SendCnt();' in source
    stream = source.split('} else if (stream_) {', 1)[1].split('} else {', 1)[0]
    assert 'counts_.RecvCnt' not in stream
    assert 'CompleteCumsum' not in stream
    assert 'PipeBarrier' not in stream and 'Reset' not in stream
    assert 'counts_.ScratchPool()' in stream and 'counts_.ReservedBytes()' in stream


def test_stream_has_no_global_prefix_reload_or_reset_before_first_probe():
    source = (BASE / 'cb59_stream/receiver.h').read_text(encoding='utf-8')
    stream = source.split('if (stream != nullptr) {', 1)[1].split('} else {', 1)[0]
    assert 'pipe_->Reset(' not in stream and 'DataCopy' not in stream
    assert 'BuildStream' in stream
    assert 'stream->Count() == 0U' in stream  # inactive owners have no LocalPrefix allocation
    assert source.index('stream->Count() == 0U') < source.index('stream->LocalPrefix()')
    assert 'LlStreamReceiverBufferBytes(tasks_.Count(), blocks)' in source


def test_stream_protocol_has_no_payload_wait_or_global_finish_barrier():
    source = (BASE / 'cb59_stream/stream_counts.h').read_text(encoding='utf-8')
    prepare = function_body(source, 'Prepare')
    assert prepare.index('PublishSums();') < prepare.index('ReadPredecessors();')
    assert 'PIPE_ALL' not in source and '->Reset(' not in source
    assert 'WaitCumSumFlag' not in source and 'CompleteCumsum' not in source
    for buffer in ('countSnapshotBuf_', 'countClearBuf_', 'localPrefixBuf_', 'publicPrefixBuf_',
                   'sumPublishBuf_', 'sumSnapshotBuf_', 'sumClearBuf_', 'expertCountsBuf_'):
        assert f'Allocate(pool, {buffer},' in prepare


def test_non_template_stream_calls_use_device_qualified_pure_helpers():
    source = (BASE / 'cb59_stream/stream_counts.h').read_text(encoding='utf-8')
    assert source.index('#define DEEPEP_LL_STREAM_DEVICE_CONTEXT 1') < source.index('#include "../common/ll_stream_receive.h"')
    assert source.index('#include "../common/ll_stream_receive.h"') < source.index('#undef DEEPEP_LL_STREAM_DEVICE_CONTEXT')
    helpers = (BASE / 'common/ll_stream_receive.h').read_text(encoding='utf-8')
    assert 'defined(DEEPEP_LL_STREAM_DEVICE_CONTEXT)' in helpers
