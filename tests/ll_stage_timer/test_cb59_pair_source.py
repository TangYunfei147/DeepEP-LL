"""Pair handoff changes receive ownership, not packet or source semantics."""
from pathlib import Path
import importlib.util

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'csrc/kernels/dispatch/low_latency'
spec=importlib.util.spec_from_file_location('pair_source_contract',Path(__file__).with_name('test_cb59_stream_source.py'))
contract=importlib.util.module_from_spec(spec);spec.loader.exec_module(contract)


def test_pair_payload_pipeline_is_exactly_the_replay_pipeline():
    original=(BASE/'cb59/device/dispatch_server_dedup_recv.h').read_text(encoding='utf-8')
    pair=(BASE/'cb59_pair/receiver.h').read_text(encoding='utf-8')
    for name in ('Init','Slot','Probe','IssueLoad','ConsumeLoaded','ClearFlags'):
        assert contract.function_body(pair,name)==contract.function_body(original,name),name
    begin='        uint32_t scan = 0U, issued = 0U, consumed = 0U;'
    end='        LL_SPARSE_MARK(RECV_DONE);'
    assert pair.split(begin,1)[1].split(end,1)[0]==original.split(begin,1)[1].split(end,1)[0]


def test_pair_preserves_count_producers_and_persistent_lifetimes():
    source=(BASE/'cb59_pair/kernel.h').read_text(encoding='utf-8')
    assert 'Cb59Replay::ServerDedupCounts counts_' in source
    assert 'Cb59Replay::ServerDedupSender<' in source
    assert 'Cb59Stream::StreamCounts streamCounts_' in source
    work=source.split('} else if (stream_) {',1)[1].split('} else {',1)[0]
    assert work.index('streamCounts_.Prepare')<work.index('pair_.Prepare')<work.index('receiver_.Process')
    assert 'Reset' not in work and 'PIPE_ALL' not in work
    assert 'TBufPool<TPosition::VECCALC, 16> streamPool_' in source
    receiver=(BASE/'cb59_pair/receiver.h').read_text(encoding='utf-8')
    prep=receiver.split('if (stream != nullptr) {',1)[1].split('} else {',1)[0]
    assert 'pair->Count() == 0U' in prep
    assert 'stream->Count() == 0U' not in prep
    assert 'stream->UbBytes() + pair->UbBytes()' in prep


def test_init_points_cover_the_actual_order_and_remain_outside_business_sync():
    entry=(BASE/'dispatch_variant_kernel.h').read_text(encoding='utf-8')
    markers=['sparseEntry =','timer.MarkAt<LlStageTimer::ENTRY>',
             'timer.Mark<LlStageTimer::TILING_DONE>', 'AscendC::TPipe pipe;',
             'timer.Mark<LlStageTimer::PIPE_READY>',
             'Algorithm::DispatchMemoryServerDedupKernel<LlVariantInput,kWireFp8,kE4M3,kPrequantized> kernel;',
             'timer.Mark<LlStageTimer::OBJECT_READY>', 'kernel.Init(',
             'timer.Mark<LlStageTimer::INIT_DONE>', 'kernel.Process();',
             'timer.Mark<LlStageTimer::WORK_DONE>']
    assert [entry.index(m) for m in markers]==sorted(entry.index(m) for m in markers)
    assert 'PipeBarrier' not in entry and 'SyncAll' not in entry
    for path in ('dispatch_server_dedup_kernel.h','cb59/dispatch_server_dedup_kernel.h',
                 'cb59_stream/kernel.h','cb59_pair/kernel.h','cb59_small/dispatch_server_dedup_kernel.h'):
        source=(BASE/path).read_text(encoding='utf-8')
        assert source.index('RotateDispatchServerDedupMagic(')<source.index('LL_SPARSE_MARK(MAGIC_DONE)')<source.index('counts_.Init(')
