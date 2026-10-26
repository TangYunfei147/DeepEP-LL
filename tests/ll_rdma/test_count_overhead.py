"""Contracts for count UB lifetimes, publication and prefix reuse."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEVICE = ROOT / "csrc/kernels/dispatch/low_latency/device"


def _source(name):
    return (DEVICE / name).read_text(encoding="utf-8")


def _block(source, marker):
    """Return the actual braced production block after a unique marker."""
    start = source.index("{", source.index(marker))
    depth = 1
    for end in range(start + 1, len(source)):
        depth += (source[end] == "{") - (source[end] == "}")
        if depth == 0:
            return source[start + 1:end]
    raise AssertionError(f"Unclosed block: {marker}")


def test_count_prefix_is_loaded_only_by_its_output_owner():
    counts = _source("dispatch_server_dedup_counts.h")
    complete = _block(counts, "void ServerDedupCounts::CompleteCumsum()")
    owner = _block(complete, "if (receiver == 0U)")
    before_owner = complete[:complete.index("if (receiver == 0U)")]
    assert "WaitCumSumFlag();" in before_owner  # Every receiver still retires its ready row.
    assert "prefixBuf_" not in before_owner
    assert "DataCopyPad(prefix," not in before_owner
    assert "InitBuffer(prefixBuf_" in owner
    assert owner.index("DataCopyPad(prefix,") < owner.index("SyncFunc<HardEvent::MTE2_S>()")
    assert owner.index("SyncFunc<HardEvent::MTE2_S>()") < owner.index("prefix.GetValue(")
    assert "DataCopyPad(expertTokenNumsOutGMTensor_" in owner
    assert "LocalTensor<int32_t> Prefix()" not in counts
    receiver = _block(_source("dispatch_server_dedup_recv.h"), "void Process(")
    assert receiver.index("pipe_->Reset();") < receiver.index("DataCopyPad(prefix, prefixGlobal")
    assert receiver.index("ServerDedupReceiveSync<HardEvent::MTE2_S>()") < receiver.index("tasks_.Build(prefix")


def test_paired_count_records_live_until_the_existing_reuse_barrier():
    counts = _source("dispatch_server_dedup_counts.h")
    send = _block(counts, "void ServerDedupCounts::SendCnt()")
    first = send.index("if (count) SendExpertCntRange(start, end)")
    overwrite = send.index("routeExchange_.ReadPeer")
    assert "SyncFunc<HardEvent::MTE3_S>();" in send[first:overwrite]
    last = send.index("SendExpertCntRange(start,end)")
    assert "SyncFunc<HardEvent::MTE3_S>();" not in send[last:]
    assert "Reset(" not in send
    assert "tpipe_->InitBuffer(statusBuf_, ReservedBytes())" in send
    assert "kEpServerDedupCountUbBytes - ReservedBytes()" in send
    assert "return stagePool_;" in _block(counts, "ScratchPool()")
    assert "PipeBarrier<PIPE_ALL>();" in _block(counts, "void Finish()")
    kernel = (DEVICE.parent / "dispatch_server_dedup_kernel.h").read_text(encoding="utf-8")
    assert "sender_.Process(counts_.ScratchPool(), counts_.ReservedBytes(), timer_)" in kernel
    complete = kernel.index("counts_.CompleteCumsum();")
    consume = kernel.index("receiver_.Process(timer_);")
    assert "PipeBarrier<PIPE_ALL>();" in kernel[complete:consume]
