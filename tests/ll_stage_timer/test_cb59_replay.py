"""CPU verification of the immutable cb59 algorithm replay and sparse boundaries."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("cb59_replay_generator", ROOT / "scripts/generate_cb59_replay.py")
assert SPEC and SPEC.loader
REPLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPLAY)
MANIFEST = json.loads(REPLAY.MANIFEST.read_text(encoding="utf-8"))


def source(relative: str) -> str:
    return (REPLAY.DEST / relative).read_text(encoding="utf-8")


@pytest.mark.parametrize("relative", REPLAY.FILES)
def test_reference_business_tokens_and_synchronization(relative: str) -> None:
    # The complete reference sequences are fixtures, not hashes of generated
    # files. This works even in a checkout without the historical Git object.
    entry = MANIFEST["files"][relative]
    REPLAY.validate_file(relative, source(relative), entry)
    assert entry["business_tokens"]
    assert len(entry["source_sha256"]) == 64


@pytest.mark.parametrize("relative", REPLAY.FILES)
def test_replay_namespace_and_include_mapping(relative: str) -> None:
    text = source(relative)
    assert text.count("namespace DispatchDedup::Cb59Replay {") == 1
    assert "namespace DispatchDedup {" not in text
    assert not re.search(r"timer_?->(?:Start|End)\(", text)
    for include in REPLAY.includes(text):
        if not include.startswith('"'):
            continue
        name = include.strip('"')
        # Ascend headers come from the configured toolchain; project includes
        # either start with .. or name one of the co-located replay headers.
        if name.startswith(".") or name.startswith("dispatch_server_dedup_"):
            assert (REPLAY.DEST / relative).parent.joinpath(name).resolve().is_file(), name


def test_unknown_business_or_sync_change_is_rejected() -> None:
    relative = "device/dispatch_server_dedup_counts.h"
    text = source(relative)
    changed = text.replace("coreNum_ / 2U", "coreNum_ / 4U", 1)
    assert changed != text
    with pytest.raises(ValueError, match="business tokens changed"):
        REPLAY.validate_file(relative, changed, MANIFEST["files"][relative])
    changed = text.replace("PipeBarrier<PIPE_ALL>();", "PipeBarrier<PIPE_V>();", 1)
    assert REPLAY.synchronization(changed) != REPLAY.synchronization(text)
    with pytest.raises(ValueError, match="business tokens changed"):
        REPLAY.validate_file(relative, changed, MANIFEST["files"][relative])


def test_include_change_cannot_hide_behind_token_normalization() -> None:
    relative = "device/dispatch_server_dedup_counts.h"
    changed = source(relative).replace('"../count_layout.h"', '"../../common/ep_memory_server_dedup_count_layout.h"')
    with pytest.raises(ValueError, match="include mapping changed"):
        REPLAY.validate_file(relative, changed, MANIFEST["files"][relative])


def test_sparse_points_preserve_the_count_protocol_order() -> None:
    text = source("device/dispatch_server_dedup_counts.h")
    gather = text[text.index("__aicore__ inline void ServerDedupCounts::GatherSumRecvCnt"):]
    assert gather.index("SyncFunc<AscendC::HardEvent::V_S>();") < gather.index("LL_SPARSE_MARK(COUNTS_RECEIVED);")
    assert gather.index("LL_SPARSE_MARK(COUNTS_RECEIVED);") < gather.index("uint32_t cumulative = 0;")
    assert gather.index("prefix.SetValue(i, cumulative);") < gather.index("LL_SPARSE_MARK(PREFIX_LOCAL_DONE);")
    assert gather.index("LL_SPARSE_MARK(PREFIX_LOCAL_DONE);") < gather.index("PublishRecvCntSum(sumOfRecvCnt);")
    wait = text[text.index("__aicore__ inline void ServerDedupCounts::WaitDispatch()"):]
    assert wait.index("GatherSumRecvCnt(gatherMaskOutTensor, gatherTmpTensor, statusSumOutTensor);") < wait.index("WaitDispatchClearStatus();")
    assert "SendExpertCntRange(start, end);\n    LL_SPARSE_MARK(COUNTS_SENT);" in text
    kernel = source(REPLAY.FILES[0])
    assert "PipeBarrier<PIPE_ALL>();\n            LL_SPARSE_MARK(PREFIX_READY);" in kernel
    assert kernel.index("counts_.CompleteCumsum();") < kernel.index("LL_SPARSE_MARK(PREFIX_READY);") < kernel.index("receiver_.Process(timer_);")
    assert "counts_.Finish();\n        LL_SPARSE_AFTER(FIRST_SEND, SEND_DONE);" in kernel


def test_sparse_sender_points_do_not_add_a_completion_barrier() -> None:
    text = source("device/dispatch_server_dedup_send.h")
    assert "PipeBarrier<PIPE_V>();\n\n        LL_SPARSE_MARK(SEND_PREPARED);" in text
    assert "LL_SPARSE_MARK(FIRST_SEND);\n        DataCopy(destinationGm, out, uint32_t(data_.tokenStride));" in text
    assert "LL_SPARSE_MARK(SEND_DONE)" not in text
    assert text.count("LL_SPARSE_MARK(FIRST_SEND)") == 1


def test_sparse_receiver_points_cover_empty_and_nonempty_workers() -> None:
    text = source("device/dispatch_server_dedup_recv.h")
    assert "if (tasks_.Count() == 0U) {\n            LL_SPARSE_MARK(RECV_PREPARED);\n            return;" in text
    assert text.count("LL_SPARSE_MARK(RECV_PREPARED)") == 2
    assert "} while (probe.readyMask == 0U);\n            LL_SPARSE_MARK(FIRST_RECV_READY);" in text
    assert "PipeBarrier<PIPE_ALL>();\n        LL_SPARSE_MARK(RECV_DONE);" in text
    probe = text[text.index("__aicore__ inline ProbeResult Probe("):]
    assert probe.index("LL_SPARSE_MARK(FIRST_RECV_PROBE);") < probe.index("ReadServerDedupFlags(")
    assert "timer_ = timer;" in text


def test_manifest_records_cross_implementation_point_difference() -> None:
    assert MANIFEST["reference_commit"] == REPLAY.REFERENCE
    assert MANIFEST["milestone_contract"] == REPLAY.MILESTONE_CONTRACT
    assert MANIFEST["milestone_contract"]["diagnostic_synchronization_added"] is False
    assert "current keeps its historical point after count-clear" in MANIFEST["milestone_contract"]["counts_received_current_difference"]


def test_exactly_the_existing_fourteen_sparse_points_are_used() -> None:
    observed = set()
    for relative in REPLAY.FILES:
        for call in re.findall(r'LL_SPARSE_(?:MARK|AFTER)\(([^)]*)\)', source(relative)):
            observed.update(word.strip() for word in call.split(","))
    assert observed == {
        "COUNTS_SENT", "COUNTS_RECEIVED", "PREFIX_LOCAL_DONE", "PREFIX_READY",
        "SEND_PREPARED", "FIRST_SEND", "SEND_DONE", "RECV_PREPARED",
        "FIRST_RECV_PROBE", "FIRST_RECV_READY", "RECV_DONE",
    }
    # ENTRY / INIT_DONE / WORK_DONE stay in the common seven-TU wrapper. No
    # old scope, counter, hidden metric or additional clock read is introduced.
    for relative in REPLAY.FILES:
        assert "GetSystemCycle" not in source(relative)
