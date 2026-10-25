"""Real portable helper and adversarial stream-order checks for the FP8 candidate."""
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
SEND = ROOT / "csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h"


def test_real_tail_emitter_bytes_and_delayed_pipeline_lifetimes(tmp_path):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("g++ unavailable; actual helper CPU check requires a C++ compiler")
    executable = tmp_path / "vector-tail.exe"
    result = subprocess.run(
        [compiler, "-std=c++17", "-O2", "-I", str(ROOT),
         str(ROOT / "tests/elastic/prequant_vector_tail_harness.cpp"), "-o", str(executable)],
        text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run([str(executable)], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "negative-control=detected" in result.stdout


def test_native_adapter_uses_aligned_vector_bits_without_scalar_packet_stores():
    source = SEND.read_text(encoding="utf-8")
    adapter = source[source.index("struct VectorTailWriter"):source.index("// The generator retains")]
    assert "Duplicate<uint32_t>" in adapter
    assert "store.blockByteOffset / 4U" in adapter
    assert "uint64_t mask[2] = {store.mask, 0U}" in adapter
    assert "value, mask, 1U, 1U, 8U" in adapter
    assert "ReinterpretCast<uint32_t>().GetValue(localIndex)" in adapter
    assert "SetValue" not in adapter and "reinterpret_cast<float>" not in adapter
    assert adapter.index("LlEmitPrequantVectorTail(") < adapter.index("PipeBarrier<PIPE_V>()")


def test_scope_events_and_original_queue_lane_dependencies_are_preserved():
    source = SEND.read_text(encoding="utf-8")
    assert source.count("vectorTail_ = LlUsePrequantVectorTail(") == 1
    assert "Prequantized, enableDedup_, overlap_.Async(), core, sendCores_" in source
    begin = source[source.index("void BeginInputPrefetch()"):source.index("void EndInputPrefetch(")]
    end = source[source.index("void EndInputPrefetch("):source.index("void IssueInput(")]
    issue = source[source.index("void IssueInput("):source.index("struct PrefetchOps")]
    consume_begin = source.index("void ConsumePacket(")
    consume = source[consume_begin:source.index("const __gm__ DispatchDedup::CommArgs*", consume_begin)]
    assert begin.index("if (!vectorTail_)") < begin.index("AllocEventID<HardEvent::MTE2_S>")
    assert end.index("if (!vectorTail_ && !overlap_.enabled)") < end.index("ReleaseEventID<HardEvent::MTE2_S>")
    assert issue.index("if (!vectorTail_)") < issue.index("SetFlag<HardEvent::MTE2_S>")
    assert consume.index("if (!vectorTail_)") < consume.index("WaitFlag<HardEvent::MTE2_S>")
    assert "if (vectorTail_) FillVectorTail(payload, token, firstK, mask);" in consume
    assert "else FillTail(payload, token, firstK, mask);" in consume
    assert "if (!vectorTail_) Sync<HardEvent::S_V>();" in consume
    assert "sender.pendingInput_=input.template DeQue<XType>();" in source
    assert consume.index("WaitFlag<HardEvent::MTE3_V>(lane)") < consume.index("Copy(outWords,")
    assert consume.index("Copy(outWords[64]") < consume.index("if constexpr (!NeedQuantize) input.template FreeTensor")
    assert consume.index("SetFlag<HardEvent::V_MTE3>(lane)") < consume.index("DataCopy(destinationGm")
    assert consume.index("DataCopy(destinationGm") < consume.index("SetFlag<HardEvent::MTE3_V>(lane)")
    assert source.count("InitBuffer(input, 2U, inputBytes)") == 1
    assert source.count("InitBuffer(input,2U,inputBytes)") == 1
