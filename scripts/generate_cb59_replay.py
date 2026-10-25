#!/usr/bin/env python3
"""Reproduce the cb59 dispatch replay without changing its business instructions.

The reference comes from an immutable Git object. Only includes/namespace and
timer plumbing are adapted. The committed manifest contains the full normalized
business-token and synchronization sequences, so validation is stronger than a
generated-file hash and does not require Git history on the target machine.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = "cb59bff99ecd39d27ef57788c5e6fd0009e4e636"
BASE = "csrc/kernels/dispatch/low_latency"
DEST = ROOT / BASE / "cb59"
MANIFEST = ROOT / "tests/ll_stage_timer/fixtures/cb59_replay_tokens.json"
FILES = (
    "dispatch_server_dedup_kernel.h",
    "device/dispatch_server_dedup_counts.h",
    "device/dispatch_server_dedup_send.h",
    "device/dispatch_server_dedup_recv.h",
    "device/dispatch_server_dedup_receive_packet.h",
    "device/dispatch_server_dedup_receive_tasks.h",
)
MILESTONE_CONTRACT = {
    "schema": 15,
    "diagnostic_synchronization_added": False,
    "counts_received": "cb59: accepted count snapshot, extraction and existing Sum V_S complete; before local prefix, sum publication, and count clear",
    "counts_received_current_difference": "current keeps its historical point after count-clear submission; do not compare this adjacent stage across implementations",
    "prefix_local_done": "producer-local scalar prefix complete, before asynchronous sum publication",
    "prefix_ready": "CompleteCumsum returned and the original caller PIPE_ALL retired its sources, before receiver Reset/reload",
    "send_done": "only when FIRST_SEND exists, after the original Counts::Finish PIPE_ALL",
    "recv_done": "only nonempty receive workers, after their original final PIPE_ALL",
    "cross_implementation_comparison": "whole OFF task and qualified ENTRY-to-PREFIX_READY/FIRST_RECV_PROBE/WORK_DONE endpoints; producer cohorts and adjacent stages differ",
}
_OLD_TIMER = re.compile(
    r"^[ \t]*if \(timer_?\) timer_?->(?:Start|End)\(LlStageTimer::\w+\);[ \t]*\n",
    re.M,
)
_COMMENTS = re.compile(r'//[^\n]*|/\*.*?\*/', re.S)
_TOKEN = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[A-Za-z_]\w*|(?:\d[\w.]*|\S)')


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f"expected one occurrence: {old!r}; found {source.count(old)}")
    return source.replace(old, new, 1)


def adapt(relative: str, source: str) -> str:
    """Only explicit, reviewable substitutions; no business-expression rewrites."""
    source = _OLD_TIMER.sub("", source)
    source = replace_once(source, "namespace DispatchDedup {", "namespace DispatchDedup::Cb59Replay {")
    source = source.replace("}  // namespace DispatchDedup", "}  // namespace DispatchDedup::Cb59Replay")
    if relative == FILES[0]:
        source = source.replace('"common/ll_stage_timer_device.h"', '"../common/ll_stage_timer_device.h"')
        source = source.replace('"device/dispatch_server_dedup_magic.h"', '"../device/dispatch_server_dedup_magic.h"')
        source = source.replace('#include "../common/ll_stage_timer_device.h"',
                                '#include "../common/ll_stage_timer_device.h"\n#include "../common/ll_expert_timer_device.h"')
        source = replace_once(source, "LlStageTimer::DeviceTimer* timer = nullptr)",
                              "LlStageTimer::DeviceTimer* timer = nullptr,\n"
                              "                                LlExpertTimer::DeviceTimer* expertTimer = nullptr)")
        source = replace_once(source, "        timer_ = timer;", "        (void)expertTimer;\n        timer_ = timer;")
        source = replace_once(source, "                     magic, pipe);",
                              "                     magic, pipe);\n        counts_.SetTimer(timer_);")
        source = replace_once(source, "            PipeBarrier<PIPE_ALL>();",
                              "            PipeBarrier<PIPE_ALL>();\n            LL_SPARSE_MARK(PREFIX_READY);")
        source = replace_once(source, "        counts_.Finish();",
                              "        counts_.Finish();\n        LL_SPARSE_AFTER(FIRST_SEND, SEND_DONE);")
    else:
        source = source.replace('"../common/', '"../../common/')
        source = source.replace('"../../../common/', '"../../../../common/')
        if relative == FILES[1]:
            source = source.replace('"../../common/ep_memory_server_dedup_count_layout.h"', '"../count_layout.h"')
            source = replace_once(source, '#include "../count_layout.h"',
                                  '#include "../count_layout.h"\n#include "../../common/ll_stage_timer_device.h"')
            source = replace_once(source, "public:\n", "public:\n"
                                  "    __aicore__ inline void SetTimer(LlStageTimer::DeviceTimer* timer) { timer_ = timer; }\n")
            source = replace_once(source, "    TPipe* tpipe_ = nullptr;",
                                  "    LlStageTimer::DeviceTimer* timer_ = nullptr;\n    TPipe* tpipe_ = nullptr;")
            source = replace_once(source, "    SendExpertCntRange(start, end);",
                                  "    SendExpertCntRange(start, end);\n    LL_SPARSE_MARK(COUNTS_SENT);")
            source = replace_once(source, "    float sumOfRecvCnt = statusSumOutTensor.ReinterpretCast<float>().GetValue(0);",
                                  "    float sumOfRecvCnt = statusSumOutTensor.ReinterpretCast<float>().GetValue(0);\n"
                                  "    LL_SPARSE_MARK(COUNTS_RECEIVED);")
            source = replace_once(source, "    PublishRecvCntSum(sumOfRecvCnt);",
                                  "    LL_SPARSE_MARK(PREFIX_LOCAL_DONE);\n    PublishRecvCntSum(sumOfRecvCnt);")
        elif relative == FILES[2]:
            source = source.replace('"dispatch_dedup_mxfp8_quant.h"', '"../../device/dispatch_dedup_mxfp8_quant.h"')
            source = replace_once(source, '#include "../../common/ll_stage_timer_device.h"',
                                  '#include "../../common/ll_stage_timer_device.h"\n#include "../count_layout.h"')
            source = replace_once(source, "        if (enableDedup_)\n            SendWithDedup(input);",
                                  "        LL_SPARSE_MARK(SEND_PREPARED);\n        if (enableDedup_)\n            SendWithDedup(input);")
            source = replace_once(source, "        DataCopy(destinationGm, out, uint32_t(data_.tokenStride));",
                                  "        LL_SPARSE_MARK(FIRST_SEND);\n        DataCopy(destinationGm, out, uint32_t(data_.tokenStride));")
        elif relative == FILES[3]:
            source = replace_once(source, '#include "../../common/ll_stage_timer_device.h"',
                                  '#include "../../common/ll_stage_timer_device.h"\n#include "../count_layout.h"')
            source = replace_once(source, "    {\n        // 1. 调用方已完成计数阶段的写入", "    {\n        timer_ = timer;\n        // 1. 调用方已完成计数阶段的写入")
            source = replace_once(source, "        if (tasks_.Count() == 0U) {\n            return;",
                                  "        if (tasks_.Count() == 0U) {\n            LL_SPARSE_MARK(RECV_PREPARED);\n            return;")
            source = replace_once(source, "        uint32_t scan = 0U, issued = 0U, consumed = 0U;",
                                  "        LL_SPARSE_MARK(RECV_PREPARED);\n        uint32_t scan = 0U, issued = 0U, consumed = 0U;")
            source = replace_once(source, "            } while (probe.readyMask == 0U);",
                                  "            } while (probe.readyMask == 0U);\n            LL_SPARSE_MARK(FIRST_RECV_READY);")
            source = replace_once(source, "        PipeBarrier<PIPE_ALL>();",
                                  "        PipeBarrier<PIPE_ALL>();\n        LL_SPARSE_MARK(RECV_DONE);")
            source = replace_once(source, "    __aicore__ inline ProbeResult Probe(uint32_t start, uint32_t count)\n    {",
                                  "    __aicore__ inline ProbeResult Probe(uint32_t start, uint32_t count)\n    {\n        LL_SPARSE_MARK(FIRST_RECV_PROBE);")
            source = replace_once(source, "    const __gm__ DispatchDedup::CommArgs* args_ = nullptr;",
                                  "    LlStageTimer::DeviceTimer* timer_ = nullptr;\n    const __gm__ DispatchDedup::CommArgs* args_ = nullptr;")
    return source


def includes(source: str) -> list[str]:
    return re.findall(r'^#include\s+(.+)$', source, re.M)


def business_source(source: str) -> str:
    """Remove only enumerated diagnostic/namespace/include adaptations."""
    source = _OLD_TIMER.sub("", source)
    source = _COMMENTS.sub("", source)
    source = re.sub(r'^#include[^\n]*\n', '', source, flags=re.M)
    source = source.replace("namespace DispatchDedup::Cb59Replay {", "namespace DispatchDedup {")
    source = re.sub(r'LL_SPARSE_(?:MARK|AFTER)\([^;]*\);', '', source)
    source = re.sub(r'__aicore__ inline void SetTimer\(LlStageTimer::DeviceTimer\* timer\)\s*\{\s*timer_ = timer;\s*\}', '', source)
    source = re.sub(r'counts_\.SetTimer\(timer_\);', '', source)
    source = re.sub(r'LlStageTimer::DeviceTimer\* timer_ = nullptr;', '', source)
    source = re.sub(r'\btimer_ = timer;', '', source)
    source = re.sub(r',\s*LlExpertTimer::DeviceTimer\* expertTimer = nullptr', '', source)
    source = source.replace('(void)expertTimer;', '')
    # The legacy timer parameters/calls are retained in both sources; unlike
    # adding stubs, this does not create any old stage IDs or runtime metrics.
    return source


def business_tokens(source: str) -> list[str]:
    return _TOKEN.findall(business_source(source))


def synchronization(source: str) -> list[str]:
    tokens = business_tokens(source)
    names = {"Sync", "SyncFunc", "PipeBarrier", "SetFlag", "WaitFlag", "ServerDedupReceiveSync", "Reset", "SetCtrlSpr"}
    result = []
    for i, token in enumerate(tokens):
        if token not in names or i + 1 >= len(tokens) or tokens[i + 1] not in {"<", "("}:
            continue
        # These are standalone calls. Including the whole statement records
        # the event ID/expression and repeated-call ordering, not just a count.
        try:
            end = tokens.index(";", i + 1)
        except ValueError:
            raise ValueError(f"unterminated synchronization statement at {i}") from None
        result.append(" ".join(tokens[i:end + 1]))
    return result


def validate_file(relative: str, source: str, entry: dict) -> None:
    if includes(source) != entry["replay_includes"]:
        raise ValueError(f"{relative}: include mapping changed")
    actual = business_tokens(source)
    expected = entry["business_tokens"]
    if actual != expected:
        mismatch = next((i for i, pair in enumerate(zip(actual, expected)) if pair[0] != pair[1]), min(len(actual), len(expected)))
        raise ValueError(f"{relative}: business tokens changed near {mismatch}: actual={actual[mismatch:mismatch+12]!r}, expected={expected[mismatch:mismatch+12]!r}")
    if synchronization(source) != entry["synchronization"]:
        raise ValueError(f"{relative}: synchronization sequence changed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify Git extraction and committed files without writes")
    args = parser.parse_args()
    entries = {}
    outputs = {}
    for relative in FILES:
        original = subprocess.check_output(["git", "show", f"{REFERENCE}:{BASE}/{relative}"], cwd=ROOT).decode("utf-8")
        output = adapt(relative, original)
        entry = {
            "source_path": f"{BASE}/{relative}",
            "source_sha256": hashlib.sha256(original.encode()).hexdigest(),
            "replay_includes": includes(output),
            "business_tokens": business_tokens(original),
            "synchronization": synchronization(original),
        }
        validate_file(relative, output, entry)
        entries[relative] = entry
        outputs[relative] = output
    manifest = {"reference_commit": REFERENCE, "normalization_version": 1,
                "milestone_contract": MILESTONE_CONTRACT, "files": entries}
    if args.check:
        saved = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if saved != manifest:
            raise ValueError("reference-token manifest does not match immutable Git source")
        for relative, output in outputs.items():
            actual = (DEST / relative).read_text(encoding="utf-8")
            validate_file(relative, actual, entries[relative])
            if actual != output:
                raise ValueError(f"{relative}: replay is not the deterministic generated text")
        print(f"Verified {len(FILES)} cb59 replay files: exact regeneration, business tokens, and synchronization sequences.")
    else:
        for relative, output in outputs.items():
            target = DEST / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(output, encoding="utf-8", newline="\n")
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        print(f"Generated {len(FILES)} cb59 replay files and their business-token/synchronization manifest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
