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
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('ll_reference_insertion', ROOT/'scripts/ll_reference_instrumentation.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
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
    "schema": 7,
    "reference": "kernel/ll-ndr-metadata-cores@c3d8640",
    "recording": "fixed scalar timestamps, ready-batch accumulators and spare-UB packet log",
    "business_synchronization_added": False,
    "publication": "diagnostic packet export and cache publication after original Counts.Finish",
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


def strip_historical_plumbing(source: str) -> str:
    """Discard the original immutable cb59 tracing while extracting its business code."""
    source = _OLD_TIMER.sub('', source)
    source = source.replace('ll_stage_timer_device.h', 'll_core_config.h')
    source = re.sub(r',\s*LlStageTimer::DeviceTimer\* timer = nullptr|LlStageTimer::DeviceTimer\* timer = nullptr', '', source)
    source = re.sub(r'LlStageTimer::DeviceTimer\* timer_ = nullptr;|\btimer_ = timer;', '', source)
    source = source.replace(', timer_', '').replace('Process(timer_)', 'Process()')
    return source


def adapt(relative: str, source: str) -> str:
    """Extract immutable business code and add only the current reference trace."""
    source = strip_historical_plumbing(source)
    source = replace_once(source, "namespace DispatchDedup {", "namespace DispatchDedup::Cb59Replay {")
    source = source.replace("}  // namespace DispatchDedup", "}  // namespace DispatchDedup::Cb59Replay")
    if relative == FILES[0]:
        source = source.replace('"common/ll_core_config.h"', '"../common/ll_core_config.h"')
        source = source.replace('"device/dispatch_server_dedup_magic.h"', '"../device/dispatch_server_dedup_magic.h"')
    else:
        source = source.replace('"../common/', '"../../common/')
        source = source.replace('"../../../common/', '"../../../../common/')
        if relative == FILES[1]:
            source = source.replace('"../../common/ep_memory_server_dedup_count_layout.h"', '"../count_layout.h"')
        elif relative == FILES[2]:
            source = source.replace('"dispatch_dedup_mxfp8_quant.h"', '"../../device/dispatch_dedup_mxfp8_quant.h"')
            source = '#include "../count_layout.h"\n' + source
        elif relative == FILES[3]:
            source = '#include "../count_layout.h"\n' + source
    return '\n'.join(line.rstrip() for line in reference.instrument_cb59(relative, source).splitlines()).rstrip() + '\n'



def includes(source: str) -> list[str]:
    return re.findall(r'^#include\s+(.+)$', source, re.M)


def business_source(source: str) -> str:
    """Remove only marked new diagnostics and immutable-source legacy plumbing."""
    source = reference.strip_reference(source)
    source = strip_historical_plumbing(source)
    source = _COMMENTS.sub("", source)
    source = re.sub(r'^#include[^\n]*\n', '', source, flags=re.M)
    return source.replace("namespace DispatchDedup::Cb59Replay {", "namespace DispatchDedup {")



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
    manifest = {"reference_commit": REFERENCE, "normalization_version": 2,
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
