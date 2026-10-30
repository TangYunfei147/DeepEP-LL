"""CPU verification of the immutable cb59 algorithm replay and reference boundaries."""
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








def test_manifest_records_cross_implementation_point_difference() -> None:
    assert MANIFEST["reference_commit"] == REPLAY.REFERENCE
    assert MANIFEST["milestone_contract"] == REPLAY.MILESTONE_CONTRACT
    assert MANIFEST["milestone_contract"]["business_synchronization_added"] is False
    assert MANIFEST["milestone_contract"]["schema"] == 7
