"""Lossless text transfer, size bounds and corrupt/untrusted input handling."""

import hashlib
import importlib.util
import lzma
from pathlib import Path
import random

import pytest


spec = importlib.util.spec_from_file_location("ll_text_codec", Path(__file__).with_name("text_codec.py"))
codec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(codec)


def noise(size):
    return random.Random(1729).randbytes(size)


def rewrite(envelope, **fields):
    lines = envelope.splitlines()
    for index, line in enumerate(lines):
        key, separator, _ = line.partition(": ")
        if separator and key in fields:
            lines[index] = f"{key}: {fields[key]}"
    return "\n".join(lines) + "\n"


def custom_parts(raw, compressed):
    metadata = dict(raw_bytes=len(raw), raw_sha256=hashlib.sha256(raw).hexdigest(),
                    compressed_bytes=len(compressed), compressed_sha256=hashlib.sha256(compressed).hexdigest())
    return [codec._envelope(metadata, compressed, 1, 1)]


@pytest.mark.parametrize("limit", [4096, 6000, 60000, 65535])
@pytest.mark.parametrize("payload", [b"", b"count,cycles\n" * 30000,
                                    "目标机，前缀和，等待\n".encode("utf-8") * 10000,
                                    noise(160000)], ids=["empty", "repeated", "unicode", "incompressible"])
def test_round_trip_and_crlf_limits(limit, payload):
    metadata, parts = codec.make_parts(payload, limit)
    assert metadata["raw_bytes"] == len(payload)
    assert metadata["part_count"] == len(parts)
    for part in parts:
        assert part.isascii()
        windows = part.replace("\n", "\r\n")
        assert len(windows) == len(windows.encode("utf-8")) <= limit
    status, restored = codec.inspect_parts(reversed(parts))
    assert status["complete"]
    assert not status["missing_parts"]
    assert restored == payload


def test_repeated_content_is_materially_smaller():
    payload = (b'{"core":0,"stage":"PREFIX_COMPLETE","cycles":12345}\n' * 20000)
    _, parts = codec.make_parts(payload)
    assert sum(len(part) for part in parts) < len(payload) // 50


def test_wrapping_bom_crlf_multiple_blocks_and_duplicates():
    payload = noise(22000)
    _, parts = codec.make_parts(payload, 4096)
    altered = []
    for part in parts:
        head, remainder = part.split("\nDATA\n")
        body, end = remainder.split(codec.END)
        base = "".join(body.split())
        wrapped = " \t\r\n".join(base[i:i + 113] for i in range(0, len(base), 113))
        altered.append("\ufeff" + head.replace("\n", "\r\n") + "\r\nDATA\r\n" + wrapped + "\r\n" + codec.END + end)
    status, restored = codec.inspect_parts(["\n".join(reversed(altered)), parts[0]])
    assert restored == payload
    assert status["duplicate_parts"] == 1


def test_missing_parts_reported_without_partial_restore():
    _, parts = codec.make_parts(noise(24000), 4096)
    status, payload = codec.inspect_parts([parts[-1], parts[0]])
    assert payload is None
    assert not status["complete"]
    assert status["missing_parts"] == list(range(2, len(parts)))


@pytest.mark.parametrize("value", [0, 4095, 65536, True, 6000.0, "6000"])
def test_invalid_limits(value):
    with pytest.raises(ValueError, match="limit"):
        codec.make_parts(b"data", value)


@pytest.mark.parametrize("text", ["", "  ", "abc", "DATA\nYWJj", codec.BEGIN + "\nabc",
                                   codec.BEGIN + "\nDATA\n" + codec.BEGIN + "\n"])
def test_missing_and_truncated_envelopes(text):
    with pytest.raises(ValueError, match="envelope"):
        codec.inspect_parts([text])


def test_bad_base64_and_chunk_hash():
    _, parts = codec.make_parts(b"all records preserved")
    with pytest.raises(ValueError, match="Base64"):
        codec.inspect_parts([parts[0].replace("\nDATA\n", "\nDATA\n!")])
    with pytest.raises(ValueError, match="chunk SHA256"):
        codec.inspect_parts([rewrite(parts[0], chunk_sha256="0" * 64)])


def test_mixed_bundles_and_conflicting_duplicate():
    metadata, parts = codec.make_parts(noise(10000), 4096)
    _, other = codec.make_parts(b"different bundle")
    with pytest.raises(ValueError, match="mixed"):
        codec.inspect_parts([parts[0], other[0]])
    chunk = b"different valid chunk"
    conflict = codec._envelope(metadata, chunk, 1, len(parts))
    with pytest.raises(ValueError, match="conflicting duplicate"):
        codec.inspect_parts([parts[0], conflict])


@pytest.mark.parametrize("field", ["compressed_sha256", "raw_sha256"])
def test_full_payload_hashes_are_verified(field):
    _, parts = codec.make_parts(b"content")
    with pytest.raises(ValueError, match="SHA256"):
        codec.inspect_parts([rewrite(parts[0], **{field: "0" * 64})])


def test_sizes_and_unsupported_fields():
    _, parts = codec.make_parts(b"content")
    for changes in ({"raw_bytes": f"{codec.MAX_RAW_BYTES + 1:012d}"},
                    {"compressed_bytes": f"{codec.MAX_COMPRESSED_BYTES + 1:012d}"},
                    {"part": "000000/000001"}, {"part": "1/1"}, {"raw_bytes": "-1"},
                    {"format": "unknown"}):
        with pytest.raises(ValueError):
            codec.inspect_parts([rewrite(parts[0], **changes)])
    with pytest.raises(ValueError, match="duplicate"):
        codec.inspect_parts([parts[0].replace("\nDATA\n", "\nraw_bytes: 000000000007\nDATA\n")])


def test_oversize_envelope_rejected():
    _, parts = codec.make_parts(b"content")
    oversized = parts[0].replace("\nDATA\n", "\nDATA\n" + " " * codec.MAX_LIMIT)
    with pytest.raises(ValueError, match="65535"):
        codec.inspect_parts([oversized])


def test_truncated_lzma_and_trailing_streams_rejected():
    raw = b"preserve every captured interval" * 200
    compressed = lzma.compress(raw)
    for invalid in (compressed[:-5], compressed + b"trailing", compressed + lzma.compress(b"second")):
        with pytest.raises(ValueError, match="truncated|trailing|additional"):
            codec.inspect_parts(custom_parts(raw, invalid))


def test_larger_than_declared_decompression_stops_at_bound():
    raw = b"a" * 1000000
    with pytest.raises(ValueError, match="size mismatch"):
        codec.inspect_parts(custom_parts(b"a", lzma.compress(raw)))


def test_decoder_memory_limit_enforced(monkeypatch):
    raw = b"small file with a decoder dictionary larger than the limit"
    compressed = lzma.compress(raw, filters=[{"id": lzma.FILTER_LZMA2, "dict_size": 8 * 1024 * 1024}])
    monkeypatch.setattr(codec, "DECOMPRESS_MEMLIMIT", 1024 * 1024)
    with pytest.raises(ValueError, match="LZMA decompression failed"):
        codec.inspect_parts(custom_parts(raw, compressed))


def test_make_parts_rejects_nonbytes_and_oversize(monkeypatch):
    with pytest.raises(TypeError, match="bytes"):
        codec.make_parts("text")
    monkeypatch.setattr(codec, "MAX_RAW_BYTES", 3)
    with pytest.raises(ValueError, match="512 MiB"):
        codec.make_parts(b"1234")


def test_character_loss_is_not_accepted_as_whitespace_change():
    _, parts = codec.make_parts(noise(1000))
    head, body = parts[0].split("\nDATA\n")
    with pytest.raises(ValueError, match="Base64|chunk size"):
        codec.inspect_parts(head + "\nDATA\n" + body[1:])


def test_group_mixed_bundles_preserves_first_seen_order_and_repetition():
    first_raw = noise(10000)
    second_raw = b"all summary and context fields" * 2000
    _, first = codec.make_parts(first_raw, 4096)
    _, second = codec.make_parts(second_raw, 4096)
    pasted = "\ufeff" + first[-1] + "\n" + second[0] + first[0]
    groups = codec.group_parts([pasted, "\n".join(first[1:]), second[0]])
    assert len(groups) == 2
    status, restored = codec.inspect_parts(groups[0])
    assert restored == first_raw
    assert status["duplicate_parts"] == 1
    status, restored = codec.inspect_parts(groups[1])
    assert restored == second_raw
    assert status["duplicate_parts"] == 1


@pytest.mark.parametrize("damage", ["truncated", "outside", "missing", "invalid_base64"])
def test_group_rejects_bad_text_instead_of_ignoring_it(damage):
    _, parts = codec.make_parts(b"original")
    corrupt = {"truncated": parts[0][:-len(codec.END)],
               "outside": parts[0] + "arbitrary unexpected text",
               "missing": "",
               "invalid_base64": parts[0].replace("\nDATA\n", "\nDATA\n!")}[damage]
    with pytest.raises(ValueError):
        codec.group_parts([parts[0], corrupt])


@pytest.mark.parametrize("changes", [{"raw_sha256": "0" * 64},
                                      {"part": "000001/000002"}])
def test_group_keeps_same_compressed_hash_conflicts_for_strict_decoder(changes):
    _, parts = codec.make_parts(b"original")
    groups = codec.group_parts([parts[0], rewrite(parts[0], **changes)])
    assert len(groups) == 1
    assert len(groups[0]) == 2
    with pytest.raises(ValueError, match="mixed.*conflicting"):
        codec.inspect_parts(groups[0])


def test_group_rejects_no_inputs():
    with pytest.raises(ValueError, match="no transfer envelope"):
        codec.group_parts([])
