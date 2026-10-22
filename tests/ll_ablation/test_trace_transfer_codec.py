"""CPU checks for the bounded trace text transport, independent of trace files."""
import base64
import hashlib
import lzma
from pathlib import Path
import re
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trace_transfer_codec as codec


@pytest.fixture(scope="module")
def bundle():
    # Deterministic high-entropy text forces real splitting at the normal limit.
    return {"description": "原始trace😀", "n": None, "flags": [True, False],
            "metadata": {"counter": 2**63 + 123, "frequency_hz": 1e9},
            "data": hashlib.shake_256(b"trace transport fixture").hexdigest(160000)}


@pytest.fixture(scope="module")
def transfer(bundle):
    manifest, parts = codec.encode_bundle(bundle)
    assert len(parts) > 2
    return manifest, parts


def decode_checked(status, compressed):
    return codec.decode_bundle(compressed, expected_raw_bytes=status["raw_bytes"],
                               expected_raw_sha256=status["raw_sha256"])


def update_header(block, field, value):
    return re.sub(rf"(?m)^{re.escape(field)}: .*?$", f"{field}: {value}", block)


def payload(block):
    return block.split("\n\n", 1)[1].split(codec.END)[0]


def replace_payload(block, value):
    return block.split("\n\n", 1)[0] + "\n\n" + value + "\n" + codec.END + "\n"


def test_round_trip_preserves_raw_numbers_unicode_and_every_byte(bundle, transfer):
    manifest, parts = transfer
    status, compressed = codec.inspect_parts(parts)
    assert status["status"] == "complete"
    assert status["received_parts"] == list(range(1, len(parts) + 1))
    assert status["missing_parts"] == []
    assert status["duplicate_count"] == 0
    assert len(compressed) == manifest["compressed_bytes"]
    assert hashlib.sha256(compressed).hexdigest() == manifest["bundle_sha256"]
    assert decode_checked(status, compressed) == bundle
    assert codec.decode_bundle(compressed) == bundle


@pytest.mark.parametrize("limit", [1024, 1025, 2048, 59999, 60000, 65535])
def test_every_part_is_bounded_in_lf_crlf_and_utf16_units(limit):
    value = {"x": hashlib.shake_256(b"bounds").hexdigest(52000)}
    manifest, parts = codec.encode_bundle(value, max_bytes=limit)
    assert manifest["max_bytes"] == limit
    for index, (part, entry) in enumerate(zip(parts, manifest["parts"]), 1):
        crlf = part.replace("\n", "\r\n")
        assert part.isascii()
        assert entry["index"] == index
        assert len(part.encode("ascii")) == entry["text_bytes"]
        assert len(crlf.encode("ascii")) == entry["crlf_bytes"] <= limit
        assert len(crlf.encode("utf-16-le")) // 2 == entry["crlf_bytes"]
    status, compressed = codec.inspect_parts(parts)
    assert decode_checked(status, compressed) == value


@pytest.mark.parametrize("limit", [0, 1023, 65536, True, 60000.0, "60000"])
def test_invalid_limits_rejected(limit):
    with pytest.raises(codec.TransferError, match="max_bytes"):
        codec.encode_bundle({}, max_bytes=limit)


def test_same_bundle_has_deterministic_manifest_and_parts(transfer, bundle):
    assert codec.encode_bundle(bundle) == transfer


def test_small_empty_bundle():
    manifest, parts = codec.encode_bundle({})
    assert manifest["part_count"] == 1
    status, compressed = codec.inspect_parts(parts)
    assert decode_checked(status, compressed) == {}


def test_crlf_folded_payload_out_of_order_duplicate_and_combined_text(bundle, transfer):
    _, parts = transfer
    reformatted = []
    for part in reversed(parts):
        flat = "".join(payload(part).split())
        folded = " \t\n".join(flat[pos:pos + 19] for pos in range(0, len(flat), 19))
        reformatted.append(replace_payload(part, folded).replace("\n", "\r\n"))
    # Exact decoded copies count as duplicates even with different wrapping.
    texts = ["\r\n".join(reformatted[:2]), "\n".join(reformatted[2:]), parts[0], parts[0]]
    status, compressed = codec.inspect_parts(texts)
    assert status["duplicate_count"] == 2
    assert status["duplicate_parts"] == [1, 1]
    assert decode_checked(status, compressed) == bundle


def test_missing_parts_are_named_without_decompressing(transfer, monkeypatch):
    _, parts = transfer
    def forbidden(*args, **kwargs):
        raise AssertionError("Partial transfers must not decompress")
    monkeypatch.setattr(codec.lzma, "LZMADecompressor", forbidden)
    status, compressed = codec.inspect_parts(parts[1:-1])
    assert compressed is None
    assert status["status"] == "missing"
    assert status["missing_parts"] == [1, len(parts)]


@pytest.mark.parametrize("transform", [
    lambda p: p.replace(codec.BEGIN, "", 1),
    lambda p: p.replace(codec.END, "", 1),
    lambda p: p[:-11],
    lambda p: "prefix " + p,
    lambda p: p + "trailing partial block",
    lambda p: p.replace("\nformat:", "format:", 1),
    lambda p: p.replace("\n" + codec.END, codec.END),
    lambda p: p.replace("\n\n", "\n", 1),
    lambda p: p.replace("\n\n", "\nunknown: 1\n\n", 1),
    lambda p: p.replace("format: ll_trace_text_v1", "format: ll_trace_text_v1\nformat: ll_trace_text_v1"),
    lambda p: p.replace("codec: xz-lzma2+base64\n", ""),
    lambda p: p.replace(codec.END, codec.BEGIN + "\n" + codec.END),
    lambda p: p + "中文",
])
def test_incomplete_malformed_or_non_ascii_blocks_rejected(transform, transfer):
    _, parts = transfer
    with pytest.raises(codec.TransferError):
        codec.inspect_parts([transform(parts[0])])


@pytest.mark.parametrize("texts", [[], ["\n \t"], [b"bytes"], "not a list", [None]])
def test_no_complete_block_or_wrong_input_type(texts):
    with pytest.raises(codec.TransferError):
        codec.inspect_parts(texts)


def test_mixed_bundles_rejected(transfer):
    _, parts = transfer
    _, other = codec.encode_bundle({"another": "bundle"})
    with pytest.raises(codec.TransferError, match="Mixed bundles"):
        codec.inspect_parts([parts[0], other[0]])


def test_changed_header_conflicting_metadata_rejected(transfer):
    _, parts = transfer
    modified = update_header(parts[0], "raw-bytes", "77")
    with pytest.raises(codec.TransferError, match="conflicting transport metadata"):
        codec.inspect_parts([parts[0], modified])


def test_conflicting_duplicate_part_even_if_individually_hashed(transfer):
    _, parts = transfer
    original = parts[0]
    data = bytearray(base64.b64decode(payload(original)))
    data[50] ^= 1
    changed = replace_payload(original, base64.b64encode(data).decode("ascii"))
    changed = update_header(changed, "part-sha256", hashlib.sha256(data).hexdigest())
    with pytest.raises(codec.TransferError, match="Conflicting duplicate"):
        codec.inspect_parts([original, changed])


def test_corrupt_payload_fails_part_hash(transfer):
    _, parts = transfer
    original = parts[0]
    data = bytearray(base64.b64decode(payload(original)))
    data[50] ^= 1
    changed = replace_payload(original, base64.b64encode(data).decode("ascii"))
    with pytest.raises(codec.TransferError, match="SHA256 mismatch"):
        codec.inspect_parts([changed])


def test_payload_invalid_base64(transfer):
    _, parts = transfer
    original = parts[0]
    body = payload(original).replace("A", "!", 1)
    assert "!" in body
    with pytest.raises(codec.TransferError, match="base64"):
        codec.inspect_parts([replace_payload(original, body)])


def test_full_bundle_checksum_is_checked_after_part_checks(transfer):
    _, parts = transfer
    changed = [update_header(part, "bundle-sha256", "0" * 64) for part in parts]
    with pytest.raises(codec.TransferError, match="Bundle SHA256 mismatch"):
        codec.inspect_parts(changed)


@pytest.mark.parametrize("field,value", [
    ("part", "0/2"), ("part", "3/2"), ("part", "1/65537"),
    ("part", "1/1000000000000000000000"), ("part", "1/01"),
    ("compressed-bytes", str(128 * 1024 * 1024 + 1)),
    ("raw-bytes", str(128 * 1024 * 1024 + 1)),
    ("raw-bytes", "-1"), ("part-bytes", "49153"),
    ("part-sha256", "A" * 64), ("bundle-sha256", "short"),
    ("format", "something else"), ("codec", "pickle"),
])
def test_header_limits_and_format(field, value, transfer):
    _, parts = transfer
    with pytest.raises(codec.TransferError):
        codec.inspect_parts([update_header(parts[0], field, value)])


def test_declared_and_accumulated_sizes_are_bounded(transfer):
    manifest, parts = transfer
    first_size = manifest["parts"][0]["compressed_bytes"]
    changed = [update_header(part, "compressed-bytes", str(first_size + len(parts)))
               for part in parts]
    with pytest.raises(codec.TransferError, match="Cumulative payload"):
        codec.inspect_parts(changed)


def test_complete_transfer_requires_exact_declared_size(transfer):
    manifest, parts = transfer
    changed = [update_header(part, "compressed-bytes", str(manifest["compressed_bytes"] + 1))
               for part in parts]
    with pytest.raises(codec.TransferError, match="length mismatch"):
        codec.inspect_parts(changed)


def test_total_input_and_block_count_limited(transfer, monkeypatch):
    _, parts = transfer
    monkeypatch.setattr(codec, "MAX_INPUT_TEXT_BYTES", len(parts[0]) - 1)
    with pytest.raises(codec.TransferError, match="Cumulative input"):
        codec.inspect_parts([parts[0]])
    monkeypatch.setattr(codec, "MAX_INPUT_TEXT_BYTES", 128 * 1024 * 1024)
    monkeypatch.setattr(codec, "MAX_BLOCKS", 1)
    with pytest.raises(codec.TransferError, match="Too many input"):
        codec.inspect_parts([parts[0], parts[0]])
    with pytest.raises(codec.TransferError, match="Too many transport"):
        codec.inspect_parts([parts[0] + parts[0]])


def test_extreme_whitespace_fold_size_is_bounded(transfer, monkeypatch):
    _, parts = transfer
    monkeypatch.setattr(codec, "MAX_BLOCK_TEXT_BYTES", 1000)
    with pytest.raises(codec.TransferError, match="Text block"):
        codec.inspect_parts([parts[0]])


def test_encode_size_and_part_count_limits(monkeypatch):
    monkeypatch.setattr(codec, "MAX_RAW_BYTES", 32)
    with pytest.raises(codec.TransferError, match="Uncompressed"):
        codec.encode_bundle({"x": "a" * 33})
    monkeypatch.setattr(codec, "MAX_RAW_BYTES", 1024 * 1024)
    monkeypatch.setattr(codec, "MAX_COMPRESSED_BYTES", 20)
    with pytest.raises(codec.TransferError, match="Compressed"):
        codec.encode_bundle({})
    monkeypatch.setattr(codec, "MAX_COMPRESSED_BYTES", 1024 * 1024)
    monkeypatch.setattr(codec, "MAX_PARTS", 1)
    with pytest.raises(codec.TransferError, match="more than 1 parts"):
        codec.encode_bundle({"x": hashlib.shake_256(b"many parts").hexdigest(2000)}, max_bytes=1024)


@pytest.mark.parametrize("value", [None, [], {"x": float("nan")}, {"x": float("inf")},
                                  {"x": float("-inf")}, {"x": b"binary"}, {1: "int key"},
                                  {"nested": {False: "bool key"}}])
def test_encode_accepts_only_strict_json_objects(value):
    with pytest.raises(codec.TransferError):
        codec.encode_bundle(value)


def test_circular_input_is_rejected():
    value = {}
    value["self"] = value
    with pytest.raises(codec.TransferError, match="Circular"):
        codec.encode_bundle(value)


@pytest.mark.parametrize("raw", [b"not JSON", b"[1,2]", b"null", b'{"x":NaN}',
                                 b'{"x":Infinity}', b'{"x":1e10000}', b'{"x":1,"x":2}',
                                 b'{"x":"\xff"}'])
def test_decode_rejects_invalid_nonfinite_or_duplicate_json(raw):
    with pytest.raises(codec.TransferError):
        codec.decode_bundle(lzma.compress(raw))


def test_decode_rejects_trailing_data_concatenated_or_truncated_streams():
    stream = lzma.compress(b'{}')
    for bad in (stream[:-1], stream + b"x", stream + stream, b"not XZ", b""):
        with pytest.raises(codec.TransferError):
            codec.decode_bundle(bad)


def test_decompression_bomb_rejected_without_unbounded_output(monkeypatch):
    compressed = lzma.compress(b'{"x":"' + b"x" * 1000000 + b'"}')
    assert len(compressed) < 1000
    monkeypatch.setattr(codec, "MAX_RAW_BYTES", 1024)
    with pytest.raises(codec.TransferError, match="Decompressed bundle exceeds"):
        codec.decode_bundle(compressed)


def test_decompression_memory_limit_enforced(monkeypatch):
    compressed = lzma.compress(b'{}', preset=6)
    monkeypatch.setattr(codec, "DECOMPRESS_MEMLIMIT", 1024 * 1024)
    with pytest.raises(codec.TransferError, match="memory limit"):
        codec.decode_bundle(compressed)


def test_decode_compressed_size_limit(monkeypatch):
    compressed = lzma.compress(b'{}')
    monkeypatch.setattr(codec, "MAX_COMPRESSED_BYTES", len(compressed) - 1)
    with pytest.raises(codec.TransferError, match="Compressed bundle exceeds"):
        codec.decode_bundle(compressed)


def test_decode_checks_optional_raw_metadata(transfer):
    _, parts = transfer
    status, compressed = codec.inspect_parts(parts)
    with pytest.raises(codec.TransferError, match="byte count differs"):
        codec.decode_bundle(compressed, expected_raw_bytes=status["raw_bytes"] + 1)
    with pytest.raises(codec.TransferError, match="SHA256 differs"):
        codec.decode_bundle(compressed, expected_raw_sha256="0" * 64)
    for value in (False, "2", 0, 128 * 1024 * 1024 + 1):
        with pytest.raises(codec.TransferError, match="Invalid expected"):
            codec.decode_bundle(compressed, expected_raw_bytes=value)
    with pytest.raises(codec.TransferError, match="expected raw SHA256"):
        codec.decode_bundle(compressed, expected_raw_sha256=123)


def test_exact_decompression_size_limit_accepts_valid_bundle(monkeypatch):
    raw = b'{"x":123}'
    monkeypatch.setattr(codec, "MAX_RAW_BYTES", len(raw))
    assert codec.decode_bundle(lzma.compress(raw)) == {"x": 123}
