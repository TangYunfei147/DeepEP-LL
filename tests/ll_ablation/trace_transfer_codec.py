"""Bounded, lossless ASCII transport for offline LL trace bundles.

This module has no filesystem access. SHA256 detects transfer corruption; it
does not authenticate the sender. A bundle is one strict JSON object compressed
as one XZ stream, split into independently checked base64 blocks. Never decode a
partial transfer or extract files before the caller has validated their paths.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import lzma
import math
import re


FORMAT = "ll_trace_text_v1"
CODEC = "xz-lzma2+base64"
BEGIN = "-----BEGIN LL_TRACE_TEXT_V1-----"
END = "-----END LL_TRACE_TEXT_V1-----"
DEFAULT_MAX_BYTES = 60000
MIN_MAX_BYTES = 1024
MAX_MAX_BYTES = 65535
MAX_COMPRESSED_BYTES = 128 * 1024 * 1024
MAX_RAW_BYTES = 128 * 1024 * 1024
DECOMPRESS_MEMLIMIT = 256 * 1024 * 1024
MAX_PARTS = 65536
MAX_BLOCKS = MAX_PARTS * 2
MAX_PART_BYTES = 49152
MAX_BLOCK_TEXT_BYTES = MAX_PART_BYTES * 8 + 4096
MAX_INPUT_TEXT_BYTES = MAX_COMPRESSED_BYTES * 4
BASE64_COLUMNS = 120


class TransferError(ValueError):
    """Malformed, inconsistent, oversized, or corrupt trace transport."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _integer(value: str, name: str, maximum: int) -> int:
    if not re.fullmatch(r"[1-9][0-9]{0,8}", value):
        raise TransferError(f"Invalid {name}: expected a positive bounded integer")
    number = int(value)
    if number > maximum:
        raise TransferError(f"{name} exceeds limit {maximum}")
    return number


def _digest(value: str, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise TransferError(f"Invalid {name}: expected lowercase SHA256")
    return value


def _strict_json_bytes(bundle: dict) -> bytes:
    if not isinstance(bundle, dict):
        raise TransferError("Trace bundle must be a JSON object")
    pending, seen = [bundle], set()
    while pending:
        value = pending.pop()
        if isinstance(value, (dict, list, tuple)):
            if id(value) in seen:
                continue  # JSONEncoder handles actual circular references.
            seen.add(id(value))
            if isinstance(value, dict):
                if any(not isinstance(key, str) for key in value):
                    raise TransferError("JSON object keys must be strings")
                pending.extend(value.values())
            else:
                pending.extend(value)
    encoder = json.JSONEncoder(ensure_ascii=True, allow_nan=False,
                              sort_keys=True, separators=(",", ":"))
    result = bytearray()
    total = 0
    try:
        for chunk in encoder.iterencode(bundle):
            raw = chunk.encode("ascii")
            total += len(raw)
            if total > MAX_RAW_BYTES:
                raise TransferError("Uncompressed bundle exceeds limit")
            result.extend(raw)
    except (TypeError, ValueError, RecursionError) as error:
        if isinstance(error, TransferError):
            raise
        raise TransferError(f"Bundle is not strict JSON: {error}") from error
    return bytes(result)


def _block(meta: dict, index: int, part_count: int, part: bytes) -> str:
    encoded = base64.b64encode(part).decode("ascii")
    body = "\n".join(encoded[pos:pos + BASE64_COLUMNS]
                     for pos in range(0, len(encoded), BASE64_COLUMNS))
    return (
        f"{BEGIN}\n"
        f"format: {FORMAT}\n"
        f"codec: {CODEC}\n"
        f"bundle-sha256: {meta['bundle_sha256']}\n"
        f"raw-sha256: {meta['raw_sha256']}\n"
        f"compressed-bytes: {meta['compressed_bytes']}\n"
        f"raw-bytes: {meta['raw_bytes']}\n"
        f"part: {index}/{part_count}\n"
        f"part-sha256: {_sha(part)}\n"
        f"part-bytes: {len(part)}\n\n"
        f"{body}\n{END}\n"
    )


def _crlf_size(text: str) -> int:
    # All generated text is ASCII/LF, so this also bounds UTF-16 code units.
    return len(text) + text.count("\n")


def _payload_capacity(meta: dict, max_bytes: int) -> int:
    # Reserve the longest legal part index/count even for a one-part bundle.
    # Binary search includes line wrapping and the two-byte CRLF representation.
    low, high = 0, min(MAX_PART_BYTES, max_bytes)
    while low < high:
        middle = (low + high + 1) // 2
        size = _crlf_size(_block(meta, MAX_PARTS, MAX_PARTS, b"x" * middle))
        if size <= max_bytes:
            low = middle
        else:
            high = middle - 1
    if low < 1:
        raise TransferError("Text limit leaves no room for a payload")
    return low


def encode_bundle(bundle: dict, max_bytes: int = DEFAULT_MAX_BYTES) -> tuple[dict, list[str]]:
    """Return a manifest and ASCII blocks, each bounded even with CRLF lines."""
    if (isinstance(max_bytes, bool) or not isinstance(max_bytes, int)
            or not MIN_MAX_BYTES <= max_bytes <= MAX_MAX_BYTES):
        raise TransferError(f"max_bytes must be in {MIN_MAX_BYTES}..{MAX_MAX_BYTES}")
    raw = _strict_json_bytes(bundle)
    compressed = lzma.compress(raw, format=lzma.FORMAT_XZ,
                               check=lzma.CHECK_CRC64, preset=6)
    if len(compressed) > MAX_COMPRESSED_BYTES:
        raise TransferError("Compressed bundle exceeds limit")
    meta = dict(format=FORMAT, codec=CODEC, bundle_sha256=_sha(compressed),
                raw_sha256=_sha(raw), compressed_bytes=len(compressed),
                raw_bytes=len(raw))
    capacity = _payload_capacity(meta, max_bytes)
    part_count = (len(compressed) + capacity - 1) // capacity
    if part_count > MAX_PARTS:
        raise TransferError(f"Bundle requires more than {MAX_PARTS} parts")
    parts, entries = [], []
    for pos in range(part_count):
        data = compressed[pos * capacity:(pos + 1) * capacity]
        block = _block(meta, pos + 1, part_count, data)
        worst_size = _crlf_size(block)
        if worst_size > max_bytes:  # Keep the public invariant explicit.
            raise TransferError("Internal block size calculation failed")
        parts.append(block)
        entries.append(dict(index=pos + 1, sha256=_sha(data),
                            compressed_bytes=len(data), text_bytes=len(block),
                            crlf_bytes=worst_size))
    manifest = dict(meta, part_count=part_count, max_bytes=max_bytes, parts=entries)
    return manifest, parts


_HEADER_KEYS = frozenset(("format", "codec", "bundle-sha256", "raw-sha256",
                          "compressed-bytes", "raw-bytes", "part",
                          "part-sha256", "part-bytes"))


def _parse_block(text: str) -> tuple[dict, int, bytes]:
    if len(text) > MAX_BLOCK_TEXT_BYTES:
        raise TransferError("Text block exceeds parsing limit")
    content = text[len(BEGIN):-len(END)].replace("\r\n", "\n").replace("\r", "\n")
    if not content.startswith("\n") or not content.endswith("\n"):
        raise TransferError("BEGIN/END markers must occupy complete lines")
    content = content[1:-1]
    # Header order is immaterial; duplicate/missing/unknown fields are rejected.
    header_lines, separator, body = content.partition("\n\n")
    if not separator:
        raise TransferError("Missing header/payload separator")
    fields = {}
    for line in header_lines.split("\n"):
        key, colon, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if not colon or key not in _HEADER_KEYS or key in fields:
            raise TransferError("Invalid or duplicate transport header field")
        fields[key] = value
    if fields.keys() != _HEADER_KEYS:
        raise TransferError("Missing transport header field")
    if fields["format"] != FORMAT or fields["codec"] != CODEC:
        raise TransferError("Unsupported trace transfer format or codec")
    index_text, slash, count_text = fields["part"].partition("/")
    if not slash:
        raise TransferError("Invalid part index/count")
    index = _integer(index_text, "part index", MAX_PARTS)
    count = _integer(count_text, "part count", MAX_PARTS)
    if index > count:
        raise TransferError("Part index is greater than part count")
    compressed_size = _integer(fields["compressed-bytes"], "compressed bytes",
                               MAX_COMPRESSED_BYTES)
    raw_size = _integer(fields["raw-bytes"], "raw bytes", MAX_RAW_BYTES)
    part_size = _integer(fields["part-bytes"], "part bytes", MAX_PART_BYTES)
    if part_size > compressed_size or count > compressed_size:
        raise TransferError("Impossible compressed size / part count")
    if compressed_size > count * MAX_PART_BYTES:
        raise TransferError("Part count cannot contain the declared compressed bytes")
    meta = dict(format=FORMAT, codec=CODEC,
                bundle_sha256=_digest(fields["bundle-sha256"], "bundle SHA256"),
                raw_sha256=_digest(fields["raw-sha256"], "raw SHA256"),
                compressed_bytes=compressed_size, raw_bytes=raw_size,
                part_count=count)
    part_sha = _digest(fields["part-sha256"], "part SHA256")
    # Only ASCII whitespace is ignored, allowing pasted/wrapped payload lines.
    encoded = re.sub(r"[ \t\n\r\v\f]", "", body)
    if len(encoded) != 4 * ((part_size + 2) // 3):
        raise TransferError("Base64 payload size differs from declared part bytes")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as error:
        raise TransferError("Invalid base64 payload") from error
    if len(data) != part_size or _sha(data) != part_sha:
        raise TransferError("Part length or SHA256 mismatch")
    # Reject noncanonical padding bits as well as ignored extra padding.
    if base64.b64encode(data).decode("ascii") != encoded:
        raise TransferError("Noncanonical base64 payload")
    return meta, index, data


def inspect_parts(texts: list[str]) -> tuple[dict, bytes | None]:
    """Check possibly unordered/repeated blocks; return None while parts are missing.

    Multiple blocks per input string are accepted. Other than whitespace, text
    outside complete BEGIN/END blocks is rejected, including partial blocks.
    """
    if not isinstance(texts, list):
        raise TransferError("texts must be a list of strings")
    if len(texts) > MAX_BLOCKS:
        raise TransferError("Too many input texts")
    total_text = 0
    for text in texts:
        if not isinstance(text, str):
            raise TransferError("Each transport input must be text")
        total_text += len(text)
        if total_text > MAX_INPUT_TEXT_BYTES:
            raise TransferError("Cumulative input text exceeds limit")
        if not text.isascii():
            raise TransferError("Trace transport must contain ASCII only")

    metadata = None
    unique, duplicates = {}, []
    blocks_seen, bytes_seen = 0, 0
    for text in texts:
        pos = 0
        while pos < len(text):
            start = text.find(BEGIN, pos)
            if start < 0:
                if text[pos:].strip():
                    raise TransferError("Missing BEGIN marker or text outside blocks")
                break
            if text[pos:start].strip():
                raise TransferError("Unexpected text outside trace blocks")
            finish = text.find(END, start + len(BEGIN))
            if finish < 0:
                raise TransferError("Missing END marker: truncated transport block")
            nested = text.find(BEGIN, start + len(BEGIN), finish)
            if nested >= 0:
                raise TransferError("Nested BEGIN marker: incomplete transport block")
            end = finish + len(END)
            if start > 0 and text[start - 1] not in "\r\n":
                raise TransferError("BEGIN marker must occupy a complete line")
            if end < len(text) and text[end] not in "\r\n":
                raise TransferError("END marker must occupy a complete line")
            blocks_seen += 1
            if blocks_seen > MAX_BLOCKS:
                raise TransferError("Too many transport blocks")
            meta, index, data = _parse_block(text[start:end])
            if metadata is None:
                metadata = meta
            elif metadata != meta:
                raise TransferError("Mixed bundles or conflicting transport metadata")
            if index in unique:
                if unique[index] != data:
                    raise TransferError(f"Conflicting duplicate part {index}")
                duplicates.append(index)
            else:
                unique[index] = data
                bytes_seen += len(data)
                if bytes_seen > metadata["compressed_bytes"] or bytes_seen > MAX_COMPRESSED_BYTES:
                    raise TransferError("Cumulative payload exceeds declared compressed bytes")
            pos = end
    if metadata is None:
        raise TransferError("No complete trace blocks found")
    missing = [index for index in range(1, metadata["part_count"] + 1)
               if index not in unique]
    status = dict(metadata, status="missing" if missing else "complete",
                  received_parts=sorted(unique), missing_parts=missing,
                  duplicate_count=len(duplicates), duplicate_parts=duplicates,
                  received_compressed_bytes=bytes_seen,
                  parts=[dict(index=index, sha256=_sha(data), compressed_bytes=len(data))
                         for index, data in sorted(unique.items())])
    if missing:
        # An incomplete set must leave at least one byte per remaining part.
        if bytes_seen + len(missing) > metadata["compressed_bytes"]:
            raise TransferError("Missing parts cannot fit the declared compressed bytes")
        if bytes_seen + len(missing) * MAX_PART_BYTES < metadata["compressed_bytes"]:
            raise TransferError("Missing parts cannot contain the declared compressed bytes")
        return status, None
    if bytes_seen != metadata["compressed_bytes"]:
        raise TransferError("Complete transfer has a compressed length mismatch")
    compressed = b"".join(unique[index] for index in range(1, metadata["part_count"] + 1))
    if _sha(compressed) != metadata["bundle_sha256"]:
        raise TransferError("Bundle SHA256 mismatch")
    return status, compressed


def _json_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise TransferError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise TransferError(f"Non-finite JSON value: {value}")


def decode_bundle(compressed: bytes, *, expected_raw_bytes: int | None = None,
                  expected_raw_sha256: str | None = None) -> dict:
    """Decode exactly one bounded XZ stream containing a strict JSON object.

    Call inspect_parts first. Passing its raw size/hash also checks that header
    metadata matches the decoded bytes. Concatenated streams and trailing bytes
    are rejected rather than silently ignored.
    """
    if not isinstance(compressed, bytes) or not compressed:
        raise TransferError("compressed must be nonempty bytes")
    if len(compressed) > MAX_COMPRESSED_BYTES:
        raise TransferError("Compressed bundle exceeds limit")
    if expected_raw_bytes is not None and (
            isinstance(expected_raw_bytes, bool) or not isinstance(expected_raw_bytes, int)
            or not 1 <= expected_raw_bytes <= MAX_RAW_BYTES):
        raise TransferError("Invalid expected raw byte count")
    if expected_raw_sha256 is not None:
        _digest(expected_raw_sha256, "expected raw SHA256")
    try:
        decoder = lzma.LZMADecompressor(format=lzma.FORMAT_XZ,
                                       memlimit=DECOMPRESS_MEMLIMIT)
        raw = decoder.decompress(compressed, max_length=MAX_RAW_BYTES + 1)
        if len(raw) > MAX_RAW_BYTES:
            raise TransferError("Decompressed bundle exceeds limit")
        if not decoder.eof:
            # With all input supplied and max_length larger than our limit, a
            # bounded result lacking EOF is truncated, never a valid bundle.
            raise TransferError("Truncated or oversized XZ stream")
        if decoder.unused_data:
            raise TransferError("Trailing data or concatenated XZ streams")
    except lzma.LZMAError as error:
        raise TransferError(f"Invalid XZ stream or decoder memory limit: {error}") from error
    if expected_raw_bytes is not None and len(raw) != expected_raw_bytes:
        raise TransferError("Uncompressed byte count differs from transfer metadata")
    if expected_raw_sha256 is not None and _sha(raw) != expected_raw_sha256:
        raise TransferError("Uncompressed SHA256 differs from transfer metadata")
    try:
        result = json.loads(raw.decode("utf-8"), object_pairs_hook=_json_object,
                            parse_constant=_invalid_constant)
    except (UnicodeError, ValueError, RecursionError) as error:
        if isinstance(error, TransferError):
            raise
        raise TransferError(f"Decoded bundle is not strict JSON: {error}") from error
    if not isinstance(result, dict):
        raise TransferError("Decoded trace bundle must be a JSON object")
    # JSON's exponent syntax can overflow Python floats without parse_constant.
    def check_finite(value):
        if isinstance(value, float):
            if not math.isfinite(value):
                raise TransferError("Non-finite JSON number")
        elif isinstance(value, dict):
            for child in value.values():
                check_finite(child)
        elif isinstance(value, list):
            for child in value:
                check_finite(child)
    try:
        check_finite(result)
    except RecursionError as error:
        raise TransferError("JSON nesting exceeds supported depth") from error
    return result
