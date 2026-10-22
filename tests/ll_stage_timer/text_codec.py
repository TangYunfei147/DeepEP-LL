"""Checksummed, lossless text transport for already captured timer data.

Only standard-library compression and encoding are used.  This module neither
reads files nor interprets the payload; its caller owns the report schema.
"""

import base64
import binascii
import hashlib
import lzma
import re


FORMAT = "deepep-ll-text-v1"
CODEC = "lzma-xz+base64"
BEGIN = "-----BEGIN DEEPEP LL TEXT V1-----"
END = "-----END DEEPEP LL TEXT V1-----"
DEFAULT_LIMIT = 60000
MAX_LIMIT = 65535
MIN_LIMIT = 4096
MAX_RAW_BYTES = 512 * 1024 * 1024
MAX_COMPRESSED_BYTES = MAX_RAW_BYTES + 1024 * 1024
DECOMPRESS_MEMLIMIT = 256 * 1024 * 1024
LINE_WIDTH = 76
MAX_PARTS = 999999
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_SIZE = re.compile(r"[0-9]{12}\Z")
_PART = re.compile(r"([0-9]{6})/([0-9]{6})\Z")
_KEYS = frozenset(("format", "codec", "raw_bytes", "raw_sha256",
                   "compressed_bytes", "compressed_sha256", "part",
                   "chunk_bytes", "chunk_sha256"))


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _envelope(metadata, chunk, part, total):
    encoded = base64.b64encode(chunk).decode("ascii")
    lines = [BEGIN, f"format: {FORMAT}", f"codec: {CODEC}",
             f"raw_bytes: {metadata['raw_bytes']:012d}",
             f"raw_sha256: {metadata['raw_sha256']}",
             f"compressed_bytes: {metadata['compressed_bytes']:012d}",
             f"compressed_sha256: {metadata['compressed_sha256']}",
             f"part: {part:06d}/{total:06d}",
             f"chunk_bytes: {len(chunk):012d}",
             f"chunk_sha256: {_digest(chunk)}", "DATA"]
    lines.extend(encoded[i:i + LINE_WIDTH] for i in range(0, len(encoded), LINE_WIDTH))
    lines.append(END)
    return "\n".join(lines) + "\n"


def _crlf_bytes(text):
    """All output is ASCII; reserve the larger Windows newline representation."""
    return len(text) + text.count("\n")


def make_parts(payload: bytes, limit=DEFAULT_LIMIT):
    """Return manifest metadata and bounded ASCII envelopes for *payload*.

    The byte/character limit includes every header, footer and newline, with
    every newline counted as CRLF.  LZMA preserves all bytes, including Unicode
    encoded by the caller; there is no sampling or lossy summarization.
    """
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    if isinstance(limit, bool) or not isinstance(limit, int) or not MIN_LIMIT <= limit <= MAX_LIMIT:
        raise ValueError(f"limit must be an integer in [{MIN_LIMIT}, {MAX_LIMIT}]")
    if len(payload) > MAX_RAW_BYTES:
        raise ValueError("raw payload exceeds the 512 MiB safety limit")
    compressed = lzma.compress(payload, format=lzma.FORMAT_XZ, preset=6)
    if len(compressed) > MAX_COMPRESSED_BYTES:
        raise ValueError("compressed payload exceeds the safety limit")
    metadata = dict(format=FORMAT, codec=CODEC, raw_bytes=len(payload),
                    raw_sha256=_digest(payload), compressed_bytes=len(compressed),
                    compressed_sha256=_digest(compressed), limit=limit)
    # Fixed-width part/size fields make this exact for every part, including
    # the eventual total count.  Binary search avoids a newline rounding guess.
    low, high = 1, min(limit, len(compressed))
    capacity = 0
    while low <= high:
        middle = (low + high) // 2
        candidate = _envelope(metadata, compressed[:middle], 1, 1)
        if _crlf_bytes(candidate) <= limit:
            capacity = middle
            low = middle + 1
        else:
            high = middle - 1
    if not capacity:
        raise ValueError("limit cannot hold one encoded byte and its envelope")
    count = (len(compressed) + capacity - 1) // capacity
    if count > MAX_PARTS:
        raise ValueError("too many transfer parts")
    parts = [_envelope(metadata, compressed[start:start + capacity], index + 1, count)
             for index, start in enumerate(range(0, len(compressed), capacity))]
    metadata["part_count"] = count
    metadata["max_part_crlf_bytes"] = max(map(_crlf_bytes, parts))
    return metadata, parts


def _blocks(text):
    if not isinstance(text, str):
        raise TypeError("each transfer input must be text")
    # A BOM is accepted at file/block boundaries, never inside a header or data.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    position = 0
    found = False
    while position < len(text):
        while position < len(text) and text[position] in " \t\n\ufeff":
            position += 1
        if position == len(text):
            break
        if not text.startswith(BEGIN + "\n", position):
            raise ValueError("missing or invalid transfer envelope")
        stop = text.find("\n" + END, position + len(BEGIN))
        if stop < 0:
            raise ValueError("truncated transfer envelope: missing END marker")
        finish = stop + 1 + len(END)
        if finish < len(text) and text[finish] not in "\n \t\ufeff":
            raise ValueError("invalid transfer envelope footer")
        block = text[position:finish] + "\n"
        if not block.isascii():
            raise ValueError("transfer envelope must contain ASCII only")
        if _crlf_bytes(block) > MAX_LIMIT:
            raise ValueError("transfer envelope exceeds the 65535-byte safety limit")
        found = True
        yield block
        position = finish
    if not found:
        raise ValueError("no transfer envelope found")


def _read_block(block):
    interior = block[len(BEGIN) + 1:-(len(END) + 2)]
    if "\nDATA\n" not in interior:
        raise ValueError("transfer envelope is missing DATA separator")
    header, encoded = interior.split("\nDATA\n", 1)
    fields = {}
    for line in header.splitlines():
        key, separator, value = line.partition(": ")
        if not separator or key not in _KEYS or key in fields:
            raise ValueError("invalid or duplicate transfer header field")
        fields[key] = value
    if set(fields) != _KEYS or fields["format"] != FORMAT or fields["codec"] != CODEC:
        raise ValueError("unsupported or incomplete transfer header")
    for key in ("raw_sha256", "compressed_sha256", "chunk_sha256"):
        if not _SHA.fullmatch(fields[key]):
            raise ValueError(f"invalid {key}")
    for key in ("raw_bytes", "compressed_bytes", "chunk_bytes"):
        if not _SIZE.fullmatch(fields[key]):
            raise ValueError(f"invalid {key}")
        fields[key] = int(fields[key])
    if fields["raw_bytes"] > MAX_RAW_BYTES:
        raise ValueError("declared raw payload exceeds the 512 MiB safety limit")
    if not 0 < fields["compressed_bytes"] <= MAX_COMPRESSED_BYTES:
        raise ValueError("declared compressed payload exceeds the safety limit")
    if not 0 < fields["chunk_bytes"] <= fields["compressed_bytes"]:
        raise ValueError("invalid compressed chunk size")
    match = _PART.fullmatch(fields["part"])
    if not match:
        raise ValueError("invalid part number")
    part, count = map(int, match.groups())
    if not 1 <= part <= count <= min(MAX_PARTS, fields["compressed_bytes"]):
        raise ValueError("invalid part number or count")
    # Whitespace changes caused by copy/paste are harmless.  All remaining
    # bytes must be canonical Base64; validate=True rejects punctuation.
    encoded = "".join(encoded.split())
    try:
        chunk = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("invalid Base64 chunk") from error
    if base64.b64encode(chunk).decode("ascii") != encoded:
        raise ValueError("noncanonical Base64 chunk")
    if len(chunk) != fields["chunk_bytes"]:
        raise ValueError("compressed chunk size mismatch (possibly truncated)")
    if _digest(chunk) != fields["chunk_sha256"]:
        raise ValueError("compressed chunk SHA256 mismatch")
    metadata = {key: fields[key] for key in ("format", "codec", "raw_bytes",
                "raw_sha256", "compressed_bytes", "compressed_sha256")}
    metadata["part_count"] = count
    return metadata, part, chunk


def inspect_parts(texts):
    """Validate received envelopes, then return ``(status, payload_or_none)``.

    Missing parts are reported without attempting partial decompression.
    Identical repeats and arbitrary order are accepted.  Corruption, conflicting
    parts, mixed bundles and excessive allocation requests raise ValueError.
    """
    if isinstance(texts, str):
        texts = (texts,)
    metadata = None
    chunks = {}
    duplicates = 0
    received_bytes = 0
    for text in texts:
        for block in _blocks(text):
            current, number, chunk = _read_block(block)
            if metadata is None:
                metadata = current
            elif current != metadata:
                raise ValueError("mixed transfer bundles or conflicting bundle metadata")
            if number in chunks:
                if chunks[number] != chunk:
                    raise ValueError(f"conflicting duplicate part {number}")
                duplicates += 1
            else:
                chunks[number] = chunk
                received_bytes += len(chunk)
                if received_bytes > metadata["compressed_bytes"]:
                    raise ValueError("received chunks exceed declared compressed size")
    if metadata is None:
        raise ValueError("no transfer envelope found")
    missing = [number for number in range(1, metadata["part_count"] + 1) if number not in chunks]
    status = dict(metadata, complete=not missing, received_parts=sorted(chunks),
                  missing_parts=missing, duplicate_parts=duplicates)
    if missing:
        return status, None
    compressed = b"".join(chunks[number] for number in range(1, metadata["part_count"] + 1))
    if len(compressed) != metadata["compressed_bytes"]:
        raise ValueError("complete compressed payload size mismatch")
    if _digest(compressed) != metadata["compressed_sha256"]:
        raise ValueError("complete compressed payload SHA256 mismatch")
    decoder = lzma.LZMADecompressor(format=lzma.FORMAT_XZ, memlimit=DECOMPRESS_MEMLIMIT)
    try:
        # The advertised size is bounded above; one extra byte detects a lying
        # size without allocating an unbounded decompressed output.
        payload = decoder.decompress(compressed, max_length=metadata["raw_bytes"] + 1)
    except lzma.LZMAError as error:
        raise ValueError(f"LZMA decompression failed: {error}") from error
    if len(payload) != metadata["raw_bytes"]:
        raise ValueError("decompressed payload size mismatch or safety limit exceeded")
    if not decoder.eof:
        raise ValueError("truncated or excessive LZMA stream")
    if decoder.unused_data:
        raise ValueError("trailing data or additional compressed streams are not allowed")
    if _digest(payload) != metadata["raw_sha256"]:
        raise ValueError("decompressed payload SHA256 mismatch")
    return status, payload


def group_parts(texts):
    """Group complete envelopes by compressed SHA256 in first-seen order.

    This is a framing helper for a suite containing independent bundles.  It
    validates each envelope/chunk, but callers must run :func:`inspect_parts`
    on every returned group before restoring anything.  In particular, using
    only the compressed hash for grouping deliberately keeps contradictory raw
    metadata or part counts together so the strict decoder rejects them.
    """
    if isinstance(texts, str):
        texts = (texts,)
    groups = {}
    for text in texts:
        for block in _blocks(text):
            metadata, _, _ = _read_block(block)
            groups.setdefault(metadata["compressed_sha256"], []).append(block)
    if not groups:
        raise ValueError("no transfer envelope found")
    return list(groups.values())
