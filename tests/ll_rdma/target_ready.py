"""Two-node preparation gate before the existing HCCL/ST entry point.

This exchanges only build/configuration identities after local preparation. It
does not create a process group, synchronize an operator, or carry operator data
or routing metadata. No files are written and no remote commands are executed.
"""

import json
import ipaddress
import math
import socket
import struct
import time


PROTOCOL = "deepep-ll-target-ready-v1"
MAX_FRAME_BYTES = 64 * 1024


class TargetReadyError(RuntimeError):
    """Preparation identity exchange failed; do not launch the test."""


def _remaining(deadline, action):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TargetReadyError(f"Preparation handshake timed out while {action}")
    return remaining


def _json_bytes(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise TargetReadyError("Preparation identity must contain finite JSON values") from exc


def _validate_identity(value):
    # JSON must preserve the caller's types/keys; do not coerce tuples or numeric
    # dictionary keys into a different identity during serialization.
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise TargetReadyError("Preparation identity JSON object keys must be strings")
            _validate_identity(item)
    elif type(value) is list:
        for item in value:
            _validate_identity(item)
    elif value is not None and type(value) not in (str, int, float, bool):
        raise TargetReadyError("Preparation identity must contain JSON values only")


def _frame(value):
    payload = _json_bytes(value)
    if not 0 < len(payload) <= MAX_FRAME_BYTES:
        raise TargetReadyError(f"Preparation frame exceeds {MAX_FRAME_BYTES} bytes")
    return struct.pack("!I", len(payload)) + payload


def _send(sock, value, deadline):
    frame = _frame(value)
    sock.settimeout(_remaining(deadline, "sending preparation result"))
    try:
        sock.sendall(frame)
    except socket.timeout as exc:
        raise TargetReadyError("Preparation handshake timed out while sending a frame") from exc
    except OSError as exc:
        raise TargetReadyError(f"Preparation frame send failed: {exc}") from exc


def _read_exact(sock, size, deadline, label):
    chunks = bytearray()
    while len(chunks) < size:
        sock.settimeout(_remaining(deadline, f"reading {label}"))
        try:
            chunk = sock.recv(size - len(chunks))
        except socket.timeout as exc:
            raise TargetReadyError(f"Preparation handshake timed out while reading {label}") from exc
        except OSError as exc:
            raise TargetReadyError(f"Preparation {label} receive failed: {exc}") from exc
        if not chunk:
            raise TargetReadyError(f"Truncated preparation {label}: received {len(chunks)} of {size} bytes")
        chunks.extend(chunk)
    return bytes(chunks)


def _object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"Non-finite JSON constant: {value}")


def _receive(sock, deadline):
    size = struct.unpack("!I", _read_exact(sock, 4, deadline, "frame header"))[0]
    if not 0 < size <= MAX_FRAME_BYTES:
        raise TargetReadyError(f"Invalid preparation frame length {size}; maximum is {MAX_FRAME_BYTES} bytes")
    payload = _read_exact(sock, size, deadline, "frame payload")
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_object_pairs,
                           parse_constant=_invalid_constant)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise TargetReadyError("Invalid preparation JSON frame") from exc
    if type(value) is not dict:
        raise TargetReadyError("Preparation frame must be a JSON object")
    return value


def _connect(master_addr, port, family, deadline):
    last_error = None
    while True:
        remaining = _remaining(deadline, f"connecting to preparation listener {master_addr}:{port}")
        connection = socket.socket(family, socket.SOCK_STREAM)
        try:
            connection.settimeout(min(remaining, 1.0))
            connection.connect((master_addr, port))
            return connection
        except OSError as exc:
            connection.close()
            last_error = exc
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TargetReadyError(
                f"Preparation handshake timed out connecting to {master_addr}:{port}: {last_error}")
        time.sleep(min(0.1, remaining))


def synchronize(node_rank, master_addr, port, run_id, identity, timeout_seconds=1800):
    """Return matching preparation reports for nodes 0/1, or raise.

    Call once per node only after its local build/install/checks have succeeded.
    ``master_addr`` must be a numeric IPv4/IPv6 address so DNS cannot escape the
    deadline. The caller supplies only non-secret build/configuration facts.
    ``port`` is the separate, explicit preparation port (the launcher normally
    chooses master_port + 2). Node 1 retries connection establishment until the
    deadline; established exchanges and tests are never retried. Identity equality
    includes JSON types, so ``True``, ``1``, and ``1.0`` are distinct.
    """
    if type(node_rank) is not int or node_rank not in (0, 1):
        raise ValueError("Preparation node_rank must be 0 or 1")
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("Preparation port must be an explicit integer in [1, 65535]")
    if not isinstance(master_addr, str):
        raise ValueError("Preparation master_addr must be a numeric IPv4/IPv6 address")
    try:
        address = ipaddress.ip_address(master_addr)
        if getattr(address, "scope_id", None) is not None:
            raise ValueError("Scoped addresses are not supported")
    except ValueError as exc:
        raise ValueError("Preparation master_addr must be an unscoped numeric IPv4/IPv6 address") from exc
    master_addr = str(address)
    family = socket.AF_INET6 if address.version == 6 else socket.AF_INET
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("Preparation run_id must be non-empty")
    if type(identity) is not dict:
        raise ValueError("Preparation identity must be a dictionary")
    if (type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0):
        raise ValueError("Preparation timeout_seconds must be finite and positive")
    try:
        _validate_identity(identity)
    except RecursionError as exc:
        raise TargetReadyError("Preparation identity is cyclic or too deeply nested") from exc
    identity_bytes = _json_bytes(identity)
    snapshot = json.loads(identity_bytes)
    hello = {"protocol": PROTOCOL, "type": "prepared", "node_rank": 1,
             "run_id": run_id, "identity": snapshot}
    report = {"protocol": PROTOCOL, "run_id": run_id, "status": "ready",
              "nodes": [{"node_rank": rank, "identity": snapshot} for rank in (0, 1)]}
    success = {"protocol": PROTOCOL, "type": "result", "run_id": run_id,
               "ok": True, "report": report}
    # Fail locally before opening a socket if either protocol message is too big.
    _frame(hello)
    _frame(success)
    deadline = time.monotonic() + timeout_seconds
    if node_rank == 1:
        with _connect(master_addr, port, family, deadline) as connection:
            _send(connection, hello, deadline)
            result = _receive(connection, deadline)
        if result.get("protocol") != PROTOCOL or result.get("type") != "result":
            raise TargetReadyError("Invalid preparation result protocol")
        if result.get("ok") is False:
            error = result.get("error")
            if not isinstance(error, str) or len(error) > 1024:
                raise TargetReadyError("Peer rejected preparation with an invalid error response")
            raise TargetReadyError(f"Peer rejected preparation: {error}")
        if (result.get("ok") is not True or result.get("run_id") != run_id
                or _json_bytes(result.get("report")) != _json_bytes(report)):
            raise TargetReadyError("Preparation result does not match this run and identity")
        return report

    try:
        listener = socket.create_server((master_addr, port), family=family, backlog=1)
    except OSError as exc:
        raise TargetReadyError(f"Cannot listen on preparation address {master_addr}:{port}: {exc}") from exc
    with listener:
        listener.settimeout(_remaining(deadline, "waiting for node 1"))
        try:
            connection, _ = listener.accept()
        except socket.timeout as exc:
            raise TargetReadyError("Preparation handshake timed out waiting for node 1") from exc
        except OSError as exc:
            raise TargetReadyError(f"Preparation connection accept failed: {exc}") from exc
        with connection:
            try:
                peer = _receive(connection, deadline)
                if (set(peer) != set(hello) or peer.get("protocol") != PROTOCOL
                        or peer.get("type") != "prepared" or type(peer.get("node_rank")) is not int
                        or peer["node_rank"] != 1):
                    raise TargetReadyError("Preparation listener requires a node 1 prepared message")
                if peer.get("run_id") != run_id:
                    raise TargetReadyError("Preparation run_id mismatch")
                if type(peer.get("identity")) is not dict or _json_bytes(peer["identity"]) != identity_bytes:
                    raise TargetReadyError("Preparation identity mismatch")
                _send(connection, success, deadline)
            except TargetReadyError as exc:
                failure = {"protocol": PROTOCOL, "type": "result", "run_id": run_id,
                           "ok": False, "error": str(exc)[:1024]}
                try:
                    _send(connection, failure, deadline)
                except TargetReadyError:
                    pass
                raise
    return report
