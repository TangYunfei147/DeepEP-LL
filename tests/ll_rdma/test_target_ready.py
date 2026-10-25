"""Loopback-only preparation handshake tests; no CANN, HCCL or devices."""

from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import socket
import struct
import time

import pytest


SPEC = importlib.util.spec_from_file_location("target_ready", Path(__file__).with_name("target_ready.py"))
ready = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ready)

IDENTITY = {"git": "example-commit", "sdk": {"revision": "sdk", "hns_1825_pi_on_chip": 0},
            "parameters": {"ep": 32, "batch": 8, "dtype": "bf16"}}


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def client(port):
    deadline = time.monotonic() + 2
    while True:
        try:
            return socket.create_connection(("127.0.0.1", port), timeout=0.2)
        except OSError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.01)


def test_matching_preparation_releases_both_nodes_and_returns_same_report():
    port = free_port()
    with ThreadPoolExecutor(max_workers=2) as workers:
        first = workers.submit(ready.synchronize, 0, "127.0.0.1", port, "run-001", IDENTITY, 2)
        second = workers.submit(ready.synchronize, 1, "127.0.0.1", port, "run-001", IDENTITY, 2)
        reports = [future.result(timeout=3) for future in (first, second)]
    assert reports[0] == reports[1]
    assert reports[0]["nodes"] == [{"node_rank": rank, "identity": IDENTITY} for rank in (0, 1)]
    assert reports[0]["run_id"] == "run-001" and reports[0]["status"] == "ready"


@pytest.mark.parametrize("other", [{"different": "configuration"}, {"value": True}, {"value": 1.0}])
def test_identity_mismatch_fails_both_nodes_including_json_type_difference(other):
    port = free_port()
    with ThreadPoolExecutor(max_workers=2) as workers:
        first = workers.submit(ready.synchronize, 0, "127.0.0.1", port, "same-run", {"value": 1}, 2)
        second = workers.submit(ready.synchronize, 1, "127.0.0.1", port, "same-run", other, 2)
        for future in (first, second):
            with pytest.raises(ready.TargetReadyError, match="identity mismatch"):
                future.result(timeout=3)


def test_wrong_run_id_fails_both_nodes():
    port = free_port()
    with ThreadPoolExecutor(max_workers=2) as workers:
        first = workers.submit(ready.synchronize, 0, "127.0.0.1", port, "one-run", IDENTITY, 2)
        second = workers.submit(ready.synchronize, 1, "127.0.0.1", port, "another-run", IDENTITY, 2)
        for future in (first, second):
            with pytest.raises(ready.TargetReadyError, match="run_id mismatch"):
                future.result(timeout=3)


@pytest.mark.parametrize("wire, error", [
    (b"\x00\x00", "Truncated.*header"),
    (struct.pack("!I", 20) + b"{}", "Truncated.*payload"),
    (struct.pack("!I", ready.MAX_FRAME_BYTES + 1), "frame length"),
    (struct.pack("!I", 0), "frame length"),
    (struct.pack("!I", 1) + b"x", "Invalid preparation JSON"),
])
def test_bad_or_truncated_frame_is_rejected_without_waiting_for_unbounded_payload(wire, error):
    port = free_port()
    with ThreadPoolExecutor(max_workers=1) as workers:
        server = workers.submit(ready.synchronize, 0, "127.0.0.1", port, "run", IDENTITY, 2)
        with client(port) as connection:
            connection.sendall(wire)
            connection.shutdown(socket.SHUT_WR)
            with pytest.raises(ready.TargetReadyError, match=error):
                server.result(timeout=3)


def test_listener_requires_node_one():
    port = free_port()
    payload = json.dumps({"protocol": ready.PROTOCOL, "type": "prepared", "node_rank": 0,
                          "run_id": "run", "identity": IDENTITY}).encode()
    with ThreadPoolExecutor(max_workers=1) as workers:
        server = workers.submit(ready.synchronize, 0, "127.0.0.1", port, "run", IDENTITY, 2)
        with client(port) as connection:
            connection.sendall(struct.pack("!I", len(payload)) + payload)
            with pytest.raises(ready.TargetReadyError, match="requires a node 1"):
                server.result(timeout=3)


@pytest.mark.parametrize("rank", [0, 1])
def test_absent_peer_has_bounded_clear_timeout(rank):
    start = time.monotonic()
    with pytest.raises(ready.TargetReadyError, match="timed out"):
        ready.synchronize(rank, "127.0.0.1", free_port(), "run", IDENTITY, timeout_seconds=0.15)
    assert time.monotonic() - start < 1.5


def test_node_one_can_start_before_the_listener():
    port = free_port()
    with ThreadPoolExecutor(max_workers=2) as workers:
        second = workers.submit(ready.synchronize, 1, "127.0.0.1", port, "run", IDENTITY, 2)
        time.sleep(0.15)
        first = workers.submit(ready.synchronize, 0, "127.0.0.1", port, "run", IDENTITY, 2)
        assert first.result(timeout=3) == second.result(timeout=3)


def test_connected_peer_cannot_extend_deadline_with_partial_frame():
    port = free_port()
    with ThreadPoolExecutor(max_workers=1) as workers:
        server = workers.submit(ready.synchronize, 0, "127.0.0.1", port, "run", IDENTITY, 0.25)
        with client(port) as connection:
            connection.sendall(struct.pack("!I", 100) + b"{")
            with pytest.raises(ready.TargetReadyError, match="timed out.*frame payload"):
                server.result(timeout=1)


@pytest.mark.parametrize("identity", [{"bad": float("nan")}, {1: "non-string key"}, {"bad": (1, 2)},
                                      {"too_large": "x" * ready.MAX_FRAME_BYTES}])
def test_invalid_local_identity_fails_before_network(identity):
    with pytest.raises(ready.TargetReadyError):
        ready.synchronize(0, "127.0.0.1", free_port(), "run", identity, 0.1)


@pytest.mark.parametrize("address", ["localhost", "", "unresolved.invalid", "fe80::1%interface"])
def test_no_hostname_resolution_can_escape_the_deadline(address):
    with pytest.raises(ValueError, match="numeric IPv4/IPv6"):
        ready.synchronize(0, address, 12345, "run", IDENTITY, 0.1)
