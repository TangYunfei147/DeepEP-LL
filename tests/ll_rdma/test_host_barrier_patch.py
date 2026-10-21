"""CPU checks of patch admission and asynchronous dissemination semantics."""
from pathlib import Path
import random
import shutil
import subprocess

import pytest

from scripts import shmem_barrier_patch as patch

ROOT = Path(__file__).resolve().parents[2]


def test_real_helper_against_cpu_transport_doubles(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("C++17 compiler required for the actual Host barrier helper")
    binary = tmp_path / "host-barrier.exe"
    subprocess.run([compiler, "-std=c++17", "-pthread", "-I" + str(ROOT),
                    str(Path(__file__).with_name("host_barrier_cpu.cpp")), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True, timeout=60)


def test_pinned_source_overlay_is_exact_and_idempotent(tmp_path, monkeypatch):
    source = ROOT / "third-party/shmem"
    if not (source / patch.TARGET).exists():
        pytest.skip("Initialize locked SHMEM to test its exact source overlay")
    original, expected = patch.expected_source(source)
    target = tmp_path / patch.TARGET
    target.parent.mkdir(parents=True)
    target.write_text(original, encoding="utf-8")
    monkeypatch.setattr(patch, "expected_source", lambda _: (original, expected))
    patch.apply(tmp_path)
    assert target.read_text(encoding="utf-8") == expected
    patch.apply(tmp_path)
    target.write_text(expected + "\n// foreign edit\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unknown edits"):
        patch.apply(tmp_path)
    assert target.read_text(encoding="utf-8").endswith("// foreign edit\n")


def test_overlay_keeps_public_device_barriers_out_of_scope():
    original = "// kernels\nACLSHMEM_GLOBAL void barrier_on_stream_kernel() {\n    aclshmemi_sync<false>(team);\n}\n"
    result = patch.patched_source(original)
    assert result.count("    deepep_host_barrier(team);") == 1
    assert "aclshmemi_sync<false>(tid)" in result  # unchanged all-MTE fallback
    assert "roce_qp_put_nbi" in result and "roce_qp_quiet" in result
    assert "roce_get" not in result and "atomic_set" not in result
    with pytest.raises(RuntimeError, match="layout differs"):
        patch.patched_source("unexpected revision")


def test_overlay_is_packaged_and_identity_is_line_ending_independent(tmp_path, monkeypatch):
    assert "recursive-include scripts/patches *.inc" in (ROOT / "MANIFEST.in").read_text()
    expected = patch.patch_identity()
    payload = tmp_path / "patch with spaces.inc"
    payload.write_bytes(patch.PAYLOAD.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    monkeypatch.setattr(patch, "PAYLOAD", payload)
    assert patch.patch_identity() == expected


@pytest.mark.parametrize("size", [2, 3, 16, 17, 32, 64, 128, 256])
@pytest.mark.parametrize("seed", range(4))
def test_dissemination_with_delays_and_repeated_generations(size, seed):
    """No rank may finish a generation before every rank entered it.

    Random delivery models delayed RDMA; send completion precedes counter reuse.
    Explicit shuffled team membership exercises mapping without rank arithmetic.
    """
    rng = random.Random(seed)
    members = rng.sample(range(1024), size)
    rounds = (size - 1).bit_length()
    slots = [[0] * rounds for _ in members]
    generation = [0] * size
    phase = [0] * size
    state = ["enter"] * size
    entered = [set() for _ in range(9)]
    pending = []
    steps = 0
    while any(g < 8 or s != "done" for g, s in zip(generation, state)):
        steps += 1
        assert steps < size * 8 * rounds * 300
        if pending and rng.randrange(3) == 0:
            sender, dest, slot, value = pending.pop(rng.randrange(len(pending)))
            slots[dest][slot] = value
            assert generation[sender] == value  # quiet protects source counter
            state[sender] = "wait"
            continue
        rank = rng.randrange(size)
        if state[rank] == "enter":
            generation[rank] += 1
            entered[generation[rank]].add(rank)
            phase[rank] = 0
            state[rank] = "send"
        elif state[rank] == "send":
            dest = (rank + (1 << phase[rank])) % size
            peer = members[dest]
            assert members.index(peer) == dest
            pending.append((rank, dest, phase[rank], generation[rank]))
            state[rank] = "quiet"
        elif state[rank] == "wait":
            if slots[rank][phase[rank]] not in (generation[rank], generation[rank] + 1):
                continue
            phase[rank] += 1
            if phase[rank] == rounds:
                assert len(entered[generation[rank]]) == size
                state[rank] = "done" if generation[rank] == 8 else "enter"
            else:
                state[rank] = "send"
