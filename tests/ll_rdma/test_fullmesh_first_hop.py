from pathlib import Path
import os
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_fullmesh_first_hop_conserves_routes_and_minimizes_clos(tmp_path):
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if not zig:
        pytest.skip("Set DEEPEP_TEST_ZIG for the native policy test")
    exe = tmp_path / "fullmesh-first-hop.exe"
    subprocess.run([zig, "c++", "-std=c++17", "-O2", "-UNDEBUG",
                    "-I" + str(ROOT / "csrc"),
                    str(Path(__file__).with_name("fullmesh_first_hop_test.cpp")),
                    "-o", str(exe)], check=True, timeout=180)
    result = subprocess.run([str(exe)], check=True, capture_output=True, text=True, timeout=30)
    assert "PASS fullmesh first-hop" in result.stdout
    print(result.stdout)


def test_sender_scopes_policy_and_keeps_earlier_members():
    code = (ROOT / "csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h").read_text(encoding="utf-8")
    assert "Group == 16U && args->rankSize == 32 && LlTransport::Rdma(args)" in code
    assert "mesh.Add(destination, k)" in code
    assert "LlRelayMemberMask(mask, firstK)" in code
    assert "firstK + 1U" not in code
    assert "for (uint32_t k = 0U; members != 0U && k < k_; ++k)" in code
