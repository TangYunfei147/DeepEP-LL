"""CPU production-helper checks: no CANN or device work."""
from pathlib import Path
import os
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_sender_route_plan_conserves_q_and_bounds(tmp_path):
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if not zig:
        pytest.skip("Set DEEPEP_TEST_ZIG for the native sender plan test")
    exe = tmp_path / ("sender-route-plan.exe" if os.name == "nt" else "sender-route-plan")
    subprocess.run([zig, "c++", "-std=c++17", "-O2", "-UNDEBUG",
                    "-I" + str(ROOT / "csrc"),
                    str(Path(__file__).with_name("sender_route_plan_test.cpp")),
                    "-o", str(exe)], check=True, timeout=180)
    result = subprocess.run([str(exe)], check=True, capture_output=True, text=True, timeout=30)
    assert "PASS sender route plan" in result.stdout
    print(result.stdout)
