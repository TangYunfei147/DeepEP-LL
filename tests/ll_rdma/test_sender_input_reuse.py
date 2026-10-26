"""CPU production cache lifecycle checks; no CANN or device work."""
from pathlib import Path
import os
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_sender_input_reuse_preserves_payload_and_async_lifetimes(tmp_path):
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if not zig:
        pytest.skip("Set DEEPEP_TEST_ZIG for the native sender cache test")
    exe = tmp_path / ("sender-input-reuse.exe" if os.name == "nt" else "sender-input-reuse")
    subprocess.run([zig, "c++", "-std=c++17", "-O2", "-UNDEBUG",
                    "-I" + str(ROOT / "csrc"),
                    str(Path(__file__).with_name("sender_input_reuse_test.cpp")),
                    "-o", str(exe)], check=True, timeout=180)
    result = subprocess.run([str(exe)], check=True, capture_output=True, text=True, timeout=30)
    assert "PASS sender input reuse" in result.stdout
    print(result.stdout)
