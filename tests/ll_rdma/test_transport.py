"""Execute the production transport helper against asynchronous CPU doubles."""
from pathlib import Path
import os
import shutil
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('timer_enabled', [False, True])
def test_transport(tmp_path, timer_enabled):
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    compiler = [zig, "c++"] if zig else ([shutil.which("g++")] if shutil.which("g++") else None)
    if compiler is None:
        pytest.skip("C++ compiler required for the protocol model")
    binary = tmp_path / "ll-transport.exe"
    subprocess.run([*compiler, "-std=c++17", "-O2", "-UNDEBUG", "-Wall", "-Wextra",
                    *(['-DDEEPEP_LL_STAGE_TIMER=1'] if timer_enabled else []),
                    "-I" + str(ROOT / "tests/ll_rdma/stubs"), "-I" + str(ROOT / "csrc"),
                    str(ROOT / "tests/ll_rdma/transport_test.cpp"), "-o", str(binary)], check=True)
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True, timeout=60)
    assert "PASS" in result.stdout
    print(result.stdout)
