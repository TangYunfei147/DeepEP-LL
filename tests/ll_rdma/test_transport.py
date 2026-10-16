"""Execute the production transport helper against asynchronous CPU doubles."""
from pathlib import Path
import shutil
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('timer_enabled', [False, True])
def test_transport(tmp_path, timer_enabled):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ required for the protocol model")
    binary = tmp_path / "ll-transport"
    subprocess.run([compiler, "-std=c++17", "-O2", "-Wall", "-Wextra",
                    *(['-DDEEPEP_LL_STAGE_TIMER=1'] if timer_enabled else []),
                    "-I" + str(ROOT / "tests/ll_rdma/stubs"), "-I" + str(ROOT / "csrc"),
                    str(ROOT / "tests/ll_rdma/transport_test.cpp"), "-o", str(binary)], check=True)
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True, timeout=60)
    assert "PASS" in result.stdout
    print(result.stdout)
