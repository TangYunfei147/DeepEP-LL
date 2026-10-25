"""Native CPU checks against the archived cb59 resource/admission formulas."""
from pathlib import Path
import os
import shlex
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).with_name("fixtures")


def compiler():
    configured = shlex.split(os.environ.get("CXX", ""))
    if configured:
        return configured
    found = shutil.which("g++") or shutil.which("clang++")
    if not found:
        pytest.skip("native C++ compiler required for replay layout checks")
    return [found]


def test_replay_shape_matrix_and_combine_contract(tmp_path):
    results = []
    for replay in (0, 1):
        executable = tmp_path / (f"layout_{replay}.exe" if os.name == "nt" else f"layout_{replay}")
        built = subprocess.run([
            *compiler(), "-std=c++17", "-O2", "-UNDEBUG", "-Werror=unused-parameter",
            "-iquote", str(ROOT), "-DDEEPEP_LL_SEND_CORES=28", "-DDEEPEP_LL_RECV_CORES=28",
            f"-DDEEPEP_LL_DISPATCH_CB59={replay}", "-DDEEPEP_LL_RANKS_PER_GROUP=16",
            str(FIXTURES / "cb59_layout.cpp"), "-o", str(executable),
        ], capture_output=True, text=True, errors="replace", timeout=180)
        (tmp_path / f"compile_{replay}.log").write_text(built.stdout + built.stderr, encoding="utf-8")
        assert built.returncode == 0, built.stderr[-6000:]
        results.append(subprocess.check_output([str(executable)], text=True, timeout=30))
    # Dispatch's private prefix layout cannot alter combine's admitted shapes or payload bytes.
    assert results[0] == results[1]


def test_replay_rejects_changed_historical_group(tmp_path):
    source = tmp_path / "bad_group.cpp"
    source.write_text('#include "csrc/kernels/dispatch/low_latency/cb59/host_layout.h"\nint main() {}\n')
    result = subprocess.run([
        *compiler(), "-std=c++17", "-iquote", str(ROOT), "-DDEEPEP_LL_DISPATCH_CB59=1",
        "-DDEEPEP_LL_RANKS_PER_GROUP=4", str(source), "-c", "-o", str(tmp_path / "bad_group.o"),
    ], capture_output=True, text=True, errors="replace", timeout=180)
    assert result.returncode != 0 and "cb59 replay preserves its default G16" in result.stderr


def test_reference_fixture_has_frozen_provenance():
    source = (FIXTURES / "cb59_reference_layout.h").read_text(encoding="utf-8")
    assert "cb59bff99ecd39d27ef57788c5e6fd0009e4e636" in source
    assert "coreNum / 2U" in source
    assert "recvCores * layout.prefixStride" in source
