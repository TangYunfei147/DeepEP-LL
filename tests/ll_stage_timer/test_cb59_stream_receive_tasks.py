"""Execute production record-owned task construction under checked CPU storage."""
from pathlib import Path
import os
import shlex
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).with_name("fixtures")


def test_actual_stream_tasks_mapping_and_cleanup_ownership(tmp_path):
    compiler = shlex.split(os.environ.get("CXX", ""))
    if not compiler:
        found = shutil.which("g++") or shutil.which("clang++")
        if not found:
            pytest.skip("native C++ compiler required for actual task construction")
        compiler = [found]
    executable = tmp_path / ("tasks.exe" if os.name == "nt" else "tasks")
    result = subprocess.run([
        *compiler, "-std=c++17", "-O2", "-UNDEBUG", "-DDEEPEP_LL_DISPATCH_CB59=2",
        "-I", str(FIXTURES / "receive_stub"), "-iquote", str(ROOT),
        str(FIXTURES / "stream_receive_tasks.cpp"), "-o", str(executable),
    ], capture_output=True, text=True, errors="replace", timeout=180)
    (tmp_path / "compile.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    assert result.returncode == 0, result.stderr[-6000:]
    subprocess.run([str(executable)], check=True, timeout=30)
