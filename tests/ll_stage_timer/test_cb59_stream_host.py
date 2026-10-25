"""Actual pure-Host stream resource admission and fallback checks; no NPU."""
from pathlib import Path
import os
import shlex
import shutil
import subprocess

import pytest

ROOT=Path(__file__).resolve().parents[2]


def test_stream_host_admission_preserves_fallback_and_rejects_selected_overflow(tmp_path):
    cc=shlex.split(os.environ.get('CXX',''))
    if not cc:
        found=shutil.which('g++') or shutil.which('clang++')
        if not found:pytest.skip('Host C++ compiler required for stream admission')
        cc=[found]
    executable=tmp_path/('stream_host.exe' if os.name=='nt' else 'stream_host')
    result=subprocess.run([*cc,'-std=c++17','-O2','-UNDEBUG','-Werror=unused-parameter',
        '-DDEEPEP_LL_DISPATCH_CB59=2','-DDEEPEP_LL_SEND_CORES=28','-DDEEPEP_LL_RECV_CORES=28',
        '-iquote',str(ROOT),str(Path(__file__).with_name('fixtures')/'cb59_stream_host.cpp'),
        '-o',str(executable)],capture_output=True,text=True,errors='replace',timeout=180)
    assert result.returncode==0,result.stderr[-6000:]
    subprocess.run([str(executable)],check=True,timeout=30)
