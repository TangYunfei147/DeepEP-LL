"""Default logical grouping; physical topology and transport remain independent."""
from pathlib import Path
import re
import shutil
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[2]
HEADER='kernels/dispatch/low_latency/common/ep_memory_server_dedup_protocol.h'

def test_default_is_g4_with_explicit_override_retained():
    source=(ROOT/'csrc'/HEADER).read_text()
    assert re.search(r'#ifndef DEEPEP_LL_RANKS_PER_GROUP\s+#if DEEPEP_LL_DISPATCH_CB59\s+'
                     r'#define DEEPEP_LL_RANKS_PER_GROUP 16\s+#else\s+'
                     r'#define DEEPEP_LL_RANKS_PER_GROUP 4\s+#endif\s+#endif',source)
    assert 'kLlRanksPerGroup = DEEPEP_LL_RANKS_PER_GROUP' in source

@pytest.mark.parametrize('override',[None,16])
def test_compiled_group_policy(tmp_path,override):
    compiler=shutil.which('g++') or shutil.which('clang++')
    if not compiler: pytest.skip('C++ compiler unavailable')
    source=tmp_path/'group.cpp'
    source.write_text(f'#include "{HEADER}"\nstatic_assert(DispatchDedup::kLlRanksPerGroup == {override or 4});\nint main() {{}}\n')
    binary=tmp_path/'group.exe'
    subprocess.run([compiler,'-std=c++17','-I'+str(ROOT/'csrc'),str(source),'-o',str(binary),
                    *([f'-DDEEPEP_LL_RANKS_PER_GROUP={override}'] if override else [])],check=True)
    subprocess.run([str(binary)],check=True)
