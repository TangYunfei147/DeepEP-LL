from pathlib import Path
import os
import subprocess
import pytest
ROOT = Path(__file__).resolve().parents[2]

def test_small_batch_policy(tmp_path):
    zig = os.environ.get('DEEPEP_TEST_ZIG')
    if not zig: pytest.skip('Set DEEPEP_TEST_ZIG for CPU C++ check')
    source = tmp_path/'policy.cpp'
    source.write_text('''
#include <cassert>
#include "csrc/kernels/dispatch/low_latency/common/ep_memory_server_dedup_mode.h"
#include "csrc/kernels/dispatch/low_latency/common/ll_group_policy.h"
int main() {
 for(unsigned world: {16U,32U,64U,128U})
  for(unsigned b: {1U,8U,16U,31U,32U,33U})
   for(unsigned m: {32U,64U,256U}) {
    if(b>m) continue;
    assert(DispatchDedup::EpServerDedupEnabled(m,b,world)==(m>32 || world>16));
    if(world>16) assert(DispatchDedup::LlGroupForWorld(world)==16);
   }
}
'''.replace('#include <cassert>', '#include <cassert>\n#include <initializer_list>'))
    exe=tmp_path/'policy.exe'
    subprocess.run([zig,'c++','-std=c++17','-UNDEBUG','-Wall','-Wextra','-Werror',
                    '-iquote',str(ROOT),str(source),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)
