"""Sparse recorder compiler-pass and metadata guards; CANN checks compile only."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import pytest
ROOT=Path(__file__).resolve().parents[2]
COMMON='csrc/kernels/dispatch/low_latency/common/'

@pytest.mark.parametrize('enabled',[False,True])
@pytest.mark.parametrize('compiler_macros',[[],['-D__CCE__=1'],['-D__CCE__=1','-D__CCE_IS_AICORE__=1']])
def test_device_template_is_qualified_in_every_compiler_pass(tmp_path,enabled,compiler_macros):
    compiler=shutil.which('g++') or shutil.which('clang++')
    if not compiler: pytest.skip('C++ preprocessor unavailable')
    (tmp_path/'kernel_operator.h').write_text('#pragma once\n#define __aicore__ LL_AICORE_MARKER\n')
    source=tmp_path/'device.cpp'
    source.write_text('#include "'+COMMON+'ll_stage_timer_device.h"\n')
    output=subprocess.check_output([compiler,'-E','-P','-std=c++17',*compiler_macros,
        *(['-DDEEPEP_LL_STAGE_TIMER=1'] if enabled else []),'-I',str(tmp_path),'-I',str(ROOT),str(source)],text=True)
    assert re.search(r'template<Point point>\s+LL_AICORE_MARKER\s+inline\s+void\s+Mark\(',output)
    assert 'ProfileIndex' not in output

def test_plain_host_can_use_sparse_metadata(tmp_path):
    compiler=shutil.which('g++') or shutil.which('clang++')
    if not compiler: pytest.skip('C++ compiler unavailable')
    source=tmp_path/'host.cpp'
    source.write_text('#include "'+COMMON+'ll_stage_timer_profile.h"\n'
        'int main(){using namespace LlStageTimer;\n'
        'static_assert(kProfileId==6 && kPointCount==14 && kSchema==15);\n'
        'static_assert(kSlots==8 && kFields==8 && kExportWords==24); }\n')
    exe=tmp_path/'host'
    subprocess.run([compiler,'-std=c++17','-Wall','-Wextra','-Werror','-I',str(ROOT),str(source),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True,timeout=30)

@pytest.mark.parametrize('profile',[None,6])
def test_bisheng_mixed_pass_sparse_recorder(tmp_path,profile):
    compiler=shutil.which('bisheng');cann=os.environ.get('ASCEND_HOME_PATH')
    if not compiler or not cann: pytest.skip('CANN compiler/environment unavailable; compile-only test')
    source=tmp_path/'profile.cpp'
    source.write_text('#include "'+COMMON+'ll_stage_timer_device.h"\n'
        'extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__ void profile_compile_probe(\n'
        'GM_ADDR records,uint32_t mode){\n'
        'using namespace LlStageTimer;DeviceTimer timer;timer.Init(records,1U,mode);\n'
        'DeviceTimer* timer_=timer.Enabled()?&timer:nullptr;\n'
        'LL_SEG_BEGIN(begin,TOKEN_INPUT);LL_SEG_END(begin,TOKEN_INPUT);\n'
        '{LL_SEG_SCOPE(INIT);}LL_SEG_FIRST(BEFORE_FIRST_SEND);LL_SEG_COUNT(TX_PACKETS,1);\n'
        'timer.Mark<ENTRY>();LL_SPARSE_MARK(INIT_DONE);LL_SPARSE_MARK(FIRST_SEND);\n'
        'timer.Mark<WORK_DONE>();timer.Flush();}\n')
    command=[compiler,'-O2','-std=c++17','-fPIC','-xasc','--npu-arch=dav-3510',
        '--cce-auto-infer-kernel-type=false','-I',str(ROOT),'-I',str(Path(cann)/'include'),'-I',str(Path(cann)/'asc/include')]
    if profile is not None:command+=['-DDEEPEP_LL_STAGE_TIMER=1',f'-DDEEPEP_LL_TIMER_PROFILE_ID={profile}']
    result=subprocess.run([*command,'-c',str(source),'-o',str(tmp_path/'profile.o')],capture_output=True,text=True,timeout=180)
    (tmp_path/'compile.log').write_text(result.stdout+result.stderr,encoding='utf-8')
    assert result.returncode==0,result.stderr[-8000:]
