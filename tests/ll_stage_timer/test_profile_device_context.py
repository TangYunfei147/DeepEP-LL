"""Profile helpers must match device callers in every mixed compiler pass.

Preprocessing is a regression guard, not a replacement for real Bisheng checks.
The optional CANN tests compile only; they never launch an NPU kernel.
"""
import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
COMMON = 'csrc/kernels/dispatch/low_latency/common/'


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('compiler_macros', [[], ['-D__CCE__=1'],
                                           ['-D__CCE__=1', '-D__CCE_IS_AICORE__=1']])
def test_device_include_qualifies_helpers_even_in_host_pass(tmp_path, enabled, compiler_macros):
    compiler = shutil.which('g++') or shutil.which('clang++')
    if not compiler:
        pytest.skip('C++ preprocessor unavailable')
    (tmp_path/'kernel_operator.h').write_text('#pragma once\n#define __aicore__ LL_AICORE_MARKER\n')
    source = tmp_path/'device.cpp'
    source.write_text('#include "' + COMMON + 'll_stage_timer_device.h"\n'
                      '#ifdef DEEPEP_LL_TIMER_DEVICE_CONTEXT\n#error device include context leaked\n#endif\n')
    output = subprocess.check_output([compiler, '-E', '-P', '-std=c++17', *compiler_macros,
        *(['-DDEEPEP_LL_STAGE_TIMER=1'] if enabled else []), '-I', str(tmp_path), '-I', str(ROOT), str(source)], text=True)
    for result, name in [('bool', 'ProfileEnabled'), ('uint32_t', 'ProfileIndex')]:
        assert re.search(r'LL_AICORE_MARKER\s+inline\s+constexpr\s+' + result + r'\s+' + name + r'\(', output)


@pytest.mark.parametrize('profile', range(6))
def test_plain_host_can_use_profile_metadata_and_constexpr_helpers(tmp_path, profile):
    compiler = shutil.which('g++') or shutil.which('clang++')
    if not compiler:
        pytest.skip('C++ compiler unavailable')
    source = tmp_path/'host.cpp'
    source.write_text('#include "' + COMMON + 'll_stage_timer_profile.h"\n'
        '#include <cassert>\nint main() { using namespace LlStageTimer;\n'
        'static_assert(kProfileMetrics > 0 && kProfileMetrics <= kMetrics);\n'
        'unsigned expected=0; for(unsigned i=0;i<kMetrics;++i) {\n'
        'assert(ProfileIndex(Stage(i))==expected); if(ProfileEnabled(Stage(i))) ++expected; }\n'
        'assert(expected==kProfileMetrics); }\n')
    exe = tmp_path/'host'
    subprocess.run([compiler, '-std=c++17', '-Wall', '-Wextra', '-Werror',
        f'-DDEEPEP_LL_TIMER_PROFILE_ID={profile}', '-I', str(ROOT), str(source), '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True, timeout=30)


@pytest.mark.parametrize('profile', [None, *range(6)])
def test_bisheng_mixed_pass_recorder_and_runtime_profile_calls(tmp_path, profile):
    compiler = shutil.which('bisheng')
    cann = os.environ.get('ASCEND_HOME_PATH')
    if not compiler or not cann:
        pytest.skip('CANN compiler/environment unavailable; compile-only test')
    source = tmp_path/'profile.cpp'
    source.write_text('#include "' + COMMON + 'll_stage_timer_device.h"\n'
        'extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__ void profile_compile_probe(\n'
        'GM_ADDR records, uint32_t runtimeStage, uint32_t mode) {\n'
        'using namespace LlStageTimer; DeviceTimer timer; timer.Init(records,1U,mode);\n'
        'AscendC::TPipe pipe; RelayCompletionScope relay(&timer,&pipe);\n'
        'DeviceTimer* timer_ = timer.Enabled() ? &timer : nullptr;\n'
        'LL_SEG_BEGIN(begin,TOKEN_INPUT); LL_SEG_END(begin,TOKEN_INPUT);\n'
        '{ LL_SEG_SCOPE(INIT); } LL_SEG_FIRST(BEFORE_FIRST_SEND); LL_SEG_COUNT(TX_PACKETS,1);\n'
        'if(runtimeStage<kMetrics) { auto stage=Stage(runtimeStage);\n'
        'timer.End(stage,timer.Begin(stage)); } timer.Flush(); }\n')
    command = [compiler, '-O2', '-std=c++17', '-fPIC', '-xasc', '--npu-arch=dav-3510',
               '--cce-auto-infer-kernel-type=false', '-I', str(ROOT), '-I', str(Path(cann)/'include'),
               '-I', str(Path(cann)/'asc/include')]
    if profile is not None:
        command += ['-DDEEPEP_LL_STAGE_TIMER=1', f'-DDEEPEP_LL_TIMER_PROFILE_ID={profile}']
    result = subprocess.run([*command, '-c', str(source), '-o', str(tmp_path/'profile.o')],
                            capture_output=True, text=True, timeout=180)
    (tmp_path/'compile.log').write_text(result.stdout+result.stderr, encoding='utf-8')
    assert result.returncode == 0, result.stderr[-8000:]
