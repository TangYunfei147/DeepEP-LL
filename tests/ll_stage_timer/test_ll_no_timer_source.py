"""CPU guards for physical LL diagnostic removal from a frozen isolated clone."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('ll_no_timer_source', ROOT / 'scripts/prepare_ll_no_timer.py')
projection = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(projection)


def historical(relative):
    return projection._git(ROOT, 'show', f'{projection.BASE}:{relative}').decode('utf-8')


@pytest.fixture(scope='module')
def expected():
    return {mode: projection.expected_files(ROOT, mode, ROOT)[0] for mode in ('normal-off', 'clean-source')}


@pytest.mark.parametrize('relative', projection.ALGORITHMS)
def test_all_algorithm_business_and_ordered_dependencies(relative, expected):
    before = historical(relative)
    after = expected['clean-source'][relative].decode()
    projection.assert_business_preserved(relative, before, after)
    assert projection.synchronization(before) == projection.synchronization(after)
    projection.reject_timer_code(relative, after)


@pytest.mark.parametrize('mutation', ('remove_wait', 'change_event', 'change_scalar', 'change_branch'))
def test_business_proof_rejects_dependency_and_scalar_mutations(mutation, expected):
    relative = projection.D + '/device/dispatch_parallel_count_prefix.h'
    before = historical(relative)
    after = expected['clean-source'][relative].decode()
    if mutation == 'remove_wait':
        needle = 'SyncFunc<HardEvent::MTE2_S>();'
        assert needle in after
        after = after.replace(needle, '', 1)
    elif mutation == 'change_event':
        after = after.replace('SyncFunc<HardEvent::V_S>();', 'SyncFunc<HardEvent::S_V>();', 1)
    elif mutation == 'change_scalar':
        after = after.replace('complete.GetValue(0U)!=0U', 'complete.GetValue(0U)==0U', 1)
    else:
        after = after.replace('if(!ready)return false;', 'if(ready)return false;', 1)
    with pytest.raises(ValueError, match='business tokens'):
        projection.assert_business_preserved(relative, before, after)


def test_mixed_line_scopes_and_business_waits_survive(expected):
    receiver = expected['clean-source'][projection.D + '/device/dispatch_server_dedup_recv.h'].decode()
    assert '{  WaitFlag<HardEvent::MTE2_S>(lane); }' in receiver
    assert 'ServerDedupReceiveSync<HardEvent::S_MTE3>();' in receiver
    assert 'SetFlag<HardEvent::MTE3_MTE2>(lane);' in receiver
    kernel = expected['clean-source'][projection.D + '/dispatch_server_dedup_kernel.h'].decode()
    assert 'counts_.Finish();' in kernel and 'sender_.Retire();' in kernel
    assert 'if (!overlap_.Async()) {  PipeBarrier<PIPE_ALL>(); }' in kernel
    assert 'tasks,cooperate' in kernel
    assert 'tasks,timer_' not in kernel


def _arguments(text, anchor):
    start = text.index(anchor) + len(anchor) - 1
    end = projection._close(text, start)
    ts = projection.tokens(text[start + 1:end - 1])
    depth, result = 0, [[]]
    for token in ts:
        if token in ('(', '[', '{'):
            depth += 1
        elif token in (')', ']', '}'):
            depth -= 1
        if token == ',' and depth == 0:
            result.append([])
        else:
            result[-1].append(token)
    return result


def test_five_selectors_share_the_clean_init_signature(expected):
    files = (projection.D + '/dispatch_server_dedup_kernel.h', projection.D + '/cb59/dispatch_server_dedup_kernel.h',
             projection.D + '/cb59_small/dispatch_server_dedup_kernel.h',
             projection.D + '/cb59_stream/kernel.h', projection.D + '/cb59_pair/kernel.h')
    entry = expected['clean-source'][projection.ENTRY].decode()
    actual = _arguments(entry, 'kernel.Init(')
    for relative in files:
        formal = _arguments(expected['clean-source'][relative].decode(), 'inline void Init(')
        assert len(formal) == len(actual), relative
    combine = expected['clean-source'][projection.COMBINE_ENTRY].decode()
    formals = _arguments(expected['clean-source'][projection.C + '/combine_server_dedup_kernel.h'].decode(), 'inline void Init(')
    assert len(_arguments(combine, 'kernel.Init(')) == len(formals)


def test_cpp_timer_headers_and_python_recorders_are_physically_absent(expected):
    clean = expected['clean-source']
    assert len([p for p, data in clean.items() if data is None]) == 10
    for relative in projection.DELETE_PATHS:
        assert clean[relative] is None
    for relative in projection.CPP_PATHS:
        projection.reject_timer_code(relative, clean[relative].decode())
    host = clean[projection.HOST].decode()
    assert 'low_latency_stage_timer_layout' not in host
    assert 'low_latency_expert_timer_layout' not in host
    assert 'pin_session_cores(session)' in host
    assert 'session.invalidate_dispatch_controls();' in host


def test_seven_tus_and_o2_other_options_remain(expected):
    original = historical('cmake/device/CMakeLists.txt')
    clean = expected['clean-source']['cmake/device/CMakeLists.txt'].decode()
    files = re.compile(r'"\$\{DEEPEP_SOURCE_DIR\}/[^"]+\.cpp"')
    assert files.findall(clean) == files.findall(original)
    assert len(re.findall(r'low_latency/variants/[^"/]+\.cpp', clean)) == 7
    for block in ('target_compile_options(deepep_kernels', 'target_link_options(deepep_kernels', 'target_include_directories(deepep_kernels'):
        old_start, new_start = original.index(block), clean.index(block)
        old_end = projection._close(original, original.index('(', old_start))
        new_end = projection._close(clean, clean.index('(', new_start))
        assert original[old_start:old_end] == clean[new_start:new_end]
    assert '--npu-arch=dav-3510' in clean and ':-O2>' in clean
    assert 'CATLASS_ARCH=3510 USE_SIMT L2_CACHE_HINT' in clean
    assert 'UNITY_BUILD OFF' in clean
    normal = expected['normal-off']['cmake/device/CMakeLists.txt'].decode()
    assert normal == original


def test_build_support_keeps_source_and_sdk_build_policies(expected):
    before = ast.parse(historical('build_support.py'))
    after = ast.parse(expected['clean-source']['build_support.py'].decode())
    old = {n.name: ast.dump(n) for n in before.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    new = {n.name: ast.dump(n) for n in after.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    for name in old.keys() - {'ll_timer_profile', 'build_configuration', 'build_cache_key', 'VariantBuild', 'CMakeBuild'}:
        assert new[name] == old[name], name
    assert 'll_timer_profile' not in new
    clean = expected['clean-source']['build_support.py'].decode()
    assert 'DEEPEP_LL_STAGE_TIMER' not in clean and 'DEEPEP_LL_TIMER_PROFILE' not in clean
    for mode in ('normal-off', 'clean-source'):
        text = expected[mode]['build_support.py'].decode()
        assert f'result["ll_instrumentation_source"] = "{mode}"' in text


@pytest.fixture(scope='module')
def clones(tmp_path_factory):
    base = tmp_path_factory.mktemp('no-timer-clones')
    result = {}
    for mode in ('normal-off', 'clean-source'):
        source = base / mode
        subprocess.run(['git', '-c', 'core.autocrlf=false', 'clone', '--quiet', '--local', '--shared', '--no-checkout', ROOT.as_posix(), source.as_posix()], check=True)
        subprocess.run(['git', '-C', str(source), '-c', 'core.autocrlf=false', 'checkout', '--quiet', '--detach', projection.BASE], check=True)
        proof = projection.prepare(source, mode, ROOT)
        result[mode] = (source, proof)
    return result


def test_prepare_verify_are_reproducible_and_sdk_helper_unchanged(clones):
    for mode, (source, proof) in clones.items():
        assert proof == projection.verify(source, mode, ROOT)
        assert proof['source_sha256'] == projection.source_fingerprint(source)
        assert proof['sdk_modified_by_projection'] is False
        namespace = {'ROOT': source, 'Path': Path, 'hashlib': hashlib}
        function = next(node for node in ast.parse((source / 'build_support.py').read_text()).body
                        if isinstance(node, ast.FunctionDef) and node.name == 'source_fingerprint')
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<frozen-source-fingerprint>', 'exec'), namespace)
        assert namespace['source_fingerprint']() == proof['source_sha256']
        assert (source / 'scripts/build_shmem.py').read_bytes() == historical('scripts/build_shmem.py').encode()
        assert (source / 'scripts/clean_build_artifacts.py').read_bytes() == historical('scripts/clean_build_artifacts.py').encode()
        with pytest.raises(ValueError, match='already exists'):
            projection.prepare(source, mode, ROOT)


def test_verify_rejects_a_changed_business_file(clones):
    source, _ = clones['clean-source']
    path = source / projection.D / 'device/dispatch_parallel_count_prefix.h'
    data = path.read_bytes()
    try:
        path.write_bytes(data.replace(b'SyncFunc<HardEvent::MTE2_S>();', b'', 1))
        with pytest.raises(ValueError, match='Projected source changed'):
            projection.verify(source, 'clean-source', ROOT)
    finally:
        path.write_bytes(data)


def test_verify_rejects_unlisted_source_files(clones):
    source, _ = clones['clean-source']
    path = source / 'csrc/accidental_extra.cpp'
    try:
        path.write_text('int accidental_extra;\n')
        with pytest.raises(ValueError, match='Untracked files'):
            projection.verify(source, 'clean-source', ROOT)
    finally:
        path.unlink()


def test_projection_refuses_controller():
    with pytest.raises(ValueError, match='independent clone'):
        projection.prepare(ROOT, 'clean-source', ROOT)


def compiler():
    configured = shlex.split(os.environ.get('CXX', ''))
    if configured:
        return configured
    found = shutil.which('g++') or shutil.which('clang++')
    if found:
        return [found]
    pytest.skip('CPU C++ compiler unavailable; set CXX to the existing local Zig c++ command')


def compile_env(directory):
    env = dict(os.environ)
    for name, relative in (('ZIG_GLOBAL_CACHE_DIR', 'cache/global'), ('ZIG_LOCAL_CACHE_DIR', 'cache/local'),
                           ('TMPDIR', 'tmp'), ('TMP', 'tmp'), ('TEMP', 'tmp')):
        path = directory / relative
        path.mkdir(parents=True, exist_ok=True)
        env[name] = str(path)
    return env


@pytest.mark.parametrize('mode,dispatch,combine,offset_d,offset_c', (
    ('normal-off', 152, 96, 140, 92), ('clean-source', 136, 88, 128, 80)))
@pytest.mark.parametrize('selector', [0, 1, 2, 3, 4])
def test_native_launch_abi_and_seven_selector_modes(mode, dispatch, combine, offset_d, offset_c, selector, clones, tmp_path):
    source, _ = clones[mode]
    clean = mode == 'clean-source'
    launcher = 'dispatch_server_dedup_kernel_do_clean136_v1' if clean else 'dispatch_server_dedup_kernel_do_sparse16'
    prefix = 'dispatch_server_dedup_launch_clean136_v1_' if clean else 'dispatch_server_dedup_launch_sparse16_'
    modes = ('prequant_e4m3', 'fp16', 'fp16_e4m3', 'fp16_e5m2', 'bf16', 'bf16_e4m3', 'bf16_e5m2')
    unit = tmp_path / 'abi_selector.cpp'
    unit.write_text(
        '#include <cstddef>\n#include <stdexcept>\n'
        f'#include "{projection.DISPATCH_ABI}"\n#include "{projection.COMBINE_ABI}"\n'
        f'static_assert(sizeof(deepep::EpServerDedupKernelArgs)=={dispatch});\n'
        f'static_assert(sizeof(deepep::EpServerDedupCombineKernelArgs)=={combine});\n'
        f'static_assert(offsetof(deepep::EpServerDedupKernelArgs,coreNum)=={offset_d});\n'
        f'static_assert(offsetof(deepep::EpServerDedupCombineKernelArgs,coreNum)=={offset_c});\n'
        'int observed=-1;\n' + ''.join(
            f'extern "C" void {prefix}{name}(void*,const deepep::EpServerDedupKernelArgs*){{observed={i};}}\n'
            for i, name in enumerate(modes)) +
        f'int main(){{ deepep::EpServerDedupKernelArgs a{{}}; for(unsigned i=0;i<7;++i){{{launcher}(nullptr,&a,i); if(observed!=int(i))return 1;}}'
        f'if(deepep_ll_dispatch_launch_bytes()!={dispatch} || deepep_ll_combine_launch_bytes()!={combine})return 2;'
        f'if(deepep_ll_instrumentation_source_id()!={int(clean)})return 3;'
        f'if(deepep_ll_dispatch_implementation()!={selector})return 5;'
        f'if(deepep_ll_core_protocol_version()!={[19028,21028,22028,23028,24028][selector]})return 6;'
        f'try{{{launcher}(nullptr,&a,7);}}catch(const std::invalid_argument&){{return 0;}}return 4;}}\n', encoding='utf-8')
    output = tmp_path / ('abi.exe' if os.name == 'nt' else 'abi')
    command = [*compiler(), '-std=c++17', '-O2', '-DCATLASS_ARCH=3510',
               f'-DDEEPEP_LL_DISPATCH_CB59={selector}', '-iquote', str(source),
               str(source / projection.SELECTOR), str(unit), '-o', str(output)]
    result = subprocess.run(command, env=compile_env(tmp_path), capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr[-6000:]
    subprocess.run([str(output)], check=True, timeout=30)


@pytest.mark.parametrize('host_mode,device_mode', (('normal-off', 'clean-source'), ('clean-source', 'normal-off')))
def test_mixed_host_device_link_symbols_are_rejected(host_mode, device_mode, clones, tmp_path):
    host, _ = clones[host_mode]
    device, _ = clones[device_mode]
    clean_device = device_mode == 'clean-source'
    launcher = 'dispatch_server_dedup_kernel_do_clean136_v1' if host_mode == 'clean-source' else 'dispatch_server_dedup_kernel_do_sparse16'
    prefix = 'dispatch_server_dedup_launch_clean136_v1_' if clean_device else 'dispatch_server_dedup_launch_sparse16_'
    unit = tmp_path / 'mixed.cpp'
    unit.write_text(f'#include "{(host / projection.DISPATCH_ABI).as_posix()}"\n' + ''.join(
        f'extern "C" void {prefix}{name}(void*,const deepep::EpServerDedupKernelArgs*){{}}\n'
        for name in ('prequant_e4m3', 'fp16', 'fp16_e4m3', 'fp16_e5m2', 'bf16', 'bf16_e4m3', 'bf16_e5m2')) +
        f'int main(){{deepep::EpServerDedupKernelArgs a{{}};{launcher}(nullptr,&a,0);}}\n', encoding='utf-8')
    result = subprocess.run([*compiler(), '-std=c++17', '-DCATLASS_ARCH=3510', '-iquote', str(device),
                             str(device / projection.SELECTOR), str(unit), '-o', str(tmp_path / 'mixed.exe')],
                            env=compile_env(tmp_path), capture_output=True, text=True, timeout=180)
    assert result.returncode != 0
    assert launcher in result.stderr
