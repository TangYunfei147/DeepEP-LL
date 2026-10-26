#!/usr/bin/env python3
"""Project a fixed LL checkout into normal OFF or physically timer-free source.

Only a separate pristine clone is accepted. No SDK build, framework import,
compiler invocation or device operation occurs here. The controller checkout is
read-only; its Python projection helper is applied to the same immutable base.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import types

ROOT = Path(__file__).resolve().parents[1]
BASE = "439d4597a52b657858217bac7fc1e965f22507f7"
FORMAT = "deepep-ll-no-timer-source-v1"
MANIFEST = "ll-no-timer-source.json"
D = "csrc/kernels/dispatch/low_latency"
C = "csrc/kernels/combine/low_latency"
HOST = "csrc/ops/low_latency.cpp"
ENTRY = D + "/dispatch_variant_kernel.h"
SELECTOR = "csrc/kernels/dispatch/low_latency.cpp"
COMBINE_ENTRY = "csrc/kernels/combine/low_latency.cpp"
DISPATCH_ABI = D + "/launch.hpp"
COMBINE_ABI = C + "/launch.hpp"
ALGORITHMS = (
    D + "/dispatch_server_dedup_kernel.h",
    *(D + "/device/" + name for name in (
        "dispatch_server_dedup_counts.h", "dispatch_server_dedup_send.h",
        "dispatch_server_dedup_recv.h", "dispatch_parallel_count_prefix.h")),
    *(D + "/" + folder + "/" + name for folder in ("cb59", "cb59_small") for name in (
        "dispatch_server_dedup_kernel.h", "device/dispatch_server_dedup_counts.h",
        "device/dispatch_server_dedup_send.h", "device/dispatch_server_dedup_recv.h")),
    *(D + "/cb59_stream/" + name for name in ("kernel.h", "stream_counts.h", "receiver.h")),
    *(D + "/cb59_pair/" + name for name in ("kernel.h", "pair_handoff.h", "receiver.h")),
    *(C + "/" + name for name in ("combine_server_dedup_kernel.h",
        "combine_server_dedup_partial.h", "combine_server_dedup_source_aggregate.h")),
)
CPP_PATHS = (*ALGORITHMS, ENTRY, SELECTOR, COMBINE_ENTRY, DISPATCH_ABI, COMBINE_ABI, HOST)
DELETE_PATHS = tuple(D + "/common/" + name for name in (
    "ll_stage_timer_device.h", "ll_stage_timer_layout.h", "ll_stage_timer_profile.h",
    "ll_expert_timer_device.h", "ll_expert_timer_layout.h")) + ("cmake/LlTimerProfile.cmake",)
BUILD_PATHS = ("build_support.py", "CMakeLists.txt", "cmake/Device.cmake", "cmake/device/CMakeLists.txt")
_TOKEN = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[A-Za-z_]\w*|0[xX][\da-fA-F]+|\d+(?:\.\d+)?[uUlLfF]*|[^\s]', re.S)


def tokens(text):
    """Keep braces, operators and literal values; drop comments only."""
    return [match.group() for match in _TOKEN.finditer(text)
            if not match.group().startswith(("//", "/*"))]


def _once(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f"Fixed-base anchor missing or ambiguous: {before[:100]!r}")
    return text.replace(before, after, 1)


def _close(text, start, left="(", right=")"):
    if text[start] != left:
        raise ValueError("Expected opening delimiter")
    depth = 0
    for match in _TOKEN.finditer(text, start):
        token = match.group()
        if token == left:
            depth += 1
        elif token == right:
            depth -= 1
            if depth == 0:
                return match.end()
    raise ValueError("Unbalanced fixed-source delimiter")


def _erase_calls(text, pattern):
    """Erase exactly a call statement, never the surrounding scope or branch."""
    while match := re.search(pattern, text):
        opening = text.find("(", match.start())
        end = _close(text, opening)
        tail = re.match(r"\s*;", text[end:])
        if tail is None:
            raise ValueError("Diagnostic expression was not a standalone call statement")
        text = text[:match.start()] + text[end + tail.end():]
    return text


def _erase_function(text, name):
    pattern = re.compile(r"^[^\n;{}]*\b" + re.escape(name) + r"\([^;{}]*\)\s*\{", re.M)
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise ValueError(f"Expected one fixed function: {name}")
    match = matches[0]
    opening = text.index("{", match.start())
    end = _close(text, opening, "{", "}")
    return text[:match.start()] + text[end:]


def _drop_argument(text, pattern):
    """Drop one formal/actual argument while preserving all other arguments."""
    def code_mask(value):
        return _TOKEN.sub(lambda m: ''.join('\n' if ch == '\n' else ' ' for ch in m.group())
                          if m.group().startswith(('//', '/*', '"', "'")) else m.group(), value)
    while match := re.search(pattern, code_mask(text)):
        start, end = match.span()
        before = start - 1
        while before >= 0 and text[before].isspace():
            before -= 1
        after = end
        while after < len(text) and text[after].isspace():
            after += 1
        if before >= 0 and text[before] == "," and text[after] in ",)":
            start = before
        elif before >= 0 and text[before] == "(" and text[after] == ",":
            end = after + 1
        elif not (before >= 0 and text[before] == "(" and text[after] == ")"):
            raise ValueError(f"Timer token is not an isolated argument: {text[max(0,start-45):end+45]!r}")
        text = text[:start] + text[end:]
    return text


def _strip_cpp_body(text):
    # Stage headers also supplied core constants; retain that ordinary include
    # explicitly instead of keeping a dummy timer header for transitive names.
    text = re.sub(r'(^\s*#include\s+"[^"\n]*)ll_stage_timer_device\.h"', r'\1ll_core_config.h"', text, flags=re.M)
    text = re.sub(r'^\s*#include\s+"[^"\n]*ll_(?:stage|expert)_timer[^"\n]*"[^\n]*\n', '', text, flags=re.M)
    for match in reversed(list(re.finditer(r'^#ifdef DEEPEP_LL_STAGE_TIMER\n(.*?)^#endif[^\n]*\n', text, re.M | re.S))):
        body = match.group(1)
        if not ("sparseEntry" in body and ("GetSystemCycle" in body or "MarkAt" in body)):
            raise ValueError("Unexpected timer conditional in active LL source")
        text = text[:match.start()] + text[match.end():]
    text = _erase_calls(text, r'\bLL_(?:SEG_[A-Z_]+|SPARSE_(?:MARK|AFTER))\s*\(')
    text = re.sub(r'__aicore__ inline void SetTimer\(LlStageTimer::DeviceTimer\* timer\)\s*\{\s*timer_ = timer;\s*\}', '', text)
    text = re.sub(r'\b\w+\.SetTimer\(timer_\);', '', text)
    text = re.sub(r'\bLl(?:Stage|Expert)Timer::DeviceTimer\* (?:timer_|expertTimer_)\s*=\s*nullptr;', '', text)
    text = re.sub(r'\b(?:timer_|expertTimer_)\s*=\s*(?:timer|expertTimer);', '', text)
    text = re.sub(r'\(void\)(?:expertTimer|timer);', '', text)
    text = re.sub(r'if \(expertTimer_\) expertTimer_->(?:Start|End)\([^;]*\);', '', text)
    if 'if (expertTimer_ && expertTimer_->Enabled())' in text:
        start = text.index('        uint32_t expert = 0;')
        guard = text.index('if (expertTimer_ && expertTimer_->Enabled())', start)
        end = _close(text, text.index('{', guard), '{', '}')
        block = text[start:end]
        if any(name in block for name in ('WaitFlag', 'SetFlag', 'Sync', 'DataCopy', 'PipeBarrier')):
            raise ValueError("Expert-only bookkeeping unexpectedly contains a business dependency")
        text = text[:start] + text[end:]
    text = re.sub(r'LlStageTimer::DeviceTimer timer;', '', text)
    text = _erase_calls(text, r'\btimer\.(?:Init|Mark(?:At)?<[^>]+>|Flush)\s*\(')
    text = _drop_argument(text, r'Ll(?:Stage|Expert)Timer::DeviceTimer\*\s*(?:timer|expertTimer)(?:\s*=\s*nullptr)?')
    if 'timer.Enabled() ? &timer : nullptr,nullptr' in text:
        text = _once(text, ',\n                    timer.Enabled() ? &timer : nullptr,nullptr', '')
    text = _drop_argument(text, r'\b(?:GM_ADDR\s+(?:stageTimer|expertTimer)|uint32_t\s+timerMode)\b')
    text = _drop_argument(text, r'(?<![\w.])(?:&timer|timer_|timer|expertTimer)\b')
    text = _drop_argument(text, r'\ba\.(?:stageTimer|timerMode|expertTimer)\b')
    return text


def _identity_cpp(relative, text, mode):
    clean = mode == "clean-source"
    identity, dispatch_bytes, combine_bytes = (1, 136, 88) if clean else (0, 152, 96)
    if relative == DISPATCH_ABI:
        text += ('\nextern "C" uint32_t deepep_ll_instrumentation_source_id();\n'
                 'extern "C" uint32_t deepep_ll_dispatch_launch_bytes();\n'
                 'extern "C" uint32_t deepep_ll_combine_launch_bytes();\n')
    elif relative == SELECTOR:
        text = _once(text, '#include "low_latency/launch.hpp"',
                     '#include "low_latency/launch.hpp"\n#include "../combine/low_latency/launch.hpp"')
        anchor = '#if CATLASS_ARCH == 3510\n'
        exports = (f'extern "C" DEEPEP_EXPORT uint32_t deepep_ll_instrumentation_source_id() {{ return {identity}U; }}\n'
                   'extern "C" DEEPEP_EXPORT uint32_t deepep_ll_dispatch_launch_bytes() { return sizeof(deepep::EpServerDedupKernelArgs); }\n'
                   'extern "C" DEEPEP_EXPORT uint32_t deepep_ll_combine_launch_bytes() { return sizeof(deepep::EpServerDedupCombineKernelArgs); }\n')
        text = _once(text, anchor, anchor + exports)
    elif relative == HOST:
        anchor = 'void bind_low_latency(py::module_& module)\n{\n'
        checks = (f'    if (deepep_ll_instrumentation_source_id() != {identity}U ||\n'
                  '        deepep_ll_dispatch_launch_bytes() != sizeof(EpServerDedupKernelArgs) ||\n'
                  '        deepep_ll_combine_launch_bytes() != sizeof(EpServerDedupCombineKernelArgs))\n'
                  '        throw std::runtime_error("LL Host/device source artifact or launch ABI mismatch");\n'
                  f'    module.attr("low_latency_instrumentation_source") = "{mode}";\n'
                  f'    module.attr("dispatch_launch_bytes") = {dispatch_bytes};\n'
                  f'    module.attr("combine_launch_bytes") = {combine_bytes};\n')
        text = _once(text, anchor, anchor + checks)
    return text


def transform_cpp(relative, text):
    """Return the clean C++ projection; reject unexpected diagnostic grammar."""
    if relative not in CPP_PATHS:
        raise ValueError(f"Not an allowed C++ projection: {relative}")
    if relative == HOST:
        text = _erase_function(text, "stage_timer")
        text = _erase_function(text, "expert_timer")
        start = text.index('    if (deepep_ll_stage_timer_schema()')
        end = text.index('    if (deepep_ll_core_protocol_version()', start)
        text = text[:start] + text[end:]
        text = re.sub(r'^\s*module\.attr\("low_latency_(?:stage_timer[^"\n]*|expert_timer[^"\n]*)"\)[^\n]*\n', '', text, flags=re.M)
        for name in ("low_latency_expert_timer_layout", "low_latency_stage_timer_layout"):
            text = _erase_calls(text, r'module\.def\("' + name + '"')
        text = re.sub(r'^\s*args\.(?:stageTimer|expertTimer)\s*=[^\n]*\n', '', text, flags=re.M)
    if relative == SELECTOR:
        for name in ("deepep_ll_stage_timer_schema", "deepep_ll_stage_timer_profile_id", "deepep_ll_stage_timer_enabled"):
            text = _erase_function(text, name)
    if relative == DISPATCH_ABI:
        text = re.sub(r'^\s*uint8_t\* (?:stageTimer|expertTimer);[^\n]*\n|^\s*uint32_t timerMode;[^\n]*\n', '', text, flags=re.M)
        text = re.sub(r'^static_assert\(offsetof\(EpServerDedupKernelArgs, (?:stageTimer|timerMode|expertTimer)\)[^\n]*\n', '', text, flags=re.M)
        text = _once(text, 'sizeof(EpServerDedupKernelArgs) == 152', 'sizeof(EpServerDedupKernelArgs) == 136')
        text = _once(text, '"launch ABI includes timer pointer, timer mode and active core count"', '"clean launch ABI retains all business addresses and active core count"')
        text = _once(text, 'coreNum) == 140', 'coreNum) == 128')
        text = re.sub(r'^extern "C" uint32_t deepep_ll_stage_timer_\w+\(\);\n', '', text, flags=re.M)
    if relative == COMBINE_ABI:
        text = re.sub(r'^\s*uint8_t\* stageTimer;\n|^\s*uint32_t timerMode;\n', '', text, flags=re.M)
        text = re.sub(r'^static_assert\(offsetof\(EpServerDedupCombineKernelArgs, (?:stageTimer|timerMode)\)[^\n]*\n', '', text, flags=re.M)
        text = _once(text, 'sizeof(EpServerDedupCombineKernelArgs) == 96', 'sizeof(EpServerDedupCombineKernelArgs) == 88')
        text = _once(text, 'coreNum) == 92', 'coreNum) == 80')
    text = _strip_cpp_body(text)
    text = text.replace('dispatch_server_dedup_kernel_do_sparse16', 'dispatch_server_dedup_kernel_do_clean136_v1')
    text = text.replace('dispatch_server_dedup_launch_sparse16_', 'dispatch_server_dedup_launch_clean136_v1_')
    text = text.replace('combine_server_dedup_kernel_do_sparse16', 'combine_server_dedup_kernel_do_clean88_v1')
    text = _identity_cpp(relative, text, "clean-source")
    reject_timer_code(relative, text)
    return text


def reject_timer_code(relative, text):
    code = ' '.join(tokens(text))
    forbidden = r'\b(?:LlStageTimer|LlExpertTimer|DeviceTimer|GetSystemCycle|LL_SEG_\w+|LL_SPARSE_\w+|DEEPEP_LL_(?:STAGE_TIMER|TIMER_\w+)|stageTimer|timerMode|expertTimer_?|timer_)\b|ll_(?:stage|expert)_timer'
    if match := re.search(forbidden, code):
        raise ValueError(f"Timer code remains in {relative}: {match.group()}")


def transform_build(relative, text, mode):
    clean = mode == "clean-source"
    if relative == "build_support.py":
        if clean:
            tree = ast.parse(text)
            fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "ll_timer_profile")
            lines = text.splitlines(keepends=True)
            text = ''.join(lines[:fn.lineno - 1] + lines[fn.end_lineno:])
            text = _once(text, 'LL_TIMER_PROFILES = ("sparse",)\n', '')
            text = _once(text, '    if "ll_timer_profile" in identity and not identity.get("ll_stage_timer", False):\n        identity["ll_timer_profile"] = "sparse"\n', '')
            text = _once(text, '    result["ll_stage_timer"] = enabled("DEEPEP_LL_STAGE_TIMER", env)\n    result["ll_timer_profile"] = ll_timer_profile(env)\n', '')
            text = _once(text, '        if enabled("DEEPEP_LL_STAGE_TIMER"):\n            variant += "-ll-timer"\n            profile = ll_timer_profile()\n            if profile != "full":\n                variant += "-" + profile\n', '')
            text = re.sub(r'^\s*f"-DDEEPEP_LL_(?:STAGE_TIMER|TIMER_PROFILE)=[^\n]*\n', '', text, flags=re.M)
        else:
            text = _once(text, 'result["ll_stage_timer"] = enabled("DEEPEP_LL_STAGE_TIMER", env)', 'result["ll_stage_timer"] = False  # Fixed normal-OFF comparison arm.')
        anchor = '    result["ll_dispatch_capability"] = dispatch_capability(result["ll_dispatch_impl"])\n'
        metadata = (f'    result["ll_instrumentation_source"] = "{mode}"\n'
                    f'    result["dispatch_launch_bytes"] = {136 if clean else 152}\n'
                    f'    result["combine_launch_bytes"] = {88 if clean else 96}\n')
        text = _once(text, anchor, anchor + metadata)
        ast.parse(text)
    elif clean:
        if relative == "cmake/device/CMakeLists.txt":
            text = _once(text, 'if(DEEPEP_LL_STAGE_TIMER)\n    target_compile_definitions(deepep_kernels PRIVATE DEEPEP_LL_STAGE_TIMER=1)\nendif()\n', '')
        text = re.sub(r'^.*(?:LlTimerProfile\.cmake|DEEPEP_LL_STAGE_TIMER|DEEPEP_LL_TIMER_PROFILE)[^\n]*\n', '', text, flags=re.M)
    return text


def synchronization(text):
    """Ordered exact dependency calls, including event expressions and braces."""
    ts = tokens(text)
    names = {'Sync', 'SyncFunc', 'PipeBarrier', 'SetFlag', 'WaitFlag', 'ServerDedupReceiveSync',
             'Reset', 'SetCtrlSpr', 'DataCopy', 'DataCopyPad', 'DataCopyPadExt', 'asc_vf_call',
             'GetValue', 'SetValue', 'GatherMask', 'ReduceSum', 'Sum', 'Duplicate'}
    calls = []
    for index, token in enumerate(ts):
        if token in names and index + 1 < len(ts) and ts[index + 1] in {'<', '('}:
            opening = index + 1
            while opening < len(ts) and ts[opening] != '(':
                opening += 1
            depth, end = 1, opening + 1
            while end < len(ts) and depth:
                depth += (ts[end] == '(') - (ts[end] == ')')
                if depth:
                    end += 1
            if depth:
                raise ValueError("Unbalanced dependency call")
            calls.append(' '.join(ts[index:end + 1]))
    return calls


def assert_business_preserved(relative, before, after):
    """Check the entire projected token stream, plus independent DMA/sync calls.

    The expected token stream is derived from the immutable input with the
    narrow diagnostic grammar above. No Scalar statement, brace or general
    conditional is normalized away. A second extractor compares every ordered
    synchronization, DMA, vector and Scalar access statement independently.
    """
    expected = transform_cpp(relative, before)
    if tokens(after) != tokens(expected):
        raise ValueError(f"Non-projection business tokens changed: {relative}")
    if relative in ALGORITHMS and synchronization(before) != synchronization(after):
        raise ValueError(f"Business Scalar/DMA/synchronization sequence changed: {relative}")


def _git(source, *args):
    return subprocess.check_output(['git', '-C', str(source), *args],
                                   env=dict(os.environ, GIT_OPTIONAL_LOCKS='0'), timeout=90)


def _helper(controller):
    path = Path(controller) / 'scripts/ll_no_timer_python.py'
    # Execute the controller-owned pure transform without creating __pycache__
    # in the read-only controller checkout.
    module = types.ModuleType('_ll_no_timer_python_projection')
    module.__file__ = str(path)
    exec(compile(path.read_text(encoding='utf-8'), str(path), 'exec'), module.__dict__)
    return module


def expected_files(source, mode, controller=ROOT):
    if mode not in {'normal-off', 'clean-source'}:
        raise ValueError("Unknown source projection")
    helper = _helper(controller)
    cpp = CPP_PATHS if mode == 'clean-source' else (SELECTOR, DISPATCH_ABI, HOST)
    paths = (*cpp, *BUILD_PATHS, *helper.TRANSFORM_PATHS)
    expected, originals = {}, {}
    for relative in dict.fromkeys(paths):
        raw = _git(source, 'show', f'{BASE}:{relative}')
        text = raw.decode('utf-8')
        originals[relative] = raw
        if relative in cpp:
            output = transform_cpp(relative, text) if mode == 'clean-source' else _identity_cpp(relative, text, mode)
            if mode == 'clean-source':
                assert_business_preserved(relative, text, output)
        elif relative in BUILD_PATHS:
            output = transform_build(relative, text, mode)
        else:
            output = helper.transform_python(relative, text, mode)
        expected[relative] = output.encode('utf-8')
    if mode == 'clean-source':
        for relative in (*DELETE_PATHS, *helper.DELETE_PATHS):
            originals[relative] = _git(source, 'show', f'{BASE}:{relative}')
            expected[relative] = None
    return expected, originals


def source_fingerprint(source):
    """The fixed base's exact source digest, without importing build_support."""
    source = Path(source)
    files = [source / name for name in ('VERSION', 'setup.py', 'pyproject.toml', 'build_support.py',
        'CMakeLists.txt', 'dependencies.lock.json', 'MANIFEST.in', 'requirements-dev.txt',
        'requirements-build.txt', '.gitmodules', 'experiments/ep_gmm_fused/tests/requirements.txt',
        'third-party/deepseek-deepep.LICENSE', 'third-party/cann.LICENSE')]
    for folder in ('deep_ep', 'csrc', 'cmake', 'scripts', 'tests', 'benchmarks', 'experiments/megamoe', 'experiments/ep_gmm_fused'):
        files.extend(path for path in (source / folder).rglob('*')
                     if (path.suffix in {'.sh', '.py', '.cpp', '.hpp', '.h', '.cmake', '.json'} or path.name == 'CMakeLists.txt')
                     and path.name != '_build_info.json')
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.relative_to(source).as_posix().encode() + b'\0')
        digest.update(path.read_bytes().replace(b'\r\n', b'\n') + b'\0')
    return digest.hexdigest()


def _proof(source, mode, expected, originals):
    clean = mode == 'clean-source'
    return {'format': FORMAT, 'reference': BASE, 'mode': mode, 'manifest': MANIFEST,
            'source_sha256': source_fingerprint(source), 'dispatch_launch_bytes': 136 if clean else 152,
            'combine_launch_bytes': 88 if clean else 96,
            'changed_paths': sorted(path for path, data in expected.items() if data is not None and data != originals[path]),
            'deleted_paths': sorted(path for path, data in expected.items() if data is None),
            'files': {path: {'before_sha256': hashlib.sha256(originals[path]).hexdigest(),
                             'after_sha256': hashlib.sha256(data).hexdigest() if data is not None else None}
                      for path, data in sorted(expected.items())},
            'business_tokens_verified': clean, 'ordered_dependency_calls_verified': clean,
            'algorithm_proofs': {relative: {
                'projected_business_tokens_sha256': hashlib.sha256(' '.join(tokens(expected[relative].decode())).encode()).hexdigest(),
                'original_dependency_sequence_sha256': hashlib.sha256(json.dumps(synchronization(originals[relative].decode())).encode()).hexdigest(),
                'projected_dependency_sequence_sha256': hashlib.sha256(json.dumps(synchronization(expected[relative].decode())).encode()).hexdigest(),
            } for relative in ALGORITHMS} if clean else {},
            'sdk_modified_by_projection': False, 'device_validation': 'not-performed'}


def _check_root(source, controller):
    source, controller = Path(source).absolute(), Path(controller).absolute()
    if source == controller or source.resolve() == controller.resolve():
        raise ValueError("Projection requires an independent clone; controller is read-only")
    for path in (*reversed(source.parents), source):
        if path.is_symlink() or getattr(path, 'is_junction', lambda: False)():
            raise ValueError("Refusing a redirected source clone")
    if _git(source, 'rev-parse', 'HEAD').decode().strip() != BASE:
        raise ValueError("Source clone must be detached at the fixed base commit")
    if Path(_git(source, 'rev-parse', '--show-toplevel').decode().strip()).resolve() != source.resolve():
        raise ValueError("Source must be its own Git checkout")
    return source


def _source_path(source, relative):
    path = source / relative
    current = source
    for part in Path(relative).parts:
        current = current / part
        if current.is_symlink() or getattr(current, 'is_junction', lambda: False)():
            raise ValueError(f"Refusing a redirected projection path: {relative}")
    if not path.resolve().is_relative_to(source.resolve()):
        raise ValueError(f"Projection path escapes source clone: {relative}")
    return path


def prepare(source: Path, mode: str, controller: Path = ROOT):
    source = _check_root(source, controller)
    if (source / MANIFEST).exists():
        raise ValueError("Source projection already exists; use verify without overwriting it")
    changed = _git(source, 'diff', '--name-only', BASE, '--', '.', ':(exclude)third-party').decode().splitlines()
    if changed:
        raise ValueError(f"Source clone has pre-existing tracked modifications: {changed[:8]}")
    untracked = _git(source, 'ls-files', '--others', '--exclude-standard', '-z').split(b'\0')
    if any(untracked):
        raise ValueError("Source clone has pre-existing untracked files")
    expected, originals = expected_files(source, mode, controller)
    for relative, data in originals.items():
        path = _source_path(source, relative)
        if path.read_bytes() != data:
            raise ValueError(f"Fixed-base preimage differs: {relative}")
    # All transformations and validations finish before the first source write.
    for relative, data in expected.items():
        path = source / relative
        if data is None:
            path.unlink()
        elif data != originals[relative]:
            path.write_bytes(data)
    proof = _proof(source, mode, expected, originals)
    (source / MANIFEST).write_text(json.dumps(proof, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return verify(source, mode, controller)


def verify(source: Path, mode: str, controller: Path = ROOT):
    source = _check_root(source, controller)
    expected, originals = expected_files(source, mode, controller)
    for relative, data in expected.items():
        path = _source_path(source, relative)
        if (path.exists() if data is None else not path.is_file() or path.read_bytes() != data):
            raise ValueError(f"Projected source changed: {relative}")
    changed = set(_git(source, 'diff', '--name-only', BASE, '--', '.', ':(exclude)third-party').decode().splitlines())
    if changed - set(expected):
        raise ValueError(f"Source changes outside projection manifest: {sorted(changed - set(expected))}")
    untracked = {name.decode() for name in _git(source, 'ls-files', '--others', '--exclude-standard', '-z').split(b'\0') if name}
    if untracked - {MANIFEST}:
        raise ValueError(f"Untracked files outside projection manifest: {sorted(untracked - {MANIFEST})[:8]}")
    if mode == 'clean-source':
        for folder in (D, C):
            for path in (source / folder).rglob('*'):
                if path.suffix in {'.h', '.hpp', '.cpp'}:
                    reject_timer_code(path.relative_to(source).as_posix(), path.read_text(encoding='utf-8'))
        for relative in (HOST, SELECTOR, COMBINE_ENTRY):
            reject_timer_code(relative, (source / relative).read_text(encoding='utf-8'))
    proof = _proof(source, mode, expected, originals)
    if json.loads((source / MANIFEST).read_text(encoding='utf-8')) != proof:
        raise ValueError("Source projection manifest or frozen fingerprint differs")
    return proof


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('_prepare', '_verify'))
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--mode', choices=('normal-off', 'clean-source'), required=True)
    parser.add_argument('--controller', type=Path, default=ROOT)
    args = parser.parse_args(argv)
    action = prepare if args.action == '_prepare' else verify
    print(json.dumps(action(args.source, args.mode, args.controller), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
