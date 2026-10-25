# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Compile the production lifecycle against link-time fault-injection doubles."""

from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.fixture(scope="module")
def lifecycle_binary(tmp_path_factory):
    root = Path(__file__).resolve().parents[2]
    compiler = shutil.which("g++")
    if compiler is None or not (root / "csrc").is_dir():
        pytest.skip("Host lifecycle compilation requires a C++17 compiler and source tree")
    temporary = tmp_path_factory.mktemp("native-lifecycle")
    binary = temporary / ("lifecycle.exe" if sys.platform == "win32" else "lifecycle")
    platform_flags = []
    if sys.platform == "win32":
        # The production target remains Linux. This CPU-only DLL lookup double
        # models the mainline SDK with no legacy peer-mask symbol available.
        (temporary / "dlfcn.h").write_text(
            '#pragma once\n#define RTLD_DEFAULT nullptr\n'
            'inline void* dlsym(void*, const char*) { return nullptr; }\n', encoding="utf-8")
        (temporary / "environment.h").write_text(
            '#pragma once\n#include <cstdlib>\n'
            'inline int setenv(const char* name,const char* value,int overwrite) {\n'
            ' if (!overwrite && std::getenv(name)) return 0;\n'
            ' return _putenv_s(name,value);\n}\n', encoding="utf-8")
        platform_flags = ['-I' + str(temporary), '-include', str(temporary / 'environment.h')]
    result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-O2",
            "-UNDEBUG",
            *platform_flags,
            "-I" + str(root / "tests/runtime/stubs"),
            "-I" + str(root / "csrc"),
            str(root / "csrc/runtime/session.cpp"),
            str(root / "tests/runtime/session_test.cpp"),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return binary


@pytest.mark.parametrize(
    "case",
    [
        "invalid",
        "init-fault",
        "allocation-fault",
        "memset-fault",
        "foreign",
        "abandoned",
        "close-fault",
        "normal",
        "udma",
        "megamoe",
        "low-latency",
        "sparse-missing",
        "ll-only-invalid",
        "ll-only-missing-masks",
        "ll-only-huge-missing-masks",
        "ll-rank-bias-16",
        "ll-rank-bias-32",
        "ll-rank-bias-64",
        "ll-rank-bias-128",
        "ll-rank-bias-256",
        "ep16-small-workspace",
        "huge-normal",
        "huge-rounded",
        "huge-exact",
        "huge-ll-guard",
        "huge-mpi",
        "huge-abandoned",
        "huge-overflow",
        "huge-granularity",
        "huge-wrong-granularity",
        "huge-allocate",
        "huge-reserve",
        "huge-map",
        "huge-access",
        "huge-init",
        "huge-translate",
        "huge-unaligned",
        "huge-memset",
        "huge-finalize",
        "huge-unmap",
        "huge-release-address",
        "huge-free-physical",
    ],
)
def test_native_lifecycle_faults(lifecycle_binary, case):
    result = subprocess.run([lifecycle_binary, case], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
