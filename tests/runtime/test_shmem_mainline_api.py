"""Compile the pinned public SHMEM attribute contract without a CANN runtime."""
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = r'''
#include <cassert>
#include "host/shmem_host_def.h"
int main() {
    aclshmemx_init_attr_t attr{};
    const auto version = attr.option_attr.version;
    assert(version == ACLSHMEM_INIT_ATTR_VERSION && (uint32_t(version) >> 16) == 2U);
    attr.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_MTE;
    attr.option_attr.shm_init_timeout = 300;
    assert(attr.option_attr.version == version);
    assert(attr.runtime_config.udma_shared_jetty == ACLSHMEM_CONFIG_DEFAULT);
    for (auto reserved : attr.runtime_config.reserved) assert(reserved == 0U);
    attr.runtime_config.udma_shared_jetty = ACLSHMEM_CONFIG_ENABLED;
    assert(attr.option_attr.version == version);
    assert(attr.runtime_config.udma_shared_jetty == ACLSHMEM_CONFIG_ENABLED);
}
'''


def test_actual_sdk_attribute_version_and_runtime_config(tmp_path):
    sdk = ROOT / 'third-party/shmem/include'
    if not (sdk / 'host/shmem_host_def.h').is_file():
        pytest.skip('Initialize the pinned SHMEM submodule for its public-header contract')
    compiler = shutil.which('g++') or shutil.which('clang++')
    if compiler is None:
        pytest.skip('C++ compiler required for the actual public SHMEM header')
    acl = tmp_path / 'acl'
    acl.mkdir()
    (acl / 'acl_rt.h').write_text('#pragma once\nusing aclrtStream = void*;\n')
    source = tmp_path / 'attributes.cpp'
    source.write_text(SOURCE)
    exe = tmp_path / 'attributes'
    subprocess.run([compiler, '-std=c++17', '-UNDEBUG', '-I', str(tmp_path), '-I', str(sdk),
                    str(source), '-o', str(exe)], check=True, timeout=60)
    subprocess.run([str(exe)], check=True, timeout=10)
