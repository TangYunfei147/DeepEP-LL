"""Compare raw-WQE compatibility against the actual pinned SDK field builder.

Only address qualifiers/device intrinsics are stubbed; structures, flag helpers,
the symmetric-MR conversion and the reference builder come from the submodule.
This is CPU semantic validation, not CANN compilation or UDMA qualification.
"""
from pathlib import Path
import os
import re
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
SDK = ROOT / "third-party/shmem"
HEADERS = ("dispatch/fm_primary.hpp", "dispatch/fm_queue.hpp", "dispatch/fm_simt.hpp",
           "dispatch/wqe_cache.hpp", "combine/wqe_batch.hpp")


def braced(text, needle, template=False, declaration=False):
    start = text.index(needle)
    if template:
        start = text.rfind("template <", 0, start)
        assert start >= 0
    begin = text.index("{", start)
    depth = 1
    end = begin + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    if declaration:
        end = text.index(";", end) + 1
    return text[start:end] + "\n"


def test_only_explicit_raw_builders_use_compatibility():
    sources = [(ROOT / "csrc/kernels" / name).read_text(encoding="utf-8") for name in HEADERS]
    assert sum(source.count("DeepepFillSqeBaseRawRemoteAddr<") for source in sources) == 7
    assert all("aclshmemi_udma_fill_sqe_base_ctx<" not in source for source in sources)
    assert all('#include "../udma_raw_remote.hpp"' in source for source in sources)
    assert "DeepepSymmetricRemoteMrAddr(flag, remote_mem)" in sources[1]
    assert "DeepepSymmetricRemoteMrAddr(workspace, remote_mem)" in sources[4]
    assert "DeepepSymmetricRemoteMrAddr" not in sources[0] + sources[2] + sources[3]


def test_actual_sdk_sqe_layout_and_raw_address_semantics(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("A C++ compiler is required for actual SDK field-builder comparison")
    native = SDK / "src/device/gm2gm/engine/shmem_device_udma.hpp"
    if not native.is_file():
        pytest.skip("Initialize the pinned SHMEM submodule for the real SDK reference")
    implementation = native.read_text(encoding="utf-8")
    structures = (SDK / "src/device/gm2gm/engine/shmemi_device_udma.h").read_text(encoding="utf-8")
    public = (SDK / "include/device/gm2gm/engine/shmem_device_udma.h").read_text(encoding="utf-8")
    common = (SDK / "src/device/shmemi_device_common.hpp").read_text(encoding="utf-8")
    declarations = "".join(braced(structures, name, declaration=True) for name in (
        "enum class aclshmemi_udma_opcode_t", "struct aclshmemi_ubmem_info_t",
        "struct aclshmemi_sqe_ctx_t", "struct aclshmemi_sge_ctx_t"))
    declarations += public[public.index("typedef struct {"):public.index("/**", public.index("} aclshmemx_udma_op_config_t;"))]
    declarations += braced(implementation, "struct aclshmemi_udma_addr_ctx_t", declaration=True)
    declarations += "\n".join(re.findall(r"^constexpr uint32_t ACLSHMEM_UDMA_(?:REMOTE_JETTY_TYPE|FLAG_[A-Z_]+) = .*;", implementation, re.M))
    declarations += "\n" + braced(implementation, "ACLSHMEM_DEVICE constexpr void aclshmemi_udma_check_config()", template=True)
    declarations += braced(implementation, "ACLSHMEM_DEVICE constexpr uint32_t aclshmemi_udma_build_flag()", template=True)
    declarations += braced(implementation, "ACLSHMEM_DEVICE constexpr uint32_t aclshmemi_udma_get_reduce_attr()", template=True)
    declarations += braced(implementation, "ACLSHMEM_DEVICE void assert_remote_jetty_type_valid(")
    declarations += braced(common, "ACLSHMEM_DEVICE void aclshmemi_debug_assert_remote_mr_addr(")
    declarations += braced(common, "ACLSHMEM_DEVICE __gm__ uint8_t* aclshmemi_remote_mr_addr(")
    declarations += braced(implementation, "ACLSHMEM_DEVICE void aclshmemi_udma_fill_sqe_base_ctx(", template=True)
    preamble = r'''
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#define __aicore__
#define __gm__
#define __ubuf__
#define ACLSHMEM_DEVICE inline
#define ACLSHMEM_DEBUG_FUNC(function, ...) function(__VA_ARGS__)
namespace AscendC {
using ::printf;
namespace Std { using std::conditional; }
template<class A, class B> using IsSameType = std::is_same<A, B>;
}
inline void trap() { throw std::runtime_error("SDK trap"); }
inline void aclshmemi_kernel_abort(const char*) { trap(); }
struct DeviceState { void* heap_base; uint64_t heap_size; };
DeviceState current_state{reinterpret_cast<void*>(0x100000000ULL), 0x1000000ULL};
inline DeviceState* aclshmemi_get_state() { return &current_state; }
'''
    tests = r'''
#include "csrc/kernels/udma_raw_remote.hpp"
inline constexpr aclshmemx_udma_op_config_t no_cqe_ro{0,0,0,ACLSHMEMX_UDMA_ODR_RO};
inline constexpr aclshmemx_udma_op_config_t cqe_ro{1,0,0,ACLSHMEMX_UDMA_ODR_RO};
inline constexpr aclshmemx_udma_op_config_t no_cqe_so{0,0,0,ACLSHMEMX_UDMA_ODR_SO};
inline constexpr aclshmemx_udma_op_config_t cqe_so{1,0,1,ACLSHMEMX_UDMA_ODR_SO};
template<class T, const aclshmemx_udma_op_config_t& config> void check() {
    uint64_t eid[2]{0x123456789abcdef0ULL,0xfedcba9876543210ULL};
    aclshmemi_ubmem_info_t remote{};
    remote.rmt_jetty_type=ACLSHMEM_UDMA_REMOTE_JETTY_TYPE;
    remote.eid_addr=reinterpret_cast<uint64_t>(eid);
    remote.addr=0x700000000000ULL;
    constexpr uint64_t mte_mapping=0x500000000000ULL;
    const uint64_t local_base=reinterpret_cast<uint64_t>(current_state.heap_base);
    for (unsigned i=0;i<512;++i) {
        remote.token_value_valid=bool(i&1U);remote.target_hint=uint8_t(i);
        remote.tpn=0xf1230000U+i;remote.tid=0xf4560000U+i;
        remote.rmt_token_value=0xcdef0000U+i;
        const uint64_t offset=uint64_t(i)*128U;
        auto* local=reinterpret_cast<uint8_t*>(local_base+offset);
        auto* mr=DeepepSymmetricRemoteMrAddr(local,&remote);
        assert(reinterpret_cast<uint64_t>(mr)==remote.addr+offset);
        assert(reinterpret_cast<uint64_t>(mr)!=mte_mapping+offset);
        assert(reinterpret_cast<uint64_t>(mr)!=local_base+offset);
        uint64_t actual[8],expected[8];
        for(unsigned word=0;word<8;++word) actual[word]=expected[word]=0xa5a5a5a5a5a5a5a5ULL;
        const aclshmemi_udma_addr_ctx_t context{local_base,remote.addr};
        aclshmemi_udma_fill_sqe_base_ctx<T,aclshmemi_udma_opcode_t::UDMA_OP_WRITE,
            aclshmemi_sqe_ctx_t*,config>(reinterpret_cast<aclshmemi_sqe_ctx_t*>(expected),local,&remote,i,context);
        DeepepFillSqeBaseRawRemoteAddr<T,aclshmemi_udma_opcode_t::UDMA_OP_WRITE,
            aclshmemi_sqe_ctx_t*,config>(reinterpret_cast<aclshmemi_sqe_ctx_t*>(actual),mr,&remote,i);
        assert(std::memcmp(actual,expected,sizeof(actual))==0);
        // Framework destinations are exchanged raw VAs, outside the symmetric
        // heap. Their high bits must survive, without a second MR translation.
        auto* raw=reinterpret_cast<uint8_t*>(0x7e1234560000ULL+offset);
        DeepepFillSqeBaseRawRemoteAddr<T,aclshmemi_udma_opcode_t::UDMA_OP_WRITE,
            aclshmemi_sqe_ctx_t*,config>(reinterpret_cast<aclshmemi_sqe_ctx_t*>(actual),raw,&remote,i);
        assert(actual[5]==reinterpret_cast<uint64_t>(raw));
        assert(std::memcmp(actual,expected,5*sizeof(uint64_t))==0);
        assert(actual[6]==expected[6] && actual[7]==expected[7]);
        bool rejected=false;
        try {
            aclshmemi_udma_fill_sqe_base_ctx<T,aclshmemi_udma_opcode_t::UDMA_OP_WRITE,
                aclshmemi_sqe_ctx_t*,config>(reinterpret_cast<aclshmemi_sqe_ctx_t*>(expected),raw,&remote,i,context);
        } catch(const std::runtime_error&) { rejected=true; }
        assert(rejected);  // Demonstrates why mechanically adding ctx is wrong.
    }
}
int main() {
    check<uint8_t,no_cqe_ro>();check<uint8_t,cqe_ro>();
    check<uint64_t,no_cqe_so>();check<uint64_t,cqe_so>();
}
'''
    source = tmp_path / "wqe.cpp"
    binary = tmp_path / ("wqe.exe" if os.name == "nt" else "wqe")
    source.write_text(preamble + declarations + tests, encoding="utf-8")
    subprocess.run([compiler, "-std=c++17", "-O2", "-UNDEBUG", "-I", str(ROOT), str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
