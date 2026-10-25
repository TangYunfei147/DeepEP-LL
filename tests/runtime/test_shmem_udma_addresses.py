"""Official SHMEM address contexts: local, mapped peer and remote MR are distinct."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_actual_address_adapter_preserves_registered_offsets_and_rejects_raw_outputs(tmp_path):
    compiler = shlex.split(os.environ.get('CXX', ''))
    if not compiler:
        found = shutil.which('g++') or shutil.which('clang++')
        if not found:
            pytest.skip('C++ compiler required for UDMA address adapter')
        compiler = [found]
    source = tmp_path / 'addresses.cpp'
    source.write_text(r'''
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#define __aicore__
#define __gm__
#define __ubuf__
enum class aclshmemi_udma_opcode_t { UDMA_OP_WRITE, UDMA_OP_READ };
struct aclshmemx_udma_op_config_t { unsigned cqe; };
inline constexpr aclshmemx_udma_op_config_t config{0};
struct aclshmemi_sqe_ctx_t { uint64_t words[8]{}; };
struct aclshmemi_ubmem_info_t { uint64_t addr; };
struct aclshmemi_udma_addr_ctx_t { uint64_t heap_base, remote_mr_base; };
struct State { void* heap_base; uint64_t heap_size; };
inline State state{reinterpret_cast<void*>(0x100000),0x10000};
inline State* aclshmemi_get_state(){return &state;}
inline unsigned posts=0,aborts=0;
inline uint64_t last_input=0,last_head=0;
inline aclshmemi_udma_opcode_t last_op{};
inline void aclshmemi_kernel_abort(const char*){++aborts;}
inline void trap(){throw std::runtime_error("trap");}
template<typename T,aclshmemi_udma_opcode_t Op,typename Pointer,
         const aclshmemx_udma_op_config_t& Config>
void aclshmemi_udma_fill_sqe_base_ctx(Pointer out,uint8_t* local,
                                    aclshmemi_ubmem_info_t* memory,uint32_t head,
                                    const aclshmemi_udma_addr_ctx_t& context){
    // Exact new SDK address equation; all three VAs are intentionally distinct.
    assert(context.heap_base==reinterpret_cast<uint64_t>(state.heap_base));
    assert(context.remote_mr_base==memory->addr);
    last_input=reinterpret_cast<uint64_t>(local);last_head=head;last_op=Op;++posts;
    assert(last_input>=context.heap_base && last_input-context.heap_base<state.heap_size);
    out->words[5]=context.remote_mr_base+(last_input-context.heap_base);
}
#include "csrc/kernels/shmem_udma_address.hpp"

template<aclshmemi_udma_opcode_t Op> void check_operation(){
    constexpr uint64_t mapped=0x80000000, remote=0x50000000, local=0x100000;
    aclshmemi_ubmem_info_t memory{remote};
    for(uint64_t offset : {0ULL,8ULL,512ULL,4096ULL,65528ULL}){
        aclshmemi_sqe_ctx_t a{},b{},c{};
        assert((DeepepUdmaAddress::FillLocalSymmetric<uint64_t,Op,config>(
            &a,reinterpret_cast<uint8_t*>(local+offset),&memory,7)));
        assert(a.words[5]==remote+offset && last_input==local+offset && last_head==7 && last_op==Op);
        assert((DeepepUdmaAddress::FillMappedPeer<uint64_t,Op,config>(
            &b,reinterpret_cast<uint8_t*>(mapped+offset),mapped,&memory,8)));
        assert(b.words[5]==remote+offset && last_input==local+offset);
        assert((DeepepUdmaAddress::FillRegisteredRemote<uint64_t,Op,config>(
            &c,reinterpret_cast<uint8_t*>(remote+offset),&memory,9)));
        assert(c.words[5]==remote+offset && last_input==local+offset);
    }
}
template<class F>void rejected(F call){auto before=posts;bool caught=false;
    try{call();}catch(const std::runtime_error&){caught=true;}assert(caught && posts==before);}
int main(){
    check_operation<aclshmemi_udma_opcode_t::UDMA_OP_WRITE>();
    check_operation<aclshmemi_udma_opcode_t::UDMA_OP_READ>();
    aclshmemi_sqe_ctx_t header{};aclshmemi_ubmem_info_t memory{0x50000000};
    using Op=aclshmemi_udma_opcode_t;
    // A mapped VA must not be passed as a local symmetric pointer.
    rejected([&]{DeepepUdmaAddress::FillLocalSymmetric<uint64_t,Op::UDMA_OP_WRITE,config>(
        &header,reinterpret_cast<uint8_t*>(0x80000010),&memory,0);});
    // Ordinary remote framework GM and accidental double translation both fail.
    rejected([&]{DeepepUdmaAddress::FillRegisteredRemote<uint64_t,Op::UDMA_OP_WRITE,config>(
        &header,reinterpret_cast<uint8_t*>(0x90000010),&memory,0);});
    rejected([&]{DeepepUdmaAddress::FillRegisteredRemote<uint64_t,Op::UDMA_OP_READ,config>(
        &header,reinterpret_cast<uint8_t*>(0x100010),&memory,0);});
    rejected([&]{DeepepUdmaAddress::FillMappedPeer<uint64_t,Op::UDMA_OP_READ,config>(
        &header,reinterpret_cast<uint8_t*>(0x80010000),0x80000000,&memory,0);});
    rejected([&]{DeepepUdmaAddress::FillLocalSymmetric<uint64_t,Op::UDMA_OP_WRITE,config>(
        &header,reinterpret_cast<uint8_t*>(0x10ffff),&memory,0);});
    rejected([&]{DeepepUdmaAddress::FillLocalSymmetric<uint64_t,Op::UDMA_OP_WRITE,config>(
        &header,reinterpret_cast<uint8_t*>(0x100000),nullptr,0);});
    aclshmemi_ubmem_info_t missing{0};
    rejected([&]{DeepepUdmaAddress::FillLocalSymmetric<uint64_t,Op::UDMA_OP_WRITE,config>(
        &header,reinterpret_cast<uint8_t*>(0x100000),&missing,0);});
    state.heap_base=nullptr;
    rejected([&]{DeepepUdmaAddress::FillLocalSymmetric<uint64_t,Op::UDMA_OP_WRITE,config>(
        &header,reinterpret_cast<uint8_t*>(8),&memory,0);});
    state.heap_base=reinterpret_cast<void*>(0x100000);
    uint64_t result=123;
    assert(!DeepepUdmaAddress::Rebase(8,16,0,64,1,result));
    assert(!DeepepUdmaAddress::Rebase(16,16,0,64,0,result));
    assert(!DeepepUdmaAddress::Rebase(24,16,UINT64_MAX-7,64,1,result));
    assert(!DeepepUdmaAddress::Rebase(24,16,UINT64_MAX-15,64,9,result));
    assert(result==123 && aborts==8);
}
''', encoding='utf-8')
    executable = tmp_path / ('addresses.exe' if os.name == 'nt' else 'addresses')
    subprocess.run([*compiler, '-std=c++17', '-O2', '-UNDEBUG', '-Wall', '-Wextra', '-Werror',
                    '-iquote', str(ROOT), str(source), '-o', str(executable)], check=True)
    subprocess.run([str(executable)], check=True)


def test_all_custom_wqe_call_sites_use_explicit_address_kinds():
    sources = {
        'dispatch/fm_queue.hpp': ('FillLocalSymmetric',),
        'dispatch/fm_primary.hpp': ('FillRegisteredRemote',),
        'dispatch/fm_simt.hpp': ('FillRegisteredRemote',),
        'dispatch/wqe_cache.hpp': ('FillRegisteredRemote',),
        'combine/wqe_batch.hpp': ('FillLocalSymmetric', 'FillRegisteredRemote'),
    }
    for name, kinds in sources.items():
        text = (ROOT / 'csrc/kernels' / name).read_text(encoding='utf-8')
        assert 'aclshmemi_udma_fill_sqe_base_ctx' not in text
        assert all('DeepepUdmaAddress::' + kind in text for kind in kinds)
        assert 'header[5] = reinterpret_cast<uint64_t>(target)' not in text
    combine = (ROOT / 'csrc/kernels/combine/wqe_batch.hpp').read_text(encoding='utf-8')
    assert 'header[7] = header[5]' in combine  # Cache the translated MR base.
    helper = (ROOT / 'csrc/kernels/shmem_udma_address.hpp').read_text(encoding='utf-8')
    assert 'context{local_base,memory->addr}' in helper
    assert '{0U,0U}' not in helper and '{0,0}' not in helper


def test_ht_unregistered_output_is_rejected_before_device_work_without_blocking_ll():
    host = (ROOT / 'csrc/ops/dispatch.cpp').read_text(encoding='utf-8')
    begin = host.index('module.def("dispatch",')
    guard = host.index('HT Dispatch ordinary peer output', begin)
    assert guard < host.index('session.workspace()', begin) < host.index('dispatch_kernel_do(', begin)
    assert 'module.attr("ht_dispatch_live_peer_output_supported") = false' in host
    ll = (ROOT / 'csrc/ops/low_latency.cpp').read_text(encoding='utf-8')
    assert 'HT Dispatch ordinary peer output' not in ll
    combine = (ROOT / 'csrc/ops/combine.cpp').read_text(encoding='utf-8')
    assert 'HT Dispatch ordinary peer output' not in combine


def test_sdk_header_uses_real_local_to_remote_mr_translation():
    sdk = ROOT / 'third-party/shmem/src/device/gm2gm/engine/shmem_device_udma.hpp'
    if not sdk.is_file():
        pytest.skip('Initialize the pinned SHMEM submodule to check the adapter contract')
    text = sdk.read_text(encoding='utf-8')
    begin = text.index('ACLSHMEM_DEVICE void aclshmemi_udma_fill_sqe_base_ctx(')
    end = text.index('\ntemplate <', begin)
    body = text[begin:end]
    assert 'const aclshmemi_udma_addr_ctx_t& addr_ctx' in body
    assert 'sqe64[5] = reinterpret_cast<uint64_t>(aclshmemi_remote_mr_addr(' in body
    assert 'addr_ctx.heap_base' in body and 'addr_ctx.remote_mr_base' in body
