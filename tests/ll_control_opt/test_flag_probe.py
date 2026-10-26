"""Run the production flag VF against Decode using a checked CPU SIMD mock.

These tests verify arithmetic, full-row reads and masked output bounds. They do
not validate Bisheng syntax, vector scheduling or live communication visibility.
"""
from pathlib import Path
import os
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "csrc/kernels/dispatch/low_latency/device/dispatch_flag_probe.h"


def test_production_flag_vf_matches_packet_decoder(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        pytest.skip("C++ compiler required for production VF semantic mock")
    api = tmp_path / "c_api"
    api.mkdir()
    (api / "asc_simd.h").write_text(MOCK, encoding="utf-8")
    source = tmp_path / "flag_probe.cpp"
    source.write_text(NATIVE, encoding="utf-8")
    exe = tmp_path / ("flag_probe.exe" if os.name == "nt" else "flag_probe")
    built = subprocess.run(
        [compiler, "-std=c++17", "-O2", "-UNDEBUG", "-I", str(tmp_path),
         "-iquote", str(ROOT), str(source), "-o", str(exe)],
        capture_output=True, text=True,
    )
    assert built.returncode == 0, built.stdout + built.stderr
    checked = subprocess.run([str(exe)], capture_output=True, text=True)
    assert checked.returncode == 0, checked.stdout + checked.stderr


MOCK = r"""
#pragma once
#include <array>
#include <cassert>
#include <cstdint>
#define __simd_vf__
#define __ubuf__
template<class T> struct Vec { std::array<T,64> v{}; };
using vector_float = Vec<float>;
using vector_uint32_t = Vec<uint32_t>;
struct vector_bool { std::array<bool,64> v{}; };
constexpr int PAT_ALL = 1;
inline unsigned loads = 0, stores = 0;
inline vector_bool asc_create_mask_b32(int) {
    vector_bool r; r.v.fill(true); return r;
}
inline vector_bool asc_update_mask_b32(uint32_t& n) {
    vector_bool r;
    for (unsigned i=0; i<64; ++i) r.v[i] = i<n;
    n = n>64 ? n-64 : 0; return r;
}
template<class T> void asc_duplicate_scalar(Vec<T>& d,T value) {
    d.v.fill(value);
}
inline void asc_loadalign(vector_float& d,float* p) {
    assert((uintptr_t(p)&31U)==0); ++loads;
    for(unsigned i=0; i<64; ++i) d.v[i]=p[i];
}
inline void asc_reduce_sum(vector_float& d,vector_float s,vector_bool m) {
    for(unsigned i=0; i<64; ++i) if(!m.v[i]) s.v[i]=0;
    for(unsigned size=64; size>1; size/=2)
        for(unsigned i=0; i<size/2; ++i) s.v[i]=s.v[2*i]+s.v[2*i+1];
    d.v.fill(0); d.v[0]=s.v[0];
}
inline vector_bool asc_eq_scalar(vector_float s,float value,vector_bool m) {
    for(unsigned i=0; i<64; ++i) m.v[i]=m.v[i] && s.v[i]==value;
    return m;
}
inline void asc_select(vector_uint32_t& d,vector_uint32_t a,
                       vector_uint32_t b,vector_bool m) {
    for(unsigned i=0; i<64; ++i) d.v[i]=m.v[i] ? a.v[i] : b.v[i];
}
inline void asc_add(vector_uint32_t& d,vector_uint32_t a,
                    vector_uint32_t b,vector_bool m) {
    for(unsigned i=0; i<64; ++i) d.v[i]=m.v[i] ? a.v[i]+b.v[i] : 0;
}
inline void asc_gather(vector_uint32_t& d,vector_uint32_t s,
                       vector_uint32_t index) {
    for(unsigned i=0; i<64; ++i) {
        assert(index.v[i]<64); d.v[i]=s.v[index.v[i]];
    }
}
inline void asc_storealign(uint32_t* p,vector_uint32_t s,vector_bool m) {
    assert((uintptr_t(p)&31U)==0); ++stores;
    for(unsigned i=0; i<64; ++i) if(m.v[i]) p[i]=s.v[i];
}
"""

NATIVE = r"""
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include "csrc/kernels/common/ep_packet_receive_layout.h"
#include "csrc/kernels/dispatch/low_latency/device/dispatch_flag_probe.h"

alignas(256) float flags[16*64];
alignas(256) uint32_t output[128];
uint32_t state=9821U;
uint32_t random_u32() { state=state*1664525U+1013904223U; return state; }

void check(uint32_t count,bool references=false) {
    uint32_t expectedReady=0, expectedKeep=0, expectedReference=0;
    for(unsigned row=0; row<count; ++row) {
        float sum=0;
        for(unsigned j=0; j<64; ++j) sum+=flags[row*64+j];
        if(references && flags[row*64]==DispatchDedup::PacketReceiveLayout::kReference) {
            expectedReady|=1U<<row;expectedReference|=1U<<row;continue;
        }
        const auto layout=DispatchDedup::PacketReceiveLayout::Decode(sum,8U);
        if(layout!=DispatchDedup::PacketReceiveLayout::State::Pending)
            expectedReady|=1U<<row;
        if(layout==DispatchDedup::PacketReceiveLayout::State::KeepFlags)
            expectedKeep|=1U<<row;
    }
    std::array<float,16*64> before;
    std::memcpy(before.data(), flags, sizeof(flags));
    std::fill(std::begin(output),std::end(output),0xcafe1234U);
    loads=stores=0;
    if(references) DispatchDedup::BuildReceiveFlagMasks(flags,output,count,8U,true);
    else DispatchDedup::BuildReceiveFlagMasks(flags,output,count,8U);
    assert(output[0]==expectedReady && output[1]==expectedKeep && output[2]==expectedReference);
    assert(loads==count && stores==1);
    for(unsigned i=3; i<128; ++i) assert(output[i]==0xcafe1234U);
    assert(std::memcmp(before.data(),flags,sizeof(flags))==0);
}

int main() {
    // Every batch length, including empty input and all tail lengths. Random
    // pending/compact/keep mixtures exercise all sixteen packet-bit positions.
    for(unsigned count=0; count<=16; ++count)
        for(unsigned round=0; round<512; ++round) {
            for(unsigned row=0; row<16; ++row) {
                std::fill(flags+row*64,flags+(row+1)*64,1.0f);
                const unsigned mode=random_u32()%5U;
                if(mode==0) flags[row*64+random_u32()%64U]=0;
                if(mode==1) flags[row*64]=1.5f;
                if(mode==2) {
                    flags[row*64]=1.5f;
                    flags[row*64+1+random_u32()%63U]=0;
                }
                if(mode==3)
                    for(unsigned j=0; j<64; ++j)
                        flags[row*64+j]=(random_u32()%4U)*0.5f;
            }
            check(count);check(count,true);
        }
    // Missing any one of the 64 words prevents a complete packet. Exercise
    // each packet bit independently, not just the marker/first flag block.
    for(unsigned row=0; row<16; ++row)
        for(unsigned word=0; word<64; ++word) {
            std::fill(std::begin(flags),std::end(flags),1.0f);
            flags[row*64+word]=0;
            check(16);
            assert((output[0]&(1U<<row))==0);
            check(16,true);
            assert((output[0]&(1U<<row))==0);
            if(word!=0) {
                flags[row*64]=1.5f;
                check(16);
                assert((output[0]&(1U<<row))==0);
                check(16,true);
                assert((output[0]&(1U<<row))==0);
            }
        }
    // Reject values between accepted sums, both infinities, and NaN. The
    // production equality decoder must not become a threshold comparison.
    for(float value : {1.25f,1.75f,
                       std::numeric_limits<float>::infinity(),
                       -std::numeric_limits<float>::infinity(),
                       std::numeric_limits<float>::quiet_NaN()}) {
        std::fill(std::begin(flags),std::end(flags),1.0f);
        flags[0]=value;
        check(16);
        assert((output[0]&1U)==0);
        check(16,true);
        assert((output[0]&1U)==0);
    }
    // Reference readiness depends only on the first word, even if the unused
    // block flags have old values or NaNs. Each batch mixes ordinary/reference
    // and incomplete packets; none of the reference bits may enter KeepFlags.
    for(unsigned count=0;count<=16;++count) for(unsigned round=0;round<64;++round) {
        for(unsigned row=0;row<16;++row) {
            std::fill(flags+row*64,flags+(row+1)*64,1.0f);
            const unsigned type=(row+round)%4;
            if(type==0) {
                for(unsigned word=0;word<64;++word)
                    flags[row*64+word]=(round%3==0 ? 0.0f : round%3==1 ? 1.5f : std::numeric_limits<float>::quiet_NaN());
                flags[row*64]=-1.0f;
            } else if(type==1) flags[row*64]=1.5f;
            else if(type==2) flags[row*64+17]=0.0f;
        }
        check(count);check(count,true);
        assert((output[1]&output[2])==0);
    }
    for(unsigned word=1;word<64;++word) {
        std::fill(std::begin(flags),std::end(flags),1.0f);flags[word]=-1.0f;
        check(16,true);assert(!(output[0]&1U)&&output[2]==0U);
    }
    // Reference takes precedence even if unused words happen to sum to the
    // ordinary KeepFlags value; the legacy decoder keeps its exact semantics.
    std::fill(std::begin(flags),std::end(flags),1.0f);flags[0]=-1.0f;flags[1]=3.5f;
    check(16);assert((output[1]&1U)&&output[2]==0U);
    check(16,true);assert(!(output[1]&1U)&&(output[2]&1U));
}
"""
