"""Boundary models and CPU execution of the production VF through a C API mock.

The mock checks arithmetic, predicate packing and masked stores. It does not
validate Bisheng syntax, vector-register allocation or NPU pipeline ordering.
"""
from collections import Counter
from pathlib import Path
import random
import shutil
import subprocess

import pytest

ROOT=Path(__file__).resolve().parents[2]
LIMIT=32768


@pytest.mark.parametrize("size",[0,1,63,64,65,32767,32768,32769,65535,65536,70001])
@pytest.mark.parametrize("width",[1,63,64,65,127,128,129,257])
def test_bounded_histogram_and_tail_masks(size,width):
    start=383
    rng=random.Random(size+width)
    # Concentrated traffic can exceed 65535 in one bin; out-of-window values
    # include uint8 aliases and unsigned subtraction underflow.
    ids=[start if i%8 else rng.choice([start-1,start+width,start+256,start+width-1])
         for i in range(size)]
    result=[0]*width
    for first in range(0,width,128):
        bins=min(128,width-first)
        for begin in range(0,size,LIMIT):
            hist=[0]*128
            stop=min(begin+LIMIT,size)
            for pos in range(begin,stop,64):
                chunk=ids[pos:min(pos+64,stop)]
                chunk+= [start+first]*(64-len(chunk))  # poison masked tail
                for lane,expert in enumerate(chunk):
                    delta=(expert-start-first)&0xffffffff
                    if pos+lane<stop and delta<bins:
                        hist[delta]+=1
            assert max(hist)<=LIMIT
            for i in range(bins):
                result[first+i]+=hist[i]
    expected=Counter(ids)
    assert result==[expected[start+i] for i in range(width)]


def test_production_vector_function_with_cpu_intrinsic_mock(tmp_path):
    compiler=shutil.which("g++")
    if not compiler:
        pytest.skip("g++ unavailable; target compiler must run the actual VF arithmetic mock")
    api=tmp_path/"c_api"
    api.mkdir()
    (api/"asc_simd.h").write_text(MOCK,encoding="utf-8")
    source=tmp_path/"histogram.cpp"
    source.write_text(NATIVE,encoding="utf-8")
    exe=tmp_path/"histogram"
    subprocess.run([compiler,"-std=c++17","-O2","-I",str(tmp_path),"-I",str(ROOT),
                    str(source),"-o",str(exe)],check=True,capture_output=True,text=True)
    subprocess.run([str(exe)],check=True,capture_output=True,text=True)


MOCK=r"""
#pragma once
#include <array>
#include <cstdint>
#define __simd_callee__
#define __simd_vf__
#define __ubuf__
constexpr int PAT_ALL=0;
template<class T> struct Vec { std::array<T,256/sizeof(T)> v{}; };
using vector_uint32_t=Vec<uint32_t>;
using vector_uint16_t=Vec<uint16_t>;
using vector_uint8_t=Vec<uint8_t>;
struct vector_bool { std::array<bool,256> v{}; };
template<class T> void asc_duplicate_scalar(Vec<T>& dst,T x){dst.v.fill(x);}
inline vector_bool asc_create_mask_b32(int){
 vector_bool m; for(unsigned i=0;i<64;++i)m.v[i*4]=true; return m;
}
inline vector_bool asc_update_mask_b32(uint32_t& n){
 vector_bool m; for(unsigned i=0;i<64 && i<n;++i)m.v[i*4]=true;
 n=n>64?n-64:0; return m;
}
inline void asc_loadalign(vector_uint32_t& dst,uint32_t* p){
 for(unsigned i=0;i<64;++i)dst.v[i]=p[i];
}
inline void asc_sub(vector_uint32_t& d,vector_uint32_t a,vector_uint32_t b,vector_bool m){
 for(unsigned i=0;i<64;++i)d.v[i]=m.v[4*i]?a.v[i]-b.v[i]:0;
}
inline vector_bool asc_lt_scalar(vector_uint32_t a,uint32_t b,vector_bool m){
 for(unsigned i=0;i<64;++i)m.v[4*i]=m.v[4*i] && a.v[i]<b; return m;
}
inline void asc_pack_to_low(vector_uint16_t& d,vector_uint32_t a){
 d.v.fill(0);for(unsigned i=0;i<64;++i)d.v[i]=uint16_t(a.v[i]);
}
inline void asc_pack_to_low(vector_uint8_t& d,vector_uint16_t a){
 d.v.fill(0);for(unsigned i=0;i<128;++i)d.v[i]=uint8_t(a.v[i]);
}
inline vector_bool asc_pack_to_low(vector_bool a){
 vector_bool d;for(unsigned i=0;i<128;++i)d.v[i]=a.v[2*i];return d;
}
inline void asc_frequency_histogram_bin0(vector_uint16_t& d,vector_uint8_t a,vector_bool m){
 for(unsigned i=0;i<256;++i)if(m.v[i] && a.v[i]<128)++d.v[a.v[i]];
}
inline void asc_unpack_lower(vector_uint32_t& d,vector_uint16_t a){
 for(unsigned i=0;i<64;++i)d.v[i]=a.v[i];
}
inline void asc_unpack_upper(vector_uint32_t& d,vector_uint16_t a){
 for(unsigned i=0;i<64;++i)d.v[i]=a.v[64+i];
}
inline void asc_add(vector_uint32_t& d,vector_uint32_t a,vector_uint32_t b,vector_bool m){
 for(unsigned i=0;i<64;++i)d.v[i]=m.v[4*i]?a.v[i]+b.v[i]:0;
}
inline void asc_storealign(uint32_t* p,vector_uint32_t a,vector_bool m){
 for(unsigned i=0;i<64;++i)if(m.v[4*i])p[i]=a.v[i];
}
"""

NATIVE=r"""
#include <cassert>
#include <vector>
#include "csrc/kernels/dispatch/low_latency/device/dispatch_frequency_histogram.h"
int main(){
 for(unsigned n:{0U,1U,63U,64U,65U,32767U,32768U,32769U,65535U,65536U,70001U})
 for(unsigned width:{1U,63U,64U,65U,127U,128U,129U,257U})
 for(unsigned pattern=0;pattern<3;++pattern){
  const unsigned start=383,guard=0xdeadbeefU;
  std::vector<uint32_t> input((n+63)/64*64+64,start),expected(width),out(width+128,guard);
  for(unsigned i=0;i<n;++i){
   input[i]=pattern==0?start:pattern==1?start+i%width:
               i%4==0?start-1:i%4==1?start+width:i%4==2?start+256:start;
   if(input[i]>=start && input[i]<start+width)++expected[input[i]-start];
  }
  if(n<=DispatchDedup::kLlHistogramSegment)
   DispatchDedup::CountFrequency<true>(out.data(),input.data(),n,start,width);
  else DispatchDedup::CountFrequency<false>(out.data(),input.data(),n,start,width);
  for(unsigned i=0;i<width;++i)assert(out[i]==expected[i]);
  for(unsigned i=width;i<out.size();++i)assert(out[i]==guard);
 }
}
"""
