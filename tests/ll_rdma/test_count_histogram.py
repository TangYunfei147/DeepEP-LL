"""CPU semantic model of the production VF histogram; not CANN qualification."""
from pathlib import Path
import os
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_production_histogram_normalization_and_bins(tmp_path):
    zig = os.environ.get('DEEPEP_TEST_ZIG')
    if not zig:
        pytest.skip('Set DEEPEP_TEST_ZIG for native CPU checks')
    (tmp_path / 'kernel_operator.h').write_text(r'''
#pragma once
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#define __aicore__
#define __local_mem__
constexpr int PIPE_V=0;
namespace AscendC {
template<class T> struct LocalTensor {T* p;T* GetPhyAddr(){return p;}};
template<int> void PipeBarrier() {}
template<class T> void Maxs(LocalTensor<T> d,LocalTensor<T> s,T v,unsigned n){for(unsigned i=0;i<n;++i)d.p[i]=std::max(s.p[i],v);}
template<class T> void Mins(LocalTensor<T> d,LocalTensor<T> s,T v,unsigned n){for(unsigned i=0;i<n;++i)d.p[i]=std::min(s.p[i],v);}
template<class T> void Adds(LocalTensor<T> d,LocalTensor<T> s,T v,unsigned n){for(unsigned i=0;i<n;++i)d.p[i]=s.p[i]+v;}
template<auto F,class... Args> void VF_CALL(Args... a){F(a...);}
namespace MicroAPI {
template<class T> struct RegTensor {std::array<T,256/sizeof(T)> data;};
struct MaskReg {unsigned count;};
enum class HistogramsBinType {BIN0};
enum class HistogramsType {FREQUENCY};
template<class T> MaskReg CreateMask(){return {256/sizeof(T)};}
template<class T> MaskReg UpdateMask(uint32_t& left){unsigned n=std::min(left,unsigned(256/sizeof(T)));left-=n;return {n};}
template<class T> void Duplicate(RegTensor<T>& out,int v){out.data.fill(T(v));}
template<class T> void DataCopy(RegTensor<T>& out,T* p){std::memcpy(out.data.data(),p,256);}
template<class T> void DataCopy(T* p,RegTensor<T>& in,MaskReg mask){std::memcpy(p,in.data.data(),mask.count*sizeof(T));}
template<class T,class U,HistogramsBinType B,HistogramsType H>
void Histograms(RegTensor<U>& out,RegTensor<T>& in,MaskReg mask){
 for(unsigned i=0;i<mask.count;++i)if(in.data[i]<128){assert(out.data[in.data[i]]<65535);++out.data[in.data[i]];}
}
}
}
''', encoding='utf-8')
    source = tmp_path / 'hist.cpp'
    source.write_text(r'''
#include <vector>
#include <random>
#include <iostream>
#include <limits>
#include "kernels/dispatch/low_latency/device/ll_count_histogram.h"
#include "kernels/dispatch/low_latency/common/ll_core_config.h"
int main(){
 static_assert(DispatchDedup::kLlCoreProtocolVersion==29);
 std::mt19937 rng(29029);unsigned cases=0;
 for(unsigned world:{32U,64U,128U,256U})for(unsigned experts:{world,1024U})
 for(unsigned rank:{0U,world/2,world-1})for(unsigned n:{1U,7U,8U,63U,64U,65U,256U,2048U,8192U,12160U})
 for(unsigned pattern=0;pattern<5;++pattern){
  const unsigned local=experts/world,first=rank*local,words=(n+63)/64*64;
  std::vector<int32_t> ids(words,-1),tmp(words+8,0x12345678);
  std::vector<uint16_t> hist(136,0xbeef);
  for(unsigned i=0;i<n;++i)ids[i]=pattern==0?int32_t(first):pattern==1?-1:
    pattern==2?(i%2?INT32_MIN:INT32_MAX):pattern==3?int32_t(first+i%(local+2))-1:int32_t(rng()%(experts+2))-1;
  DispatchDedup::CountRankHistogram({ids.data()},{tmp.data()},{hist.data()},n,first,local);
  for(unsigned e=0;e<local;++e){unsigned expected=0;for(unsigned i=0;i<n;++i)expected+=ids[i]==int32_t(first+e);assert(hist[e+1]==expected);}
  unsigned sum=0;for(unsigned i=0;i<128;++i)sum+=hist[i];assert(sum==4*n);
  for(unsigned i=128;i<hist.size();++i)assert(hist[i]==0xbeef);
  for(unsigned i=n;i<tmp.size();++i)assert(tmp[i]==0x12345678);
  ++cases;
 }
 std::cout<<"PASS production histogram "<<cases<<" cases, bounds/invalid IDs/zero/hotspot/overflow guards\n";
}
''', encoding='utf-8')
    exe = tmp_path / 'hist.exe'
    subprocess.run([zig, 'c++', '-std=c++17', '-O1', '-UNDEBUG', '-DDEEPEP_LL_COUNT_HISTOGRAM=1',
                    '-I' + str(tmp_path), '-I' + str(ROOT / 'csrc'), str(source), '-o', str(exe)], check=True, timeout=180)
    result = subprocess.run([str(exe)], check=True, capture_output=True, text=True, timeout=30)
    assert 'PASS production histogram' in result.stdout
    print(result.stdout)


def test_switch_reaches_host_device_cache_and_identity():
    import build_support
    assert build_support.build_cache_key({'ll_count_histogram': False}) != build_support.build_cache_key({'ll_count_histogram': True})
    for path in ('CMakeLists.txt', 'cmake/Device.cmake', 'cmake/device/CMakeLists.txt', 'build_support.py'):
        assert 'DEEPEP_LL_COUNT_HISTOGRAM' in (ROOT / path).read_text(encoding='utf-8')
    counts = (ROOT / 'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_counts.h').read_text(encoding='utf-8')
    body = counts.split('void ServerDedupCounts::CountRemoteRank(', 1)[1]
    alternative = body.split('#ifdef DEEPEP_LL_COUNT_HISTOGRAM', 1)[1].split('#else', 1)[0]
    assert alternative.count('SyncFunc<HardEvent::V_S>') == 1
    assert alternative.index('SyncFunc<HardEvent::V_S>') < alternative.index('for (uint32_t i')
