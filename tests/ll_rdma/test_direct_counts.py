"""Execute production count ownership, frame transport/reader and Vector body on CPU."""
from pathlib import Path
import os
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
COUNTS = ROOT / 'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_counts.h'


def block(source, marker):
    start = source.index('{', source.index(marker))
    depth = 1
    for end in range(start + 1, len(source)):
        depth += (source[end] == '{') - (source[end] == '}')
        if not depth:
            return source[start + 1:end]
    raise AssertionError(marker)


def native(source, tmp_path):
    zig = os.environ.get('DEEPEP_TEST_ZIG')
    if not zig:
        pytest.skip('Set DEEPEP_TEST_ZIG for C++ CPU validation')
    exe = tmp_path / 'direct-count.exe'
    subprocess.run([zig, 'c++', '-std=c++17', '-O1', '-UNDEBUG',
                    '-I' + str(ROOT / 'tests/ll_rdma/stubs'), '-I' + str(ROOT / 'csrc'),
                    str(source), '-o', str(exe)], check=True, timeout=180)
    result = subprocess.run([str(exe)], capture_output=True, text=True, check=True, timeout=90)
    assert 'PASS' in result.stdout
    print(result.stdout)


def test_direct_ownership_frames_and_retirement(tmp_path):
    native(Path(__file__).with_name('direct_count_test.cpp'), tmp_path)


def test_production_vector_counts(tmp_path):
    body = block(COUNTS.read_text(encoding='utf-8'), 'void ServerDedupCounts::CountRemoteRank(')
    # The actual production function runs with deferred Vector completion.
    source = tmp_path / 'vector-count.cpp'
    source.write_text(r'''
#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <random>
#include <vector>
enum class HardEvent { V_S };
enum class CMPMODE { EQ };
constexpr int PIPE_V=1;
static std::vector<std::function<void()>> pending;
static unsigned scalarWrites=0;
static void Flush() {for(auto& f:pending)f();pending.clear();}
template<int> static void PipeBarrier() {Flush();}
template<class T> struct LocalTensor {
    T* data;
    template<class U> LocalTensor<U> ReinterpretCast() const {return {reinterpret_cast<U*>(data)};}
    void SetValue(unsigned i,T value) const {assert(pending.empty());data[i]=value;++scalarWrites;}
};
struct Buf {
    void* data;
    template<class T> LocalTensor<T> Get() const {return {reinterpret_cast<T*>(data)};}
};
template<class T> void Duplicate(LocalTensor<T> out,T value,unsigned count) {
    pending.push_back([=]{for(unsigned i=0;i<count;++i)out.data[i]=value;});
}
template<class T> void Duplicate(LocalTensor<T> out,T value,uint64_t* mask,unsigned repeats,unsigned blockStride,unsigned repeatStride) {
    assert(blockStride==1 && repeatStride==8 && mask[1]==0);
    const auto bits=mask[0];
    pending.push_back([=]{for(unsigned r=0;r<repeats;++r)for(unsigned i=0;i<64;++i)
        if(bits&(uint64_t(1)<<i))out.data[r*64+i]=value;});
}
static void Compares(LocalTensor<uint8_t> mask,LocalTensor<int32_t> ids,int32_t expert,CMPMODE mode,unsigned n) {
    assert(mode==CMPMODE::EQ);
    pending.push_back([=]{
        std::memset(mask.data,0,(n+7)/8);
        for(unsigned i=0;i<n;++i)if(ids.data[i]==expert)mask.data[i/8]|=uint8_t(1U<<(i%8));
    });
}
struct GatherParams {uint16_t repeats,blockStride,repeatStride,maskStride;};
static void GatherMask(LocalTensor<int32_t> out,LocalTensor<int32_t> ids,LocalTensor<uint32_t> mask,
                       bool reduce,unsigned n,GatherParams p,uint64_t& count) {
    assert(reduce && p.repeats==1 && p.blockStride==1 && !p.repeatStride && !p.maskStride);
    pending.push_back([=,&count]{
        auto bits=reinterpret_cast<const uint8_t*>(mask.data);count=0;
        for(unsigned i=0;i<n;++i)if(bits[i/8]&(1U<<(i%8)))out.data[count++]=ids.data[i];
    });
}
struct Fixture {
    static constexpr unsigned UB_ALIGN_DATA_COUNT=8;
    unsigned statusCntAlign_,moeExpertNumPerRank_,expertIdsCnt_;
    LocalTensor<int32_t> statusTensor_,validExpertIdsTensor_;
    Buf subExpBuf_,dstExpBuf_;
    template<HardEvent> void SyncFunc() {Flush();}
    void Run(uint32_t first, LocalTensor<int32_t> frame, uint32_t frameRecords) {
''' + body + r'''
    }
};
int main() {
    std::mt19937 rng(27029);unsigned cases=0;
    constexpr int32_t guard=0x5a5a5a5a;
    for(unsigned experts: {32U,64U,384U,1024U})
      for(unsigned n: {1U,7U,8U,31U,63U,64U,65U,255U,256U,2048U,8192U,12000U})
        for(unsigned dst: {0U,15U,16U,31U}) for(unsigned pattern=0;pattern<4;++pattern) {
          const unsigned local=experts/32,aligned=(local+1+7)/8*8,routeWords=(n+63)/64*64;
          std::vector<int32_t> status(aligned*8+8,guard),ids(routeWords,guard),selected(routeWords+8,guard);
          std::vector<uint8_t> mask(routeWords*4+32,0xa5);
          for(unsigned i=0;i<n;++i)ids[i]=pattern==0?-1:pattern==1?int32_t(dst*local):
              pattern==2?int32_t((dst*local+i)%experts):int32_t(rng()%(experts+2))-1;
          Fixture f{aligned,local,n,{status.data()},{ids.data()},{mask.data()},{selected.data()}};
          const auto before=scalarWrites; f.Run(dst*local,{status.data()},aligned);
          assert(pending.empty() && scalarWrites-before==local);
          for(unsigned i=0;i<aligned;++i) {
            assert(uint32_t(status[i*8])==0x3f800000U);
            unsigned expected=0;
            if(i<local)for(unsigned j=0;j<n;++j)expected+=ids[j]==int32_t(dst*local+i);
            assert(status[i*8+1]==int32_t(expected));
            for(unsigned w=2;w<8;++w)assert(status[i*8+w]==0);
          }
          for(unsigned i=aligned*8;i<status.size();++i)assert(status[i]==guard);
          for(unsigned i=n;i<selected.size();++i)assert(selected[i]==guard);
          for(unsigned i=routeWords*4;i<mask.size();++i)assert(mask[i]==0xa5);
          ++cases;
        }
    std::cout<<"PASS production Vector counts: "<<cases<<" route/target cases, exactly one Scalar store per expert, initialized tail, UB guards\n";
}
''', encoding='utf-8')
    native(source, tmp_path)


def test_remote_protocol_is_wired_consistently():
    source = COUNTS.read_text(encoding='utf-8')
    send = block(source, 'void ServerDedupCounts::SendCnt()')
    assert 'CountAssignment(' in send and 'assignment.remoteRank' in send
    assert 'CountRemoteRank(first, remoteFrame, assignment.remoteRecords);' in send
    assert 'LlTransport::DirectCountFrame' in send
    assert 'auto remoteFrame = workBuf_.Get<int32_t>();' in send
    assert 'moeExpertNumPerRank_, dstExpBuf_.Get<uint8_t>()' in send
    assert 'routeExchange_' not in source and 'publishingSource_' not in source
    assert 'readPlan, directRanks_' in source
    assert 'epRankId_, source, expert, directRanks_' in source
    transport = (COUNTS.parent / 'll_count_frame_transport.h').read_text(encoding='utf-8')
    remote = block(transport, 'void DirectCountFrame(')
    assert remote.count('Put(') == remote.count('Quiet(') == 1
    assert remote.index('Sync<HardEvent::MTE3_S>') < remote.index('Put(') < remote.index('Quiet(')
