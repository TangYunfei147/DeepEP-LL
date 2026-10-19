"""CPU execution of production LL routing and ready-protocol methods.

These checks exercise control-flow/addressing only, not NPU DMA ordering.
"""
from pathlib import Path
import subprocess
import shutil
import pytest

ROOT = Path(__file__).resolve().parents[2]


def method(text, name):
    start = text.index('__aicore__ inline void ' + name + '(')
    opening = text.index('{', start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start:end].replace('__aicore__ ', '')


@pytest.mark.parametrize("group", range(1, 17))
def test_production_routing_and_ready_addresses(tmp_path, group):
    if not shutil.which("g++"): pytest.skip("g++ required for production ready-protocol model")
    sender = (ROOT/'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h').read_text()
    combine = (ROOT/'csrc/kernels/combine/low_latency/combine_server_dedup_kernel.h').read_text()
    source = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <map>
#include <vector>
#include "csrc/kernels/dispatch/low_latency/common/ep_memory_server_dedup_protocol.h"
#include "csrc/kernels/dispatch/low_latency/common/ep_memory_server_dedup_layout.h"
#include "csrc/kernels/combine/low_latency/combine_layout.h"
#include "csrc/kernels/common/ep_server_order.h"
using namespace DispatchDedup;
struct Args { unsigned rank, rankSize; };
template<class T> struct Tensor {
    T* p;
    T GetValue(unsigned i) { return p[i]; }
    void SetValue(unsigned i,T v) { p[i]=v; }
    Tensor operator[](unsigned i) { return {p+i}; }
};
struct Route {
    Args* args_;
    unsigned start_=0,k_=0,localExperts_=4;
    Tensor<int32_t> routeIds_;
    Tensor<uint32_t> serverState_,firstHopMasks_;
''' + method(sender, 'Classify') + r'''
};
enum class HardEvent { V_S, V_MTE3, S_MTE3 };
template<class T> void Duplicate(Tensor<T> tensor,T value,unsigned count) {
    std::fill(tensor.p,tensor.p+count,value);
}
extern unsigned modelActiveRecv;
struct Copy { unsigned blocks,bytes,srcStride,dstStride,reserved; };
void DataCopyPad(Tensor<float> dst,Tensor<float> src,Copy c) {
    for(unsigned b=0;b<c.blocks;++b) {
        if(src.p[b*(c.bytes+c.srcStride)/4]==1.0f && b<modelActiveRecv) assert(dst.p[b*(c.bytes+c.dstStride)/4]==0.0f);
        std::copy_n(src.p+b*(c.bytes+c.srcStride)/4,c.bytes/4,dst.p+b*(c.bytes+c.dstStride)/4);
    }
}
unsigned modelActiveRecv=32;
using CombineDedup::kCombinePrepareRowBytes;
using CombineDedup::kCombineNotifyRows;
struct Ready {
    Args* args_;
    unsigned core_;
    std::vector<std::vector<float>>* gm;
    unsigned coreNum_,sendCores_,notifyBegin_,notifyCores_;
    uint64_t selected_=0;
    Tensor<float> preparePublish_{scratch+3072};
    float scratch[4096]{};
    Tensor<float> prepareInput_{scratch},prepareWork_{scratch},prepareSum_{scratch};
    Tensor<float> prepareZero_{scratch+1024},notifyUb_{scratch+2048};
    Tensor<float> readyInput_{scratch},readyWork_{scratch},readySum_{scratch},readyZero_{scratch+2560};
    template<HardEvent E> void Sync() {}
    Tensor<float> Control(unsigned rank,uint64_t offset) {
        assert(rank<gm->size() && offset<32768);
        return {(*gm)[rank].data()+offset/4};
    }
    void WaitFlags(Tensor<float> mem,Tensor<float>,Tensor<float>,Tensor<float>,unsigned count) {
        for(unsigned i=0;i<count;++i) assert(mem.GetValue(i*8)==1.0f);
    }
''' + 'void Publish() {\n' + 'Duplicate<float>(preparePublish_, 0.0f, 64U);' + method(combine, 'CopyLocalPendingTokens').split('Duplicate<float>(preparePublish_, 0.0f, 64U);', 1)[1] + '\n' + method(combine, 'NotifyServerCard') + '\n' + method(combine, 'WaitServerReady') + r'''
};
int main() {
    constexpr unsigned G=kLlRanksPerGroup;
    for(unsigned world: {16U,32U,64U,128U,256U,16U*G}) {
        if(world%G) {
            EpMemoryServerDedupLayout rejected;
            assert(BuildDispatchServerDedupLayout(world,4,256,4,world*4,&rejected)!=0);
            CombineDedup::Layout rejectedCombine;
            assert(!CombineDedup::BuildLayout(world,4,256,4,world*4,rejectedCombine));
            continue;
        }
        EpMemoryServerDedupLayout layout;
        assert(BuildDispatchServerDedupLayout(world,4,256,4,world*4,&layout)==0);
        CombineDedup::Layout combined;
        assert(CombineDedup::BuildLayout(world,4,256,4,world*4,combined));
        // Exercise every source rank, including both sides of each host boundary.
        for(unsigned rank=0;rank<world;++rank) {
            Args args{rank,world};
            for(unsigned shift=0;shift<world;++shift) {
                std::vector<int32_t> ids;
                for(unsigned j=0;j<16;++j) ids.push_back(((shift+j)%world)*4+j%4);
                std::vector<uint32_t> states(world/G),masks(16);
                Route route{&args,0,16,4,{ids.data()},{states.data()},{masks.data()}};
                unsigned direct=0,groups=0;
                route.Classify(0,direct,groups);
                std::map<unsigned,unsigned> expected;
                unsigned expectedDirect=0;
                for(unsigned j=0;j<16;++j) {
                    unsigned host=unsigned(ids[j])/4/G;
                    if(host==rank/G) expectedDirect|=1U<<j;
                    else expected[host]|=1U<<j;
                }
                std::vector<unsigned> expectedGroups,actual(masks.begin(),masks.begin()+groups);
                for(auto [host,mask]:expected) {
                    if((mask&(mask-1))==0) expectedDirect|=mask;
                    else expectedGroups.push_back(mask);
                }
                std::sort(actual.begin(),actual.end());
                std::sort(expectedGroups.begin(),expectedGroups.end());
                assert(direct==expectedDirect && actual==expectedGroups);
                if(world==G) assert(direct==65535 && groups==0);
            }
            std::vector<unsigned> visited;
            for(unsigned i=0;i<world/G;++i)
                visited.push_back(EpServerOrder::ServerDedupHalfOrder(world/G,rank/G,i));
            std::sort(visited.begin(),visited.end());
            for(unsigned i=0;i<visited.size();++i) assert(visited[i]==i);
        }
        // Verify all allowed receiver counts, no missing/duplicate peer flags,
        // and clearing across two rounds with 58 total cores and asymmetric splits.
        for(unsigned send: {1U,20U,29U,40U,57U}) {
            const unsigned total=58,receivers=58-send,notify=std::min(8U,receivers);
            modelActiveRecv=receivers;
            std::vector<std::vector<float>> memory(world,std::vector<float>(8192,0));
            for(unsigned round=0;round<2;++round) {
                for(unsigned rank=0;rank<world;++rank) {
                    Args args{rank,world};
                    for(unsigned core=0;core<total;++core) {
                        Ready publisher{&args,core,&memory,total,send,total-notify,notify};
                        publisher.Publish();
                    }
                    for(unsigned slot=0;slot<notify;++slot) {
                        Ready ready{&args,total-notify+slot,&memory,total,send,total-notify,notify};
                        ready.NotifyServerCard();
                    }
                }
                for(unsigned rank=0;rank<world;++rank) {
                    Args args{rank,world};
                    for(unsigned receiver=0;receiver<receivers;++receiver) {
                        for(unsigned peer=0;peer<G;++peer)
                            assert(memory[rank][COMBINE_DEDUP_EP_SERVER_DEDUP_COMBINE_SERVER_READY_OFFSET/4+receiver*128+peer*8]==1);
                        Ready ready{&args,send+receiver,&memory,total,send,total-notify,notify};
                        ready.WaitServerReady();
                    }
                    for(unsigned index=0;index<memory[rank].size();++index) {
                        const unsigned readyBase=COMBINE_DEDUP_EP_SERVER_DEDUP_COMBINE_SERVER_READY_OFFSET/4;
                        const bool unused=index>=readyBase+receivers*128 && index<readyBase+kCombineNotifyRows*128;
                        const float expected=unused && (index-readyBase)%128<G*8 && (index-readyBase)%8==0 ? 1.0f:0.0f;
                        assert(memory[rank][index]==expected);
                    }
                }
            }
        }
    }
}
'''
    # Ready storage can reach 16 KiB + 48*512 bytes for alternate core splits.
    source = source.replace('offset<32768', 'offset<65536').replace('vector<float>(8192,0)', 'vector<float>(16384,0)')
    cpp = tmp_path/'protocol.cpp'
    cpp.write_text(source)
    binary = tmp_path/'protocol'
    subprocess.run(['g++','-std=c++17','-O2',f'-DDEEPEP_LL_RANKS_PER_GROUP={group}','-I',str(ROOT),str(cpp),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True,timeout=30)


@pytest.mark.parametrize("group", [0, 17, -1])
def test_invalid_group_size_rejected(tmp_path, group):
    if not shutil.which("g++"): pytest.skip("g++ required for compile-time rejection test")
    cpp = tmp_path / "invalid.cpp"
    cpp.write_text('#include "csrc/kernels/dispatch/low_latency/common/ep_memory_server_dedup_protocol.h"\n')
    result = subprocess.run(['g++', '-std=c++17', f'-DDEEPEP_LL_RANKS_PER_GROUP={group}',
                             '-I', str(ROOT), '-fsyntax-only', str(cpp)], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'LL logical group must be in [1, 16]' in result.stderr
