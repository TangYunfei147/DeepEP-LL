"""CPU checks for queried LL core selection, ownership and resource boundaries."""
from pathlib import Path
import subprocess
from tests.elastic.test_ll16_protocol import method

ROOT = Path(__file__).resolve().parents[2]


def compile_run(tmp_path, source, includes=()):
    cpp=tmp_path/'check.cpp';cpp.write_text(source)
    executable=tmp_path/'check'
    subprocess.run(['g++','-std=c++17','-O2',*[f'-I{p}' for p in includes],f'-I{ROOT}',
                    str(cpp),'-o',str(executable)],check=True)
    subprocess.run([str(executable)],check=True,timeout=30)


def test_actual_count_partitions_and_layouts(tmp_path):
    counts=(ROOT/'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_counts.h').read_text()
    compile_run(tmp_path,r'''
#include <cassert>
#include <vector>
#include <cstdint>
#include "csrc/kernels/dispatch/low_latency/common/ep_memory_server_dedup_layout.h"
#include "csrc/kernels/combine/low_latency/combine_layout.h"
using namespace DispatchDedup;
struct Split {
''' + method(counts,'SplitToCore') + r'''
};
int main() {
    Split split;
    for(uint32_t cores: {16U,32U,48U,56U,60U,64U,72U}) {
        assert(LlCoreCountValid(cores));
        const uint32_t half=cores/2;
        for(uint32_t total: {1U,16U,32U,63U,256U,1024U}) {
            // Both all-core counts and half-core route/receive work cover every item once.
            for(uint32_t participants: {cores,half}) {
                std::vector<unsigned> hits(total,0);
                unsigned min=total,max=0;
                for(uint32_t core=0;core<participants;++core) {
                    uint32_t begin,end,count;
                    split.SplitToCore(total,participants,core,begin,end,count);
                    assert(end-begin==count && end<=total);
                    min=count<min?count:min;max=count>max?count:max;
                    for(uint32_t i=begin;i<end;++i) ++hits[i];
                }
                assert(max-min<=1);
                for(auto n:hits) assert(n==1);
            }
        }
        for(uint32_t experts: {16U,32U,256U,1024U}) {
            auto count=DispatchServerDedupCountLayout(experts,cores);
            assert(count.cntOffset==kLlMaxCoreNum*512);
            assert(count.cumsumOffset==experts*512);
            assert(count.cumsumOffset+half*half*32<=count.prefixOffset);
            assert(count.prefixOffset+half*count.prefixStride<=count.cumsumFlagOffset);
            assert(count.cumsumFlagOffset+half*half*32+count.cntOffset<=count.stateBytes);
            assert(count.stateBytes<=COMBINE_DEDUP_EP_SERVER_DEDUP_COMBINE_CONTROL_OFFSET);
        }
        // The user reproduction must remain admissible at the actual core count.
        EpMemoryServerDedupLayout dispatch;
        assert(BuildDispatchServerDedupLayout(16,32,7168,6,256,&dispatch,false,cores)==0);
        assert(DispatchDedupPacketUbFits(dispatch,32,6,256,16,cores));
        CombineDedup::Layout combine;
        assert(CombineDedup::BuildLayout(16,32,7168,6,256,combine,cores));
        auto shared=CombineDedup::EpCombineM45Layout(3072,32,7168,6,half);
        assert(shared.inputLanes && shared.outputLanes);
        assert(shared.totalBytes<=CombineDedup::kEpServerDedupCombineRecvUbBytes);
        // Every partial row and source token has enough reserved per-core storage.
        assert(CombineDedup::EpCombineM4TaskBytes(3072,half)>=(3072+half-1)/half*4);
        assert(CombineDedup::EpCombineM5TokenCount(32,half)>=(32+half-1)/half);
    }
    for(uint32_t bad: {0U,1U,15U,55U,73U,80U}) {
        assert(!LlCoreCountValid(bad));
        EpMemoryServerDedupLayout d;CombineDedup::Layout c;
        assert(BuildDispatchServerDedupLayout(16,32,256,4,64,&d,false,bad)!=0);
        assert(!CombineDedup::BuildLayout(16,32,256,4,64,c,bad));
    }
}
''')


def test_platform_and_current_device_selection(tmp_path):
    acl=tmp_path/'acl';acl.mkdir()
    (acl/'acl.h').write_text('''#pragma once
#include <cstdint>
constexpr int ACL_SUCCESS=0,ACL_DEV_ATTR_VECTOR_CORE_NUM=201;
inline int current_result=0,info_result=0;
inline int64_t available=56;
inline int aclrtGetDevice(int32_t* device){*device=current_result?-1:0;return current_result;}
inline int aclrtGetDeviceInfo(uint32_t,int,int64_t* value){*value=available;return info_result;}
''')
    plat=tmp_path/'tiling/platform';plat.mkdir(parents=True)
    (plat/'platform_ascendc.h').write_text('''#pragma once
#include <cstdint>
namespace platform_ascendc {
inline uint32_t reported=64;inline bool missing=false;
struct PlatformAscendCManager {
 static PlatformAscendCManager* GetInstance(){static PlatformAscendCManager p;return missing?nullptr:&p;}
 uint32_t GetCoreNumAiv(){return reported;}
};}
''')
    compile_run(tmp_path,r'''
#include <cassert>
#include "csrc/runtime/ll_core_config.hpp"
int main() {
    auto cfg=deepep::query_ll_core_config();
    assert(cfg.platform==64 && cfg.available==56 && cfg.active==56 && cfg.device==0);
    platform_ascendc::reported=56;available=60;
    assert(deepep::query_ll_core_config().active==56);
    current_result=1;
    assert(deepep::query_ll_core_config().device==-1);
    assert(deepep::query_ll_core_config().active==56);
    current_result=0;available=55;
    bool threw=false;try { deepep::query_ll_core_config(); } catch(...) {threw=true;} assert(threw);
    available=56;info_result=1;threw=false;
    try { deepep::query_ll_core_config(); } catch(...) {threw=true;} assert(threw);
    info_result=0;platform_ascendc::missing=true;threw=false;
    try { deepep::query_ll_core_config(); } catch(...) {threw=true;} assert(threw);
}
''', [tmp_path])
