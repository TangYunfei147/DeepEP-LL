"""Native Host bounds for shared descriptors; no accelerator operations."""
from pathlib import Path
import shutil
import subprocess
import pytest
ROOT=Path(__file__).resolve().parents[2]

@pytest.mark.parametrize('send',[1,10,28,55])
def test_native_shared_control_and_ub_bounds(tmp_path,send):
    cc=shutil.which('g++') or shutil.which('clang++')
    if not cc:pytest.skip('C++ compiler required for Host control/UB bounds')
    source=tmp_path/'bounds.cpp'
    source.write_text(r'''
#include <cassert>
#include <initializer_list>
#include "csrc/kernels/dispatch/low_latency/common/ep_memory_server_dedup_layout.h"
#include "csrc/kernels/dispatch/low_latency/common/ll_overlap_layout.h"
using namespace DispatchDedup;
int main() {
 unsigned accepted=0,rejected=0;
 for(unsigned world:{16U,32U,64U,128U})
 for(unsigned b:{1U,8U,32U,256U,1024U,4096U})
 for(unsigned k:{1U,8U,32U}) for(unsigned e:{384U,1024U}) {
   EpMemoryServerDedupLayout layout{};
   auto mode=EpServerDedupInputMode::PrequantizedFp8Packs;
   if(BuildDispatchServerDedupLayout(world,b,3584,k,e,&layout,mode)) {++rejected;continue;}
   ++accepted;
   auto plan=LlMakeSharedGroupPlan(b,k,world/kLlRanksPerGroup,kLlSendCoreNum);
   assert(layout.countStateBytes+plan.gmBytes<=COMBINE_DEDUP_EP_SERVER_DEDUP_COMBINE_CONTROL_OFFSET);
   assert(plan.descriptorBytes%32==0 && plan.headerBytes%32==0);
   assert(plan.tableBytes+plan.descriptorBytes*kLlSendCoreNum==plan.gmBytes);
   assert(plan.cacheBytes>=((b*k+kLlSendCoreNum-1)/kLlSendCoreNum)*32);
   assert(plan.ubBytes==2*plan.descriptorBytes+plan.tableBytes+plan.readyBytes+plan.headerBytes+plan.cacheBytes);
   for(unsigned core=0;core<56;++core) {
     auto p=LlMakeOverlapLayout(layout.data,b,k,e,world,0,core,mode);
     if(p.enabled)assert(p.totalBytes<=kEpServerDedupCountUbBytes);
   }
 }
 assert(accepted && rejected);
 if(kLlSendCoreNum==28) {
   auto d=DispatchServerDedupDataLayout(256,3584,8,384,EpServerDedupInputMode::PrequantizedFp8Packs);
   auto p=LlMakeOverlapLayout(d,256,8,384,16,0,0,EpServerDedupInputMode::PrequantizedFp8Packs);
   assert(p.enabled && p.sendLanes==6 && p.recvLanes==6 && p.totalBytes==121536);
   auto gm=LlMakeSharedGroupPlan(256,8,4,28);
   assert(gm.gmBytes==19712 && gm.ubBytes==6400);
 }
}''',encoding='utf-8')
    exe=tmp_path/('bounds.exe' if __import__('os').name=='nt' else 'bounds')
    subprocess.run([cc,'-std=c++17',f'-DDEEPEP_LL_SEND_CORES={send}',f'-DDEEPEP_LL_RECV_CORES={56-send}',
                    '-iquote',str(ROOT),str(source),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)
