"""CPU checks for overlap capture and layout/progress invariants."""
import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
fixtures=load("overlap_segments","tests/ll_stage_timer/test_segments.py")
timer=load("overlap_timer","deep_ep/utils/ll_stage_timer.py")
def test_schema11_and_legacy():
    for schema in (7,9,10,11):
        assert timer.parse_records(fixtures.raw(schema=schema),"dispatch")["schema"]==schema
    assert timer.SCHEMA==11
    assert timer.STAGES[70:]==("SEND_COMPLETE_WAIT","TRANSITION_DRAIN","RECV_PREPARE","SEND_SUBMIT_END","OVERLAP_PATH","SERIAL_FALLBACK","SEND_LANES","RECV_LANES")
    assert (8+2*timer.RECORDS)*8<=timer.SLOTS*timer.FIELDS*8


def test_native_layout_matrix(tmp_path):
    import shutil, subprocess, pytest
    cc=shutil.which("g++") or shutil.which("clang++")
    if not cc: pytest.skip("C++ compiler unavailable; run native layout matrix on target")
    source=tmp_path/"layout.cpp"
    source.write_text(r'''#include "csrc/kernels/dispatch/low_latency/common/ll_overlap_layout.h"
#include <cassert>
int main() {
 using namespace DispatchDedup;
 unsigned enabled=0, fallback=0;
 for (auto mode:{EpServerDedupInputMode::Plain16,EpServerDedupInputMode::Quantize16,EpServerDedupInputMode::PrequantizedFp8Packs})
 for (unsigned p:{16U,32U,64U,128U}) for(unsigned bs:{1U,8U,256U,1024U})
 for(unsigned h:{128U,3584U,7168U}) for(unsigned k:{1U,8U}) {
   unsigned e=p*6;
   auto data=DispatchServerDedupDataLayout(bs,h,k,e,mode);
   for(unsigned core=0;core<56;++core) {
    auto l=LlMakeOverlapLayout(data,bs,k,e,p,0,core,mode);
    if(!l.enabled) {++fallback;continue;}
    ++enabled;assert(l.totalBytes<=190*1024);
    assert(l.sendLanes>=1 && l.sendLanes<=6 && l.recvLanes>=1 && l.recvLanes<=6);
    assert(l.sendBytes>=l.sendLanes*data.tokenStride);
    assert((core<kLlSendCoreNum)==(l.countBytes==0));
   }
 }
 assert(enabled && fallback);
}''',encoding="utf-8")
    exe=tmp_path/"layout"
    subprocess.run([cc,"-std=c++17","-include","initializer_list","-I",str(ROOT),str(source),"-o",str(exe)],check=True)
    subprocess.run([str(exe)],check=True)
