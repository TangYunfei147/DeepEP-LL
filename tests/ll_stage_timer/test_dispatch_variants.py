"""Dispatch compile partition and profiler identity regression tests; CPU only."""
import csv
import importlib.util
import shutil
import subprocess
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
MODES=('prequant_e4m3','fp16','fp16_e4m3','fp16_e5m2','bf16','bf16_e4m3','bf16_e5m2')

def test_variant_inventory_and_timer_lifetimes():
    base=ROOT/'csrc/kernels/dispatch/low_latency'
    assert {p.stem for p in (base/'variants').glob('*.cpp')}==set(MODES)
    cmake=(ROOT/'cmake/device/CMakeLists.txt').read_text()
    for mode in MODES: assert f'variants/{mode}.cpp' in cmake
    entry=(base/'dispatch_variant_kernel.h').read_text()
    ordered=['kernel.Init(','timer.End(LlStageTimer::INIT','timer.Begin(LlStageTimer::KERNEL',
             'kernel.Process();','timer.End(LlStageTimer::KERNEL','AscendC::SyncAll<true>()','timer.Flush()']
    positions=[entry.index(x) for x in ordered]
    assert positions==sorted(positions)
    assert entry.count('DispatchMemoryServerDedupKernel<')==1
    assert 'LlDispatchVariantForTiling' in entry
    host=(ROOT/'csrc/ops/low_latency.cpp').read_text()
    assert host.count('dispatch_format_tiling(dtype)')==2
    assert 'dispatch_server_dedup_kernel_do_v2(' in host

@pytest.mark.parametrize('timer',[False,True])
def test_real_host_selector_all_modes_and_rejections(tmp_path,timer):
    cc=shutil.which('g++') or shutil.which('clang++')
    if not cc:pytest.skip('C++ compiler required; also run on A3')
    code=r'''#include <cassert>
#include <stdexcept>
#include <initializer_list>
#include "csrc/kernels/dispatch/low_latency/common/ll_dispatch_variant.h"
#include "csrc/kernels/dispatch/low_latency/launch.hpp"
using namespace DispatchDedup;
static unsigned calls=0,selected=99;
#define STUB(name,n) extern "C" void dispatch_server_dedup_launch_##name(void*, const deepep::EpServerDedupKernelArgs*) {++calls;selected=n;}
STUB(prequant_e4m3,0) STUB(fp16,1) STUB(fp16_e4m3,2) STUB(fp16_e5m2,3)
STUB(bf16,4) STUB(bf16_e4m3,5) STUB(bf16_e5m2,6)
int main() {
 const unsigned table[7][4]={{15,4,15,2},{3,0,3,0},{3,4,15,1},{3,4,16,1},{11,0,11,0},{11,4,15,1},{11,4,16,1}};
 for(unsigned dtype=0;dtype<256;++dtype) for(unsigned quant:{0U,1U,4U,255U})
 for(unsigned output:{3U,11U,15U,16U,255U}) for(unsigned input=0;input<4;++input) {
   unsigned expected=0xffffffffU;
   for(unsigned m=0;m<7;++m) if(dtype==table[m][0] && quant==table[m][1] && input==table[m][3] &&
       (output==table[m][2] || (output==255 && (m==1 || m==4)))) expected=m;
   assert(LlDispatchVariantForTiling(dtype,quant,output,input)==expected);
 }
 deepep::EpServerDedupKernelArgs args{};
 static_assert(sizeof(args)==152);
 for(unsigned m=0;m<7;++m) { calls=0;dispatch_server_dedup_kernel_do_v2(nullptr,&args,m);assert(calls==1 && selected==m); }
 for(unsigned m:{7U,255U,0xffffffffU}) {
   calls=0;bool threw=false;try {dispatch_server_dedup_kernel_do_v2(nullptr,&args,m);}catch(const std::invalid_argument&){threw=true;}
   assert(threw && !calls);
 }
 bool threw=false;try {dispatch_server_dedup_kernel_do_v2(nullptr,nullptr,0);}catch(const std::invalid_argument&){threw=true;}assert(threw);
 assert(deepep_ll_core_protocol_version()==14028);
 assert(deepep_ll_ranks_per_group()==4);
#ifdef DEEPEP_LL_STAGE_TIMER
 assert(deepep_ll_stage_timer_enabled()==1);
#else
 assert(deepep_ll_stage_timer_enabled()==0);
#endif
}'''
    source=tmp_path/'selector.cpp';source.write_text(code)
    exe=tmp_path/'selector'
    flags=['-DDEEPEP_LL_STAGE_TIMER=1'] if timer else []
    subprocess.run([cc,'-std=c++17','-DCATLASS_ARCH=3510',*flags,'-I',str(ROOT),str(source),str(ROOT/'csrc/kernels/dispatch/low_latency.cpp'),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)

@pytest.mark.parametrize('mode',MODES)
def test_profiler_matches_exact_variant_and_rejects_mixed(tmp_path,mode):
    spec=importlib.util.spec_from_file_location('variants_st',ROOT/'tests/elastic/test_ep_low_latency.py')
    st=importlib.util.module_from_spec(spec);spec.loader.exec_module(st)
    path=tmp_path/'kernel_details.csv'
    def write(names):
        with path.open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['Name','Duration(us)','Start Time(us)'])
            for i,name in enumerate(names):writer.writerow([name,2+i,10+i])
    prefix='dispatch_server_dedup_kernel'
    write([prefix+'_'+mode]*3)
    assert st.kernel_durations(tmp_path,prefix,3)==[2,3,4]
    write([prefix+'_'+mode,prefix])
    with pytest.raises(ValueError,match='Mixed dispatch'):st.kernel_durations(tmp_path,prefix,2)
    write([prefix+'_'+mode+'_unrelated'])
    with pytest.raises(ValueError,match='found 0'):st.kernel_durations(tmp_path,prefix,1)
