"""Dispatch compile partition and profiler identity regression tests; CPU only."""
import csv
import importlib.util
import re
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
MODES=('prequant_e4m3','fp16','fp16_e4m3','fp16_e5m2','bf16','bf16_e4m3','bf16_e5m2')

def test_device_modes_cannot_collapse_back_into_one_translation_unit():
    base=ROOT/'csrc/kernels/dispatch/low_latency'
    # Ignore comments: these checks guard build structure, not explanatory text.
    def code(path):
        return re.sub(r'/\*.*?\*/|//[^\n]*','',path.read_text(),flags=re.S)
    bridge=code(base.parent/'low_latency.cpp')
    assert '__global__' not in bridge
    assert 'DispatchMemoryServerDedupKernel<' not in bridge
    assert 'dispatch_server_dedup_kernel.h' not in bridge
    assert 'dispatch_variant_kernel.h' not in bridge
    for number,mode in enumerate(MODES):
        unit=code(base/'variants'/f'{mode}.cpp')
        assert re.findall(r'#define\s+DEEPEP_LL_VARIANT_MODE\s+(\d+)',unit)==[str(number)]
        assert re.findall(r'#define\s+DEEPEP_LL_VARIANT_NAME\s+(\w+)',unit)==[mode]
        assert re.findall(r'#include\s+"([^"]+)"',unit)==['../dispatch_variant_kernel.h']
        assert '__global__' not in unit and 'DispatchMemoryServerDedupKernel<' not in unit
    cmake=(ROOT/'cmake/device/CMakeLists.txt').read_text()
    # An explicit target property overrides a caller's global unity-build setting.
    assert re.search(r'set_target_properties\(deepep_kernels\s+PROPERTIES\s+UNITY_BUILD\s+OFF\b',cmake)

def test_variant_inventory_and_timer_lifetimes():
    base=ROOT/'csrc/kernels/dispatch/low_latency'
    assert {p.stem for p in (base/'variants').glob('*.cpp')}==set(MODES)
    cmake=(ROOT/'cmake/device/CMakeLists.txt').read_text()
    inventory=(ROOT/'cmake/OperatorSources.cmake').read_text()
    assert 'OperatorSources.cmake' in cmake and '${DEEPEP_CORE_DEVICE_SOURCES}' in cmake
    for mode in MODES: assert f'variants/{mode}.cpp' in inventory
    entry=(base/'dispatch_variant_kernel.h').read_text()
    ordered=['kernel.Init(','timer.Mark<LlStageTimer::INIT_DONE>()',
             'kernel.Process();','timer.Mark<LlStageTimer::WORK_DONE>()','timer.Flush()']
    positions=[entry.index(x) for x in ordered]
    assert positions==sorted(positions)
    assert entry.count('DispatchMemoryServerDedupKernel<')==1
    assert 'LlDispatchVariantForTiling' in entry
    assert 'SyncAll' not in entry
    host=(ROOT/'csrc/ops/low_latency.cpp').read_text()
    assert host.count('dispatch_format_tiling(dtype)')==2
    assert 'dispatch_server_dedup_kernel_do_sc18(' in host

@pytest.mark.parametrize('timer',[False,True])
def test_real_host_selector_all_modes_and_rejections(tmp_path,timer):
    path=Path(__file__).with_name('test_sparse_control_native.py')
    spec=importlib.util.spec_from_file_location('sparse_selector_native',path)
    helper=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.run_sparse_host_selector_mock(tmp_path,timer)

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
