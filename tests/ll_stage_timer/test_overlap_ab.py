"""A/B identity, policies and metric boundaries; no NPU usage."""
import importlib.util
from pathlib import Path
import shutil,subprocess,json
import pytest
ROOT=Path(__file__).resolve().parents[2]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
st=load('ab_entry','tests/elastic/test_ep_low_latency.py')

def test_stale_wheel_stage_rejected_even_without_timer():
    stage=st.expected_overlap_stage()
    st.validate_overlap_stage_capability(dict(implementation=json.loads((ROOT/'tests/ll_stage_timer/branch.json').read_text())['implementation'],overlap_stage=stage))
    for wrong in (0,1,2,3,4,5,None):
        if wrong==stage:continue
        with pytest.raises(RuntimeError,match='Rebuild/install'):
            st.validate_overlap_stage_capability(dict(implementation=json.loads((ROOT/'tests/ll_stage_timer/branch.json').read_text())['implementation'],overlap_stage=wrong))
    with pytest.raises(RuntimeError):
        st.validate_overlap_stage_capability(dict(implementation='g4-overlap-recv',overlap_stage=stage))

def test_common_timing_boundaries():
    base=ROOT/'csrc/kernels/dispatch/low_latency'
    sender=(base/'device/dispatch_server_dedup_send.h').read_text(encoding='utf-8')
    counts=(base/'device/dispatch_server_dedup_counts.h').read_text(encoding='utf-8')
    assert 'profile_->Mark<12>' in counts and 'profile_->Mark<13>' in counts
    assert 'profile_->Mark<14>' in counts and 'profile_->Mark<15>' in counts
    assert 'profile_->Mark<38>' in sender and 'profile_->Mark<39>' in sender
    assert 'overlap_.Cooperative() ? 8U:tasks' in sender
    assert 'if (overlap_.Async())' in sender and 'if (!overlap_.Async())' in sender


def test_real_cpp_stage_policy(tmp_path):
    cc=shutil.which('g++') or shutil.which('clang++')
    if not cc:pytest.skip('C++ compiler required on validation host')
    source=tmp_path/'policy.cpp'
    source.write_text(r'''#include <cassert>
#include "csrc/kernels/dispatch/low_latency/common/ll_overlap_layout.h"
int main(){using namespace DispatchDedup; LlOverlapLayout p{};
 assert(!p.Async() && !p.Cooperative());p.enabled=true;
 assert(p.Async()==(kLlOverlapExperimentStage>=3));
 assert(p.Cooperative()==(kLlOverlapExperimentStage>=4));
 auto data=DispatchServerDedupDataLayout(256,3584,8,384,EpServerDedupInputMode::Quantize16);
 auto plan=LlMakeOverlapLayout(data,256,8,384,16,0,0,EpServerDedupInputMode::Quantize16);
 assert(plan.enabled==(kLlOverlapExperimentStage>=2));
}''')
    exe=tmp_path/'policy'
    subprocess.run([cc,'-std=c++17','-I',str(ROOT),str(source),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)

def test_stage_is_recorded_in_reference_identity():
    device=(ROOT/'csrc/kernels/dispatch/low_latency/common/ll_reference_profile_device.h').read_text(encoding='utf-8')
    assert 'record_[48] = DispatchDedup::kLlCb59Replay ? 0U : DispatchDedup::kLlOverlapExperimentStage' in device
