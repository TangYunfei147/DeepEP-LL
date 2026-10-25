"""A5 device0 allocation/schema check; does not create any LL communication domain."""
import os
from pathlib import Path
import runpy
import json
for name in ('CUDA_VISIBLE_DEVICES','ASCEND_RT_VISIBLE_DEVICES','ASCEND_VISIBLE_DEVICES','PROBE_VISIBLE_DEVICES'):
    if os.environ.get(name) != '0,1':
        raise RuntimeError('A5-only schema validation requires all device visibility variables fixed to 0,1')
ROOT=Path(__file__).resolve().parents[2]
helpers=runpy.run_path(str(ROOT/'tests/elastic/test_dispatch.py'))
package=helpers['_import_installed_deep_ep']()
import torch
import torch_npu
from deep_ep._native import require_native
from deep_ep.utils.ll_stage_timer import StageTrace, SLOTS, FIELDS
torch.npu.set_device(0)
native=require_native('LL timer schema validation')
st=runpy.run_path(str(ROOT/'tests/elastic/test_ep_low_latency.py'))
selection=st['installed_dispatch_identity'](package,native)['dispatch_impl']
registry=runpy.run_path(str(ROOT/'scripts/ll_dispatch_implementation.py'))
cfg=native.low_latency_core_config()
prefix=st['installed_prefix_identity'](package,native,cfg['recv_cores'])['small_b_prefix_cores']
identity=registry['dispatch_configuration'](selection,cfg['send_cores'],cfg['recv_cores'],prefix)
assert native.low_latency_implementation==identity['implementation']
assert native.low_latency_overlap_stage==identity['overlap_stage']
assert registry['dispatch_group_valid'](selection,native.low_latency_ranks_per_group)
assert native.low_latency_stage_timer_enabled
assert native.low_latency_core_protocol==identity['core_protocol']
runpy.run_path(str(ROOT/'tests/elastic/test_ep_low_latency.py'))['validate_sparse_timer_layout'](
    native.low_latency_stage_timer_layout())
trace=StageTrace(object(),'dispatch')
cfg=native.low_latency_core_config()
assert tuple(trace.tensor.shape)==(cfg['active_aiv'],SLOTS,FIELDS)
assert trace.tensor.data_ptr()%512==0
assert trace.tensor.device.index==0 and torch.count_nonzero(trace.tensor).item()==0
print(f"PASS installed timer ABI/schema and actual device0 allocation ({cfg['active_aiv']} AIV); no LL launch")
print('Sparse milestones: 512 bytes per core; legacy expert capture is disabled')
