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
helpers['_import_installed_deep_ep']()
import torch
import torch_npu
from deep_ep._native import require_native
from deep_ep.utils.ll_stage_timer import StageTrace, SLOTS, FIELDS
torch.npu.set_device(0)
native=require_native('LL timer schema validation')
assert native.low_latency_implementation==json.loads((ROOT/'tests/ll_stage_timer/branch.json').read_text())['implementation']
assert native.low_latency_overlap_stage==3
cfg=native.low_latency_core_config()
assert native.low_latency_stage_timer_enabled
assert native.low_latency_core_protocol==19000+cfg["send_cores"]
runpy.run_path(str(ROOT/'tests/elastic/test_ep_low_latency.py'))['validate_sparse_timer_layout'](
    native.low_latency_stage_timer_layout())
trace=StageTrace(object(),'dispatch')
cfg=native.low_latency_core_config()
assert tuple(trace.tensor.shape)==(cfg['active_aiv'],SLOTS,FIELDS)
assert trace.tensor.data_ptr()%512==0
assert trace.tensor.device.index==0 and torch.count_nonzero(trace.tensor).item()==0
print(f"PASS installed timer ABI/schema and actual device0 allocation ({cfg['active_aiv']} AIV); no LL launch")
print('Sparse milestones: 512 bytes per core; legacy expert capture is disabled')
