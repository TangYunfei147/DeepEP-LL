"""A5 device0 allocation/schema check; does not create any LL communication domain."""
import os
from pathlib import Path
import runpy
for name in ('CUDA_VISIBLE_DEVICES','ASCEND_RT_VISIBLE_DEVICES','ASCEND_VISIBLE_DEVICES','PROBE_VISIBLE_DEVICES'):
    if os.environ.get(name) != '0,1':
        raise RuntimeError('A5-only schema validation requires all device visibility variables fixed to 0,1')
ROOT=Path(__file__).resolve().parents[2]
helpers=runpy.run_path(str(ROOT/'tests/elastic/test_dispatch.py'))
helpers['_import_installed_deep_ep']()
import torch
import torch_npu
from deep_ep._native import require_native
from deep_ep.utils.ll_stage_timer import StageTrace, STAGES, SLOTS, FIELDS, SCHEMA
torch.npu.set_device(0)
native=require_native('LL timer schema validation')
assert native.low_latency_implementation=='g4-r0-prefix'
assert native.low_latency_overlap_stage==3
cfg=native.low_latency_core_config()
assert native.low_latency_stage_timer_enabled
assert native.low_latency_core_protocol==19000+cfg["send_cores"]
layout=native.low_latency_stage_timer_layout()
assert layout['schema']==SCHEMA and layout['slots']==SLOTS and layout['fields']==FIELDS
assert layout['stages']==list(STAGES) and layout['recording']=='systemcycle-stages'
assert layout['timer_profile']=='systemcycle' and layout['timer_profile_id']==9
trace=StageTrace(object(),'dispatch')
cfg=native.low_latency_core_config()
assert tuple(trace.tensor.shape)==(cfg['active_aiv'],SLOTS,FIELDS)
assert trace.tensor.data_ptr()%512==0
assert trace.tensor.device.index==0 and torch.count_nonzero(trace.tensor).item()==0
print(f"PASS installed timer ABI/schema and actual device0 allocation ({cfg['active_aiv']} AIV); no LL launch")
print('SystemCycle capture: legacy expert capture is disabled')
