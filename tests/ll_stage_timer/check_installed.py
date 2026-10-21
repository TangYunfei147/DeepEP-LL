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
cfg=native.low_latency_core_config()
assert native.low_latency_stage_timer_enabled
assert native.low_latency_prefix_mode in ('parallel','r0')
assert native.low_latency_core_protocol==(17000 if native.low_latency_prefix_mode=='r0' else 16000)+cfg["send_cores"]
assert native.low_latency_stage_timer_layout()==dict(schema=SCHEMA,slots=SLOTS,fields=FIELDS,stages=list(STAGES))
trace=StageTrace(object(),'dispatch')
cfg=native.low_latency_core_config()
assert tuple(trace.tensor.shape)==(cfg['active_aiv'],SLOTS,FIELDS)
assert trace.tensor.data_ptr()%512==0
assert trace.tensor.device.index==0 and torch.count_nonzero(trace.tensor).item()==0
print(f"PASS installed timer ABI/schema and actual device0 allocation ({cfg['active_aiv']} AIV); no LL launch")
print('Reference segment mode: legacy expert capture is disabled')
