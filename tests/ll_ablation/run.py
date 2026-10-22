"""Run one full-protocol ablation on the user's target, with explicit device list."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import json
from common import ROOT,MODES,idle,sha

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=MODES,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--devices',required=True)
    p.add_argument('--run-name',required=True)
    p.add_argument('--config',type=Path,default=ROOT/'tests/ll_ablation/b256.json')
    p.add_argument('--plan',action='store_true')
    a,extra=p.parse_known_args()
    manifest=a.manifest if a.manifest.is_absolute() else ROOT/a.manifest
    info=json.loads(manifest.read_text())
    if info['mode']!=a.mode:raise RuntimeError('Manifest mode mismatch')
    wheel=manifest.parent/info['wheel']
    if sha(wheel)!=info['wheel_sha256']:raise RuntimeError('Wheel digest mismatch')
    config=a.config if a.config.is_absolute() else ROOT/a.config
    settings=json.loads(config.read_text())
    if settings.get('dtype')!='fp8' or settings.get('fp8_scale_dtype')!='int32':
        raise RuntimeError('This suite requires FP8/int32 scales')
    if extra:raise RuntimeError('Use config JSON for overrides; arbitrary CLI overrides are rejected')
    if info['send_cores']+info['recv_cores']!=56:raise RuntimeError('Invalid core split')
    env=dict(os.environ,LL_ABLATION_MODE=str(MODES[a.mode]),LL_ABLATION_MANIFEST=str(manifest))
    command=[sys.executable,str(ROOT/'tests/ll_stage_timer/run.py'),'--devices',a.devices,
             '--config',str(config),'--run-name',a.run_name,'--dispatch-only']
    if a.mode=='ready-replay':
        if info['timer']!='ON':raise RuntimeError('Replay requires ON wheel')
        command+=['--ll-stage-timer','--ll-timer-iters','5','--skip-perf-test']
    elif info['timer']=='ON':command+=['--ll-stage-timer','--ll-timer-iters','5']
    if a.plan:print(json.dumps(dict(command=command,manifest=info),indent=2));return
    # npu-smi before launcher imports/initializes any NPU runtime.
    idle()
    from importlib.util import spec_from_file_location,module_from_spec
    spec=spec_from_file_location('ll_st_launcher',ROOT/'tests/ll_stage_timer/run.py')
    launcher=module_from_spec(spec);spec.loader.exec_module(launcher)
    env.update(launcher.environment(a.devices))
    env.update(LL_ABLATION_MODE=str(MODES[a.mode]),LL_ABLATION_MANIFEST=str(manifest))
    sdk=ROOT/info['source']/'third-party/shmem/install/shmem/lib'
    if not (sdk/'libshmem.so').is_file():raise RuntimeError('Matching isolated SDK missing')
    env['LD_LIBRARY_PATH']=str(sdk)+':'+env.get('LD_LIBRARY_PATH','')
    # Verify installed wheel's ABI and timer configuration without starting a device.
    code='from deep_ep._native import require_native; n=require_native("ablation"); '
    code+=f'assert n.low_latency_ablation_mode=={MODES[a.mode]}; assert n.low_latency_core_protocol=={29000+100*MODES[a.mode]+info["send_cores"]}; '
    code+=f'assert bool(n.low_latency_stage_timer_enabled)=={info["timer"]=="ON"}'
    subprocess.run([sys.executable,'-c',code],cwd=manifest.parent,env=env,check=True)
    subprocess.run(command,cwd=ROOT,env=env,check=True)

if __name__=='__main__':main()
