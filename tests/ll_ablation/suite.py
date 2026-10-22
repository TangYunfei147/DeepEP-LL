"""Compile all variants first, then install all wheels and run experiments serially."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from common import ROOT,MODES as ALL_MODES,source_checkout,sha,git
# Preserve the original five-wheel experiment; receiver_suite.py builds one new wheel.
MODES={name:mode for name,mode in ALL_MODES.items() if mode<=4}


def plan_commands(a):
    config=a.config if a.config.is_absolute() else ROOT/a.config
    settings=json.loads(config.read_text())
    if settings.get('num_nodes',1)!=1:raise ValueError('Use individual run.py per host for multi-host jobs')
    existing={}
    for mode in MODES:
        timer='ON' if mode=='ready-replay' else 'OFF'
        manifest=ROOT/'.ablation/wheels'/f'{mode}-{timer}-s{a.send_cores}'/'manifest.json'
        if not a.build and not manifest.is_file():
            raise ValueError(f'Missing wheel manifest: {manifest}; run --build-only --resume first')
        if (a.resume or not a.build) and manifest.exists():
            info=json.loads(manifest.read_text())
            if (info['mode'],info['timer'],info['send_cores'],info['recv_cores'])!=(mode,timer,a.send_cores,56-a.send_cores):
                raise ValueError(f'Existing manifest identity mismatch: {manifest}')
            wheel=manifest.parent/info['wheel']
            if not wheel.is_file() or sha(wheel)!=info['wheel_sha256']:
                raise ValueError(f'Existing wheel missing/corrupt: {wheel}')
            existing[mode]=info
    revisions={info['commit'] for info in existing.values()}
    if len(revisions)>1:raise ValueError('Existing wheels have different source commits; cannot resume a matched suite')
    revision=next(iter(revisions)) if revisions else git('rev-parse','HEAD')
    build_commands=[];install_commands=[];run_commands=[]
    for mode in MODES:
        timer='ON' if mode=='ready-replay' else 'OFF'
        tag=f'{mode}-{timer}-s{a.send_cores}'
        manifest=ROOT/'.ablation/wheels'/tag/'manifest.json'
        env=ROOT/'.ablation/envs'/tag
        python=env/('Scripts/python.exe' if sys.platform=='win32' else 'bin/python')
        if a.build and mode not in existing:
            command=[sys.executable,str(ROOT/'tests/ll_ablation/build.py'),'--mode',mode,
                '--timer',timer,'--send-cores',str(a.send_cores),'--prepare-sdk','--revision',revision]
            if a.resume:command+=['--resume']
            if a.offline_deps:command+=['--offline-deps',str(a.offline_deps.resolve())]
            build_commands.append(command)
        if not env.exists():install_commands.append([sys.executable,'-m','venv','--system-site-packages',str(env)])
        install_commands.append([str(python),str(ROOT/'tests/ll_ablation/install.py'),str(manifest)])
        run_commands.append([str(python),str(ROOT/'tests/ll_ablation/run.py'),'--mode',mode,'--manifest',str(manifest),
            '--devices',a.devices,'--run-name',a.run_prefix+'-'+mode,'--config',str(config)])
    result={'build':build_commands,'install':[] if a.build_only else install_commands,
            'run':[] if a.build_only else run_commands}
    return dict(source_revision=revision,reused_modes=list(existing),phases=result)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--devices','--device',dest='devices')
    p.add_argument('--send-cores',type=int,default=28)
    p.add_argument('--run-prefix')
    p.add_argument('--config',type=Path,default=ROOT/'tests/ll_ablation/b256.json')
    p.add_argument('--build',action='store_true',help='Build all variants before any installation or device run')
    p.add_argument('--build-only',action='store_true',help='Compile all wheels; no installation, occupancy check or NPU use')
    p.add_argument('--resume',action='store_true',help='Reuse validated wheels and pin missing builds to their source revision')
    p.add_argument('--offline-deps',type=Path)
    p.add_argument('--plan',action='store_true')
    a=p.parse_args()
    if a.build_only:a.build=True
    if not a.build_only:
        ids=(a.devices or '').split(',')
        if len(ids)!=16 or any(not i.isdecimal() for i in ids) or len(set(ids))!=16:p.error('Explicit 16 distinct device IDs required')
        if not a.run_prefix or Path(a.run_prefix).name!=a.run_prefix or a.run_prefix in ('.','..'):p.error('Invalid run prefix')
    else:
        a.devices=a.devices or '';a.run_prefix=a.run_prefix or 'build-only'
    if not 0<a.send_cores<56:p.error('send-cores must be 1..55')
    try:plan=plan_commands(a)
    except (ValueError,KeyError,OSError) as error:p.exit(2,str(error)+'\n')
    if a.plan:print(json.dumps(plan,indent=2));return
    if plan['phases']['build']:
        try:source_checkout()
        except RuntimeError as error:p.exit(2,str(error)+'\n')
    print(f"Source revision: {plan['source_revision']}; reused wheels: {', '.join(plan['reused_modes']) or 'none'}",flush=True)
    for phase,commands in plan['phases'].items():
        if commands:print(f'Phase {phase}: {len(commands)} commands',flush=True)
        for command in commands:subprocess.run(command,cwd=ROOT,check=True)

if __name__=='__main__':main()
