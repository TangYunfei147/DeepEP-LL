"""Build/install/run the single-host EP16 suite serially. --plan performs no work."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from common import ROOT,MODES

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--devices',required=True)
    p.add_argument('--send-cores',type=int,default=28)
    p.add_argument('--run-prefix',required=True)
    p.add_argument('--config',type=Path,default=ROOT/'tests/ll_ablation/b256.json')
    p.add_argument('--build',action='store_true',help='Build new isolated wheels, including project-local SDKs')
    p.add_argument('--offline-deps',type=Path)
    p.add_argument('--plan',action='store_true')
    a=p.parse_args()
    ids=a.devices.split(',')
    if len(ids)!=16 or any(not i.isdecimal() for i in ids) or len(set(ids))!=16:p.error('Explicit 16 distinct device IDs required')
    if not 0<a.send_cores<56:p.error('send-cores must be 1..55')
    if Path(a.run_prefix).name!=a.run_prefix or a.run_prefix in ('.','..'):p.error('Invalid run prefix')
    config=a.config if a.config.is_absolute() else ROOT/a.config
    settings=json.loads(config.read_text())
    if settings.get('num_nodes',1)!=1:p.error('Use individual run.py per host for multi-host jobs')
    commands=[]
    for mode in MODES:
        timer='ON' if mode=='ready-replay' else 'OFF'
        tag=f'{mode}-{timer}-s{a.send_cores}'
        manifest=ROOT/'.ablation/wheels'/tag/'manifest.json'
        env=ROOT/'.ablation/envs'/tag
        python=env/('Scripts/python.exe' if sys.platform=='win32' else 'bin/python')
        if a.build:commands.append([sys.executable,str(ROOT/'tests/ll_ablation/build.py'),'--mode',mode,
            '--timer',timer,'--send-cores',str(a.send_cores),'--prepare-sdk'])
        if a.build and a.offline_deps:commands[-1]+=['--offline-deps',str(a.offline_deps.resolve())]
        if not env.exists():commands.append([sys.executable,'-m','venv','--system-site-packages',str(env)])
        commands.append([str(python),str(ROOT/'tests/ll_ablation/install.py'),str(manifest)])
        commands.append([str(python),str(ROOT/'tests/ll_ablation/run.py'),'--mode',mode,'--manifest',str(manifest),
            '--devices',a.devices,'--run-name',a.run_prefix+'-'+mode,'--config',str(config)])
    if a.plan:print(json.dumps(commands,indent=2));return
    for command in commands:subprocess.run(command,cwd=ROOT,check=True)

if __name__=='__main__':main()
