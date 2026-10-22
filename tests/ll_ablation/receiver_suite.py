"""One command: reuse ON build cache, build one wheel, run three replay arms, report."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from common import ROOT,git,sha

def commands(a):
    mode='ready-breakdown';tag=f'{mode}-ON-s{a.send_cores}'
    manifest=ROOT/'.ablation/wheels'/tag/'manifest.json'
    config=a.config if a.config.is_absolute() else ROOT/a.config
    settings=json.loads(config.read_text())
    if settings.get('num_nodes',1)!=1:raise ValueError('One-command suite currently requires one16-device host')
    revision=git('rev-parse','HEAD');result=[]
    if manifest.exists():
        info=json.loads(manifest.read_text())
        if (info['mode'],info['timer'],info['send_cores'],info['recv_cores'])!=(mode,'ON',a.send_cores,56-a.send_cores):
            raise ValueError('Existing breakdown manifest does not match')
        if sha(manifest.parent/info['wheel'])!=info['wheel_sha256']:raise ValueError('Existing wheel checksum mismatch')
        if git('diff',info['commit'],revision,'--','csrc','deep_ep','cmake','CMakeLists.txt','build_support.py',
               'setup.py','pyproject.toml','dependencies.lock.json','requirements-build.txt'):
            raise ValueError('Existing breakdown wheel has different native sources; use a fresh suite checkout')
    else:
        build=[sys.executable,str(ROOT/'tests/ll_ablation/build.py'),'--mode',mode,'--timer','ON',
               '--send-cores',str(a.send_cores),'--revision',revision,'--prepare-sdk','--resume']
        donor=ROOT/'.ablation/wheels'/f'ready-replay-ON-s{a.send_cores}'/'manifest.json'
        if not a.fresh_build and donor.is_file():build+=['--reuse-workspace-manifest',str(donor)]
        elif a.offline_deps:build+=['--offline-deps',str(a.offline_deps.resolve())]
        result.append(build)
    if a.build_only:return result
    env=ROOT/'.ablation/envs'/tag
    python=env/('Scripts/python.exe' if sys.platform=='win32' else 'bin/python')
    if not env.exists():result.append([sys.executable,'-m','venv','--system-site-packages',str(env)])
    result.append([str(python),str(ROOT/'tests/ll_ablation/install.py'),str(manifest)])
    run_name=a.run_prefix+'-ready-breakdown'
    result.append([str(python),str(ROOT/'tests/ll_ablation/run.py'),'--mode',mode,'--manifest',str(manifest),
                   '--devices',a.devices,'--run-name',run_name,'--config',str(config)])
    out=Path(settings.get('output_dir','.ablation/results'));out=out if out.is_absolute() else ROOT/out
    result.append([sys.executable,str(ROOT/'tests/ll_ablation/receiver_report.py'),str(out/run_name),
                   '--output',str(out/(a.run_prefix+'-receiver-comparison.json'))])
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--devices','--device',dest='devices')
    p.add_argument('--send-cores',type=int,default=30);p.add_argument('--run-prefix',default='recv40-001')
    p.add_argument('--config',type=Path,default=ROOT/'tests/ll_ablation/b256.json')
    p.add_argument('--fresh-build',action='store_true');p.add_argument('--offline-deps',type=Path)
    p.add_argument('--build-only',action='store_true');p.add_argument('--plan',action='store_true')
    a=p.parse_args()
    if not 0<a.send_cores<56:p.error('S must be1..55')
    if Path(a.run_prefix).name!=a.run_prefix or a.run_prefix in ('.','..'):p.error('Invalid run prefix')
    if not a.build_only:
        devices=(a.devices or '').split(',')
        if len(devices)!=16 or any(not d.isdecimal() for d in devices) or len(set(devices))!=16:p.error('Explicit16devices required')
    cmds=commands(a)
    if a.plan:print(json.dumps(cmds,indent=2));return
    for command in cmds:subprocess.run(command,cwd=ROOT,check=True)

if __name__=='__main__':main()
