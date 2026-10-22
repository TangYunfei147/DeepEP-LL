"""Build one isolated full-wheel variant. Explicit --prepare-sdk for initial setup."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from common import ROOT,MODES,git,sha,write_json

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=MODES,required=True)
    p.add_argument('--timer',choices=['ON','OFF'],default='OFF')
    p.add_argument('--send-cores',type=int,default=28)
    p.add_argument('--prepare-sdk',action='store_true')
    p.add_argument('--offline-deps',type=Path,help='Uploaded deps.py bundle directory; no remote Git access')
    p.add_argument('--plan',action='store_true')
    a=p.parse_args()
    if not 0<a.send_cores<56:p.error('send-cores must be 1..55')
    if a.mode=='ready-replay' and a.timer!='ON':p.error('ready-replay requires ON')
    if not a.prepare_sdk and not a.plan:p.error('Isolated first build requires --prepare-sdk')
    tag=f'{a.mode}-{a.timer}-s{a.send_cores}'
    source=ROOT/'.ablation'/'sources'/tag
    output=ROOT/'.ablation'/'wheels'/tag
    config=dict(mode=a.mode,mode_id=MODES[a.mode],timer=a.timer,send_cores=a.send_cores,
                recv_cores=56-a.send_cores,commit=git('rev-parse','HEAD'))
    if a.plan:print(config);return
    if git('status','--porcelain'):raise RuntimeError('Commit source changes before building variants')
    if source.exists() or output.exists():raise RuntimeError('Variant directory exists; use a fresh checkout, never overwrite')
    source.parent.mkdir(parents=True,exist_ok=True);output.mkdir(parents=True)
    subprocess.run(['git','-C',str(ROOT),'worktree','add','--detach',str(source),config['commit']],check=True)
    header=source/'csrc/kernels/dispatch/low_latency/common/ll_ablation_config.h'
    header.write_text(header.read_text().replace('kLlAblationMode = 0;',f'kLlAblationMode = {MODES[a.mode]};'))
    env=dict(os.environ,DEEPEP_NPU_ARCH='Ascend950',DEEPEP_LL_STAGE_TIMER=a.timer,
             DEEPEP_LL_SEND_CORES=str(a.send_cores),DEEPEP_LL_RECV_CORES=str(56-a.send_cores))
    for name in ['SHMEM_ROOT','CATLASS_ROOT']:env.pop(name,None)
    for name,rel in {'TMPDIR':'.tmp','TMP':'.tmp','TEMP':'.tmp','PIP_CACHE_DIR':'.cache/pip',
                     'XDG_CACHE_HOME':'.cache','CCACHE_DIR':'.cache/ccache',
                     'TORCH_EXTENSIONS_DIR':'.cache/torch_extensions','ASCEND_CACHE_PATH':'.cache/ascend',
                     'SHMEM_LOG_PATH':'.logs/shmem','ASCEND_WORK_PATH':'.logs/ascend',
                     'ASCEND_PROCESS_LOG_PATH':'.logs/ascend'}.items():
        dest=source/rel;dest.mkdir(parents=True,exist_ok=True);env[name]=str(dest)
    def run(command):
        with (output/'build.log').open('a') as log:
            subprocess.run(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    if a.offline_deps:
        bundle=a.offline_deps if a.offline_deps.is_absolute() else ROOT/a.offline_deps
        run([sys.executable,str(ROOT/'tests/ll_ablation/deps.py'),'unpack','--checkout',str(source),'--bundle',str(bundle)])
        env['PIP_NO_INDEX']='1'
    else:run(['git','submodule','update','--init','--recursive','--depth','1'])
    if a.prepare_sdk:run([sys.executable,'scripts/build_shmem.py'])
    else:raise RuntimeError('First isolated build requires --prepare-sdk (SDK remains inside this checkout)')
    # Existing environment must satisfy requirements-build.txt; no system dependency installation.
    run([sys.executable,'-m','pip','wheel','--no-deps','--no-build-isolation','--wheel-dir',str(output),'.'])
    wheels=list(output.glob('*.whl'))
    if len(wheels)!=1:raise RuntimeError('Expected exactly one wheel')
    config.update(wheel=wheels[0].name,wheel_sha256=sha(wheels[0]),header_sha256=sha(header),
                  source=str(source.relative_to(ROOT)),python=sys.version,
                  cann=os.environ.get('ASCEND_HOME_PATH'))
    write_json(output/'manifest.json',config)
    print(output/'manifest.json')

if __name__=='__main__':main()
