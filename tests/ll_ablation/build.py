"""Build one isolated full-wheel variant. Explicit --prepare-sdk for initial setup."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import hashlib
import zipfile
import json
import uuid
import time
import atexit
from common import ROOT,MODES,git,sha,write_json,source_checkout

def build_paths(root,tag,resume):
    source=root/'.ablation'/'sources'/tag
    canonical=root/'.ablation'/'wheels'/tag
    if (canonical/'manifest.json').exists():
        raise RuntimeError('Completed variant exists; use suite.py --resume to validate and reuse it')
    output=canonical
    if source.exists() or canonical.exists():
        if not resume:
            raise RuntimeError('Incomplete variant directory exists; retry with --resume (old files are preserved)')
        if canonical.exists() and (not canonical.is_dir() or canonical.is_symlink()):
            raise RuntimeError('Variant output must be a real directory')
        attempt='attempt-'+uuid.uuid4().hex
        source=source.with_name(tag+'-'+attempt)
        output=canonical/attempt
        print(f'Preserving incomplete variant. Fresh build attempt: {source}; logs: {output}',flush=True)
    return source,output,canonical

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=MODES,required=True)
    p.add_argument('--timer',choices=['ON','OFF'],default='OFF')
    p.add_argument('--send-cores',type=int,default=28)
    p.add_argument('--prepare-sdk',action='store_true')
    p.add_argument('--offline-deps',type=Path,help='Uploaded deps.py bundle directory; no remote Git access')
    p.add_argument('--revision',default='HEAD',help='Pinned source revision shared by all resumed variants')
    p.add_argument('--resume',action='store_true',help='Preserve incomplete directories and build in a fresh attempt')
    p.add_argument('--reuse-workspace-manifest',type=Path,help='Verified ready-replay ON cache for receiver breakdown only')
    p.add_argument('--plan',action='store_true')
    a=p.parse_args()
    if not 0<a.send_cores<56:p.error('send-cores must be 1..55')
    if a.mode in ('ready-replay','ready-breakdown') and a.timer!='ON':p.error('ready-replay requires ON')
    if not a.prepare_sdk and not a.plan:p.error('Isolated first build requires --prepare-sdk')
    if a.reuse_workspace_manifest and a.mode!='ready-breakdown':p.error('Cache reuse is restricted to ready-breakdown')
    tag=f'{a.mode}-{a.timer}-s{a.send_cores}'
    config=dict(mode=a.mode,mode_id=MODES[a.mode],timer=a.timer,send_cores=a.send_cores,
                recv_cores=56-a.send_cores,logical_group=4,commit=git('rev-parse',a.revision+'^{commit}'))
    if a.plan:print(config);return
    config.update(source_checkout())
    source,output,canonical=build_paths(ROOT,tag,a.resume)
    source.parent.mkdir(parents=True,exist_ok=True);output.mkdir(parents=True)
    reuse=bool(a.reuse_workspace_manifest)
    if reuse:
        from cache_workspace import advance,inspect,acquire
        donor=a.reuse_workspace_manifest if a.reuse_workspace_manifest.is_absolute() else ROOT/a.reuse_workspace_manifest
        cache_source,_=inspect(ROOT,donor,config['commit'],a.send_cores)
        atexit.register(acquire(cache_source))
        source=advance(ROOT,donor,config['commit'],a.send_cores,MODES[a.mode])
        print(f'Reusing native cache and SDK in {source}; dependency build checks remain enabled',flush=True)
    else:
        subprocess.run(['git','-C',str(ROOT),'worktree','add','--detach',str(source),config['commit']],check=True)
    header=source/'csrc/kernels/dispatch/low_latency/common/ll_ablation_config.h'
    if not reuse:header.write_text(header.read_text().replace('kLlAblationMode = 0;',f'kLlAblationMode = {MODES[a.mode]};'))
    env=dict(os.environ,DEEPEP_NPU_ARCH='Ascend950',DEEPEP_LL_STAGE_TIMER=a.timer,
             DEEPEP_LL_SEND_CORES=str(a.send_cores),DEEPEP_LL_RECV_CORES=str(56-a.send_cores),
             DEEPEP_LL_RANKS_PER_GROUP='4')
    for name in ['SHMEM_ROOT','CATLASS_ROOT']:env.pop(name,None)
    for name,rel in {'TMPDIR':'.tmp','TMP':'.tmp','TEMP':'.tmp','PIP_CACHE_DIR':'.cache/pip',
                     'XDG_CACHE_HOME':'.cache','CCACHE_DIR':'.cache/ccache',
                     'TORCH_EXTENSIONS_DIR':'.cache/torch_extensions','ASCEND_CACHE_PATH':'.cache/ascend',
                     'SHMEM_LOG_PATH':'.logs/shmem','ASCEND_WORK_PATH':'.logs/ascend',
                     'ASCEND_PROCESS_LOG_PATH':'.logs/ascend'}.items():
        dest=source/rel;dest.mkdir(parents=True,exist_ok=True);env[name]=str(dest)
    timings=[]
    def run(command):
        start=time.perf_counter();code=None
        try:
            with (output/'build.log').open('a') as log:
                log.write('\nCOMMAND '+repr(command)+'\n');log.flush()
                code=subprocess.run(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
            if code:raise subprocess.CalledProcessError(code,command)
        finally:
            timings.append(dict(command=command,seconds=time.perf_counter()-start,exit_code=code))
            write_json(output/'build-timing.json',timings)
    if reuse:
        env['PIP_NO_INDEX']='1';env['GIT_ALLOW_PROTOCOL']='file'
    elif a.offline_deps:
        bundle=a.offline_deps if a.offline_deps.is_absolute() else ROOT/a.offline_deps
        run([sys.executable,str(ROOT/'tests/ll_ablation/deps.py'),'unpack','--checkout',str(source),'--bundle',str(bundle)])
        env['PIP_NO_INDEX']='1'
        env['GIT_ALLOW_PROTOCOL']='file'
    else:run(['git','submodule','update','--init','--recursive','--depth','1'])
    if a.prepare_sdk:run([sys.executable,'scripts/build_shmem.py'])
    else:raise RuntimeError('First isolated build requires --prepare-sdk (SDK remains inside this checkout)')
    # Existing environment must satisfy requirements-build.txt; no system dependency installation.
    run([sys.executable,'-m','pip','wheel','-v','--no-deps','--no-build-isolation','--wheel-dir',str(output),'.'])
    wheels=list(output.glob('*.whl'))
    if len(wheels)!=1:raise RuntimeError('Expected exactly one wheel')
    with zipfile.ZipFile(wheels[0]) as archive:
        native=[n for n in archive.namelist() if Path(n).name.startswith('_C.') and n.endswith('.so')]
        if len(native)!=1:raise RuntimeError('Cannot identify native extension in wheel')
        config['native_sha256']=hashlib.sha256(archive.read(native[0])).hexdigest()
    config.update(wheel=str(wheels[0].relative_to(canonical)),wheel_sha256=sha(wheels[0]),header_sha256=sha(header),
                  source=str(source.relative_to(ROOT)),python=sys.version,
                  cann=os.environ.get('ASCEND_HOME_PATH'),cache_reused=reuse)
    # Never replace a manifest published by another successful build.
    with (canonical/'manifest.json').open('x') as manifest:
        json.dump(config,manifest,indent=2,sort_keys=True)
        manifest.write('\n')
    print(canonical/'manifest.json')

if __name__=='__main__':main()
