"""Build, validate and run the current MTE reference-profile branch on target hosts."""
import argparse
from contextlib import contextmanager
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import runpy
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
BRANCH='kernel/ll-g4-systemcycle-profile'
REFERENCE_SOURCE='c3d86406733fd28fe4e081e563eaf7e51321c18f'
VISIBILITY=('CUDA_VISIBLE_DEVICES','ASCEND_RT_VISIBLE_DEVICES','ASCEND_VISIBLE_DEVICES','PROBE_VISIBLE_DEVICES')
RANK_ENV=('RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE','GROUP_RANK','ROLE_RANK','MASTER_ADDR','MASTER_PORT','DEEPEP_SHMEM_ENDPOINT')


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('node_rank',type=int)
    p.add_argument('master_addr')
    p.add_argument('run_name')
    p.add_argument('--num-nodes',type=int,default=1)
    p.add_argument('--mode',choices=('reference','perf'),default='reference')
    p.add_argument('--devices',default=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'))
    p.add_argument('--tokens',nargs='+',type=int,default=[256])
    p.add_argument('--dtype',nargs='+',choices=('bf16','fp16','fp8'),default=['fp8'])
    p.add_argument('--capacity',type=int,default=256)
    p.add_argument('--hidden',type=int,default=3584)
    p.add_argument('--topk',type=int,default=8)
    p.add_argument('--experts',type=int,default=384)
    p.add_argument('--route',choices=('normal','hotspot'),default='normal')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--send-cores',type=int,default=30)
    p.add_argument('--reference-iters',type=int,default=1)
    p.add_argument('--master-port',type=int,default=os.environ.get('MASTER_PORT',29500))
    p.add_argument('--shmem-port',type=int,default=os.environ.get('SHMEM_PORT',29501))
    p.add_argument('--ready-port',type=int,default=29502)
    p.add_argument('--ready-timeout',type=int,default=1800)
    p.add_argument('--rebuild-shmem',action='store_true')
    p.add_argument('--skip-build',action='store_true')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--plan',action='store_true')
    p.add_argument('--identity-output',type=Path,help=argparse.SUPPRESS)
    a=p.parse_args(argv)
    if 16*a.num_nodes not in (16,32,64,128,256) or not 0<=a.node_rank<a.num_nodes:
        p.error('Use a supported LL domain with 16 ranks per host and node_rank in range')
    try:ipaddress.IPv4Address(a.master_addr)
    except ipaddress.AddressValueError:p.error('master_addr must be a numeric IPv4 address shared by all hosts')
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,79}',a.run_name):p.error('Unsafe run name')
    ports=(a.master_port,a.shmem_port,a.ready_port)
    if len(set(ports))!=3 or any(not 1024<=port<=65535 for port in ports):p.error('Three explicit distinct ports in 1024..65535 are required')
    devices=(a.devices or '').split(',')
    if len(devices)!=16 or any(not value.isdecimal() for value in devices) or len(set(map(int,devices)))!=16:
        p.error('Supply exactly 16 distinct physical device IDs')
    a.devices=','.join(str(int(value)) for value in devices)
    if not 1<=a.send_cores<56:p.error('send-cores must be in 1..55; receive cores are 56-send')
    if not 1<=a.reference_iters<=1000:p.error('reference-iters must be in 1..1000')
    if not 1<=a.ready_timeout<=86400:p.error('ready-timeout must be in 1..86400')
    if a.skip_build and a.rebuild_shmem:p.error('skip-build and rebuild-shmem are incompatible')
    return a


def project_path(path):
    path=Path(path);resolved=(path if path.is_absolute() else ROOT/path).resolve()
    if not resolved.is_relative_to(ROOT.resolve()):raise ValueError('Path must remain inside this project: '+str(path))
    return resolved


def run_id(a):return a.run_name+'-'+a.mode


def test_arguments(a):
    result=['--backend','mte','--num-processes','16','--num-nodes',str(a.num_nodes),'--node-rank',str(a.node_rank),
        '--master-addr',a.master_addr,'--master-port',str(a.master_port),'--shmem-port',str(a.shmem_port),
        '--num-tokens',*map(str,a.tokens),'--num-max-tokens-per-rank',str(a.capacity),'--hidden',str(a.hidden),
        '--num-topk',str(a.topk),'--num-experts',str(a.experts),'--dtype',*a.dtype,'--fp8-scale-dtype','int32',
        '--do-cpu-sync','0','--group-backend','hccl','--use-1g-huge-pages','0','--route',a.route,'--seed',str(a.seed),
        '--warmup','10','--num-iters','100','--perf-tail','50','--correctness-repeats','3','--repeat-combine','3',
        '--num-cpu-timeout-secs','300','--dispatch-only','--output-dir',str(ROOT/'results/ll-reference-profile'),
        '--run-name',run_id(a)]
    if a.mode=='reference':
        result+=['--ll-reference-profile','--ll-timer-iters',str(a.reference_iters),'--ll-timer-frequency-hz','1000000000','--skip-perf-test']
    return result


def validate_test_arguments(a):
    st=runpy.run_path(str(ROOT/'tests/elastic/test_ep_low_latency.py'))
    st['validate_args'](st['parse_args'](test_arguments(a)),16*a.num_nodes)


def command_plan(a,python):
    commands=[]
    if not a.skip_build:
        commands=[['git','submodule','update','--init','--recursive','--depth','1','--','third-party/shmem'],
            [str(python),'scripts/build_shmem.py',*(['--rebuild'] if a.rebuild_shmem else [])],
            [str(python),'-m','pip','install','-r','requirements-build.txt'],
            [str(python),'-m','pip','install','-v','--force-reinstall','--no-build-isolation','--no-deps','.']]
    commands+=[[str(python),'scripts/record_shmem_sdk.py','--check','--sdk','third-party/shmem/install/shmem'],
               [str(python),'tests/elastic/test_ep_low_latency.py',*test_arguments(a),'--plan']]
    return commands


def child_environment(a,create=False):
    env=dict(os.environ)
    for key in (*RANK_ENV,'SHMEM_ROOT','CATLASS_ROOT','DEEPEP_LL_RDMA','DEEPEP_LL_STAGE_TIMER','DEEPEP_LL_TIMER_PROFILE','DEEPEP_LL_EXPECT_TIMER_PROFILE'):
        env.pop(key,None)
    env.update({key:a.devices for key in VISIBILITY})
    env.update(DEEPEP_NPU_ARCH='Ascend950',DEEPEP_LL_BACKEND='mte',DEEPEP_LL_EXPECT_IMPLEMENTATION='g4-r0-prefix',
        DEEPEP_LL_SEND_CORES=str(a.send_cores),DEEPEP_LL_RECV_CORES=str(56-a.send_cores),DEEPEP_LL_RANKS_PER_GROUP='4',
        DEEPEP_BUILD_HT='OFF',DEEPEP_BUILD_MEGAMOE='OFF',DEEPEP_BUILD_EP_GMM_FUSED='OFF',
        DEEPEP_LL_REFERENCE_PROFILE='ON' if a.mode=='reference' else 'OFF',
        PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',GIT_TERMINAL_PROMPT='0')
    paths={'TMPDIR':'.tmp','TMP':'.tmp','TEMP':'.tmp','XDG_CACHE_HOME':'.cache','PIP_CACHE_DIR':'.cache/pip',
        'CCACHE_DIR':'.cache/ccache','TORCH_EXTENSIONS_DIR':'.cache/torch_extensions','ASCEND_CACHE_PATH':'.cache/ascend',
        'SHMEM_LOG_PATH':'.logs/shmem','ASCEND_WORK_PATH':'.logs/ascend','ASCEND_PROCESS_LOG_PATH':'.logs/ascend'}
    for key,relative in paths.items():
        directory=project_path(relative)
        if create:directory.mkdir(parents=True,exist_ok=True)
        env[key]=str(directory)
    sdk=project_path('third-party/shmem/install/shmem/lib')
    env['LD_LIBRARY_PATH']=str(sdk)+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    return env


def git_environment(env):
    result=dict(env);result.pop('LD_LIBRARY_PATH',None);result.pop('LD_PRELOAD',None);return result


def run_logged(command,env,log):
    command=list(map(str,command));print('+ '+shlex.join(command),flush=True)
    with Path(log).open('x',encoding='utf-8') as output:
        output.write('+ '+shlex.join(command)+'\n');output.flush()
        with subprocess.Popen(command,cwd=ROOT,env=git_environment(env) if command[0]=='git' else env,
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors='replace',bufsize=1) as process:
            for line in process.stdout:print(line,end='',flush=True);output.write(line);output.flush()
            code=process.wait()
        if code:raise RuntimeError(f'Command failed (exit {code}); log retained: {log}')


def installed_identity(a):
    sys.path.insert(0,str(ROOT));import build_support
    lock=json.loads((ROOT/'dependencies.lock.json').read_text())['shmem']
    sdk_root=project_path('third-party/shmem/install/shmem')
    sdk=build_support.validate_shmem_sdk(sdk_root,lock)
    fingerprint=build_support.source_fingerprint()
    package=runpy.run_path(str(ROOT/'tests/elastic/test_dispatch.py'))['_import_installed_deep_ep']()
    from deep_ep._native import require_native
    native=require_native('MTE reference installation check')
    info=json.loads((Path(package.__file__).parent/'_build_info.json').read_text())
    sdk_inputs={path.relative_to(sdk_root).as_posix():hashlib.sha256(path.read_bytes()).hexdigest()
                for path in build_support.sdk_artifacts(sdk_root)}
    version_file=Path(os.environ['ASCEND_HOME_PATH'])/'compiler/version.info'
    current_cann=dict(line.split('=',1) for line in version_file.read_text().splitlines() if '=' in line)
    validate_wheel_inputs(info,sdk,fingerprint,sdk_inputs,current_cann)
    reference=a.mode=='reference';cfg=native.low_latency_core_config()
    if native.low_latency_implementation!='g4-r0-prefix' or native.low_latency_core_protocol!=19000+a.send_cores:
        raise RuntimeError('Wrong LL branch/core protocol')
    if cfg['active_aiv']!=56 or cfg['send_cores']!=a.send_cores or cfg['recv_cores']!=56-a.send_cores:
        raise RuntimeError('Installed S/R configuration differs')
    group=native.low_latency_ranks_per_group
    if (group(16*a.num_nodes) if callable(group) else group)!=4 or native.high_throughput_enabled is not False:
        raise RuntimeError('Expected G4 and HT OFF')
    if getattr(native,'low_latency_reference_profile_enabled',None) is not reference or info.get('ll_reference_profile') is not reference:
        raise RuntimeError('Installed reference mode does not match requested mode')
    from deep_ep.utils.ll_reference_profile_parse import SCHEMA,SLOTS,FIELDS,WORDS
    expected=dict(schema=SCHEMA,slots=SLOTS,fields=FIELDS,words=list(WORDS),implementation='mte-shared-receive',
                  send_cores=a.send_cores,recv_cores=56-a.send_cores)
    if native.low_latency_reference_profile_layout()!=expected:raise RuntimeError('Reference profile layout mismatch')
    if Path(native.shmem_library_path()).resolve()!=(sdk_root/'lib/libshmem.so').resolve():
        raise RuntimeError('Loaded SHMEM is outside selected SDK')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,env=git_environment(os.environ),text=True).strip()
    args=test_arguments(a);args[args.index('--node-rank')+1]='<node>';args[args.index('--output-dir')+1]='<project-results>'
    return dict(schema=1,git_revision=revision,source_sha256=fingerprint,sdk_revision=lock['revision'],sdk_build=sdk['build'],
        sdk_manifest_sha256=hashlib.sha256((sdk_root/'deepep-sdk.json').read_bytes()).hexdigest(),
        core_protocol=19000+a.send_cores,send_cores=a.send_cores,recv_cores=56-a.send_cores,
        reference_profile=reference,reference_schema=SCHEMA,mode=a.mode,arguments=args)


def validate_wheel_inputs(info,sdk,fingerprint,sdk_inputs,current_cann):
    if info.get('source_sha256')!=fingerprint:raise RuntimeError('Installed sources differ; rebuild the complete wheel')
    if (info.get('shmem',{}).get('revision')!=sdk['revision'] or info.get('shmem',{}).get('build')!=sdk['build'] or
            info.get('shmem',{}).get('sha256')!=sdk_inputs):
        raise RuntimeError('Installed wheel and selected SDK differ; rebuild the wheel')
    expected=dict(version=current_cann.get('Version'),timestamp=current_cann.get('timestamp','unknown'))
    if not expected['version'] or any(info.get('cann',{}).get(key)!=value for key,value in expected.items()):
        raise RuntimeError('Installed wheel and current CANN version/timestamp differ; rebuild in the selected environment')


def require_idle(devices):
    output=subprocess.check_output(['npu-smi','info'],text=True)
    empty={int(value) for value in re.findall(r'No running processes found in NPU\s+(\d+)',output)}
    if not set(map(int,devices.split(','))).issubset(empty):
        raise RuntimeError('Selected devices are not reported idle; no launch and no process terminated')
    return output


@contextmanager
def project_lock():
    path=project_path('.tmp/ll-reference-one-click.lock');path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as output:output.write(f'pid={os.getpid()}\n')
    try:yield
    finally:path.unlink()


def main(argv=None):
    a=parse_args(argv);validate_test_arguments(a)
    python=ROOT/'.venv-ll-reference/bin/python'
    if a.plan:
        print(json.dumps(dict(project_root=str(ROOT),mode=a.mode,run_id=run_id(a),
            build=command_plan(a,python),launch=[str(python),'tests/elastic/test_ep_low_latency.py',*test_arguments(a)],
            analyze=['tests/ll_reference_profile/analyze.py',str(ROOT/'results/ll-reference-profile'/run_id(a))],
            reference_source=REFERENCE_SOURCE),indent=2));return 0
    if a.identity_output:
        with project_path(a.identity_output).open('x',encoding='utf-8') as stream:
            json.dump(installed_identity(a),stream,indent=2)
        return 0
    if sys.platform!='linux' or not os.environ.get('ASCEND_HOME_PATH'):
        raise RuntimeError('Target execution requires Linux and the existing CANN environment')
    env=child_environment(a,create=True)
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,env=git_environment(env),text=True).strip()
    dirty=subprocess.check_output(['git','status','--porcelain','--untracked-files=normal','--ignore-submodules=dirty'],cwd=ROOT,env=git_environment(env),text=True).strip()
    if branch!=BRANCH or dirty:raise RuntimeError('Use the clean requested branch; sources are not reset or stashed automatically')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,env=git_environment(env),text=True).strip()
    env.update(DEEPEP_LL_RUN_REVISION=revision,DEEPEP_LL_RUN_DIRTY='False')
    result=project_path(Path('results/ll-reference-profile')/run_id(a))
    if result.exists():raise FileExistsError('Run exists; use a new run_name')
    logs=project_path(Path('.logs/ll-reference')/run_id(a)/f'node-{a.node_rank}'/('prepare' if a.prepare_only else 'execute'))
    with project_lock():
        logs.mkdir(parents=True,exist_ok=False)
        venv=project_path('.venv-ll-reference');python=venv/'bin/python'
        if not python.is_file():
            if a.skip_build:raise FileNotFoundError('Isolated venv absent; omit --skip-build')
            run_logged([sys.executable,'-m','venv','--system-site-packages',venv],env,logs/'00-venv.log')
        env['PATH']=str(venv/'bin')+os.pathsep+env.get('PATH','');env['VIRTUAL_ENV']=str(venv);env.pop('PYTHONHOME',None)
        for index,command in enumerate(command_plan(a,python)):run_logged(command,env,logs/f'10-step-{index:02d}.log')
        identity=logs/'identity.json';original=list(sys.argv[1:] if argv is None else argv)
        run_logged([python,Path(__file__).resolve(),*original,'--identity-output',identity],env,logs/'20-identity.log')
        if a.prepare_only:print(f'Prepared only: {logs}');return 0
        (logs/'devices-idle.txt').write_text(require_idle(a.devices),encoding='utf-8')
        from target_ready import synchronize
        agreed=synchronize(a.node_rank,a.master_addr,a.ready_port,run_id(a),read_identity(identity),a.ready_timeout,a.num_nodes)
        (logs/'agreement.json').write_text(json.dumps(agreed,indent=2)+'\n',encoding='utf-8')
        run_logged([python,'-I',ROOT/'tests/elastic/test_ep_low_latency.py',*test_arguments(a)],env,logs/'30-test.log')
        if a.mode=='reference':
            run_logged([python,ROOT/'tests/ll_reference_profile/analyze.py',result,'--iteration','last'],env,logs/'40-analysis.log')
        print(f'PASS node {a.node_rank}: {result}\nLogs: {logs}')
    return 0


def read_identity(path):
    identity=json.loads(Path(path).read_text())
    # Firmware-specific SDK binaries are verified locally. Compare the shared
    # MTE consumer build inputs while preserving local PI/hash in identity.json.
    identity.pop('sdk_manifest_sha256',None)
    identity['sdk_build'].pop('hns_1825_pi_on_chip',None)
    return identity


if __name__=='__main__':
    try:sys.exit(main())
    except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as error:
        print(f'Reference run failed: {error}',file=sys.stderr,flush=True);sys.exit(1)
