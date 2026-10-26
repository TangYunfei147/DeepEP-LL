"""Measure dispatch send-phase bandwidth with matched OFF/observe/completion captures."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid
import zipfile

_spec = importlib.util.spec_from_file_location('bandwidth_profile_suite', Path(__file__).with_name('suite.py'))
suite = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(suite)
build = suite.build
ROOT = Path(__file__).resolve().parents[2]
BASELINE_REFERENCE = '9027275b31a28608296b1b71c93fa6e2bb456db5'
# Same business algorithm as BASELINE_REFERENCE, with the separately committed
# schema17 send-phase bandwidth instrumentation. Both OFF/ON wheels use this exact source.
DIAGNOSTIC_REVISION = '0ebb65b59327e69b20f6e96b456fee5352be0ecc'
STATE = Path('.tmp/ll-send-bandwidth')
CONDITIONS = dict(observe=dict(mode='observe',mode_id=1),
                  complete=dict(mode='bandwidth-complete',mode_id=6))


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument('--devices', default=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'))
    result.add_argument('--run-prefix', default='send-bandwidth-001')
    result.add_argument('--seeds', nargs='+', type=int, default=[0, 17, 101])
    result.add_argument('--timer-iters', type=int, choices=(3, 5), default=5)
    result.add_argument('--master-addr', default=os.environ.get('MASTER_ADDR'))
    result.add_argument('--master-port', type=int, default=os.environ.get('MASTER_PORT'))
    result.add_argument('--shmem-port', type=int, default=os.environ.get('SHMEM_PORT'))
    result.add_argument('--output-dir', type=Path, default=Path('results/ll-send-bandwidth'))
    modes = result.add_mutually_exclusive_group()
    modes.add_argument('--build-only', action='store_true')
    modes.add_argument('--run-only', action='store_true')
    result.add_argument('--plan', action='store_true')
    return result


def make_plan(args, root=ROOT):
    revision = build.git(root, 'rev-parse', DIAGNOSTIC_REVISION+'^{commit}')
    baseline_reference = build.git(root, 'rev-parse', BASELINE_REFERENCE+'^{commit}')
    seeds = suite.selected_seeds(args)
    if len(seeds) > 3:
        raise ValueError('This compact diagnostic accepts at most three seeds per report')
    if not args.build_only:
        suite.device_ids(args.devices)
        suite.safe_name(args.run_prefix or '')
        if not args.master_addr or any(port is None or not 1024 <= port <= 65535 for port in (args.master_port,args.shmem_port)):
            raise ValueError('Explicit MASTER_ADDR, MASTER_PORT and SHMEM_PORT in1024..65535 are required')
        if args.master_port == args.shmem_port:
            raise ValueError('MASTER_PORT and SHMEM_PORT must differ')
    home = root/STATE/(revision+'-bandwidth-17-s30-htoff')
    source=home/'source'
    builds={}
    for variant,timer in (('off','OFF'),('on','ON')):
        wheel_dir,venv=home/variant/'wheel',home/variant/'venv'
        python=venv/('Scripts/python.exe' if sys.platform=='win32' else 'bin/python')
        builds[variant]=dict(source=str(source),wheel_dir=str(wheel_dir),manifest=str(wheel_dir/'manifest.json'),
                             venv=str(venv),python=str(python),timer=timer,timer_profile='bandwidth',timer_profile_id=8)
    runs = []
    output = build.resolve(root,args.output_dir)/(args.run_prefix or 'build-only')
    if not output.resolve().is_relative_to(root.resolve()):
        raise ValueError('Bandwidth diagnostic output must stay inside the project')
    if not args.build_only:
        config = argparse.Namespace(num_nodes=1,node_rank=0,master_addr=args.master_addr,
                                    master_port=args.master_port,shmem_port=args.shmem_port)
        for index,seed in enumerate(seeds):
            order=['observe','complete'] if index % 2 == 0 else ['complete','observe']
            item=dict(seed=seed,measured_route_seed=seed+2,condition_order=['off']+order)
            for variant in item['condition_order']:
                selected=builds['off' if variant=='off' else 'on']
                directory=output/f'seed-{seed}'/variant
                command=suite.st_command(selected['python'],config,directory,root,seed=seed)
                command+=['--warmup','10','--num-iters','100','--perf-tail','50','--use-1g-huge-pages','0','--dispatch-only']
                if variant!='off':
                    command+=['--ll-stage-timer','--ll-timer-mode',CONDITIONS[variant]['mode'],'--ll-timer-iters',str(args.timer_iters),
                              '--ll-timer-frequency-hz','1000000000','--ll-profile-capture','--skip-perf-test']
                item[variant+'_directory']=str(directory)
                item[variant+'_command']=command
            runs.append(item)
    return dict(schema=1,revision=revision,baseline_reference=baseline_reference,
                runner_revision=build.git(root,'rev-parse','HEAD'),dependency=build.dependency_identity(root,revision),
                send_cores=30,recv_cores=26,shape=suite.SHAPE,timer_iters=args.timer_iters,seeds=seeds,
                builds=builds,output=str(output),runs=runs,
                configuration=dict(shape=suite.SHAPE,world_size=16,send_cores=30,recv_cores=26,logical_group=4,
                    timer_schema=17,timer_profile='bandwidth',timer_profile_id=8,timer_modes={name:dict(value) for name,value in CONDITIONS.items()},timer_iters=args.timer_iters,
                    frequency_hz=1_000_000_000,dispatch_only=True,intervention_cores='phase_owner_cores',high_throughput=False,use_1g_huge_pages=False,
                    warmup=10,num_iters=100,perf_tail=50,correctness_repeats=3,seeds=seeds),
                export_command=[sys.executable,str(root/'tests/ll_control_opt/report_bandwidth.py'),str(output),
                                '--output',str(output/'bandwidth-report.txt')])


def profile_environment(source, timer):
    env = build.environment(source,30)
    env.update(DEEPEP_LL_STAGE_TIMER=timer,DEEPEP_BUILD_HT='OFF',DEEPEP_LL_TIMER_PROFILE='bandwidth',
               DEEPEP_LL_EXPECT_TIMER_PROFILE='bandwidth')
    return env


def verify_manifest(path, plan, variant, root=ROOT):
    path=Path(path)
    data=json.loads(path.read_text(encoding='utf-8'))
    for field,expected in dict(revision=plan['revision'],timer=plan['builds'][variant]['timer'],
                               timer_schema=17,timer_profile='bandwidth',timer_profile_id=8,timer_bytes_per_core=1024,send_cores=30,recv_cores=26,
                               logical_group=4,build_type='Release',high_throughput=False,
                               **plan['dependency']).items():
        if data.get(field)!=expected:
            raise ValueError(f'Bandwidth profile manifest mismatch: {field}')
    if not isinstance(data.get('source_sha256'),str) or not re.fullmatch(r'[0-9a-f]{64}',data['source_sha256']):
        raise ValueError('Bandwidth profile manifest is missing the installed-source fingerprint')
    wheel=build.resolve(path.parent,data['wheel'])
    if not wheel.is_relative_to(path.parent.resolve()) or build.sha(wheel)!=data['wheel_sha256']:
        raise ValueError('Bandwidth profile wheel is missing or corrupt')
    sdk=build.resolve(root,data['sdk'])
    if not sdk.is_relative_to(root.resolve()):
        raise ValueError('Bandwidth profile SDK must stay inside the project')
    if build.sha(sdk/'deepep-sdk.json')!=data['sdk_manifest_sha256'] or build.sha(sdk/'lib/libshmem.so')!=data['sdk_library_sha256']:
        raise ValueError('Bandwidth profile SDK changed or is unavailable')
    if json.loads((sdk/'deepep-sdk.json').read_text(encoding='utf-8')).get('revision') != data['shmem_revision']:
        raise ValueError('Bandwidth profile SDK revision mismatch')
    return data


def build_wheel(plan, variant, root=ROOT):
    selected=plan['builds'][variant]
    manifest=Path(selected['manifest'])
    if manifest.exists():
        return verify_manifest(manifest,plan,variant,root)
    source=Path(selected['source'])
    source.parent.mkdir(parents=True,exist_ok=True)
    if source.exists():
        if not (source/'.git').exists() or build.common_dir(source)!=build.common_dir(root):
            raise ValueError('Profile cache must be an independent worktree of this repository')
        build.clean_source(source)
        if build.git(source,'rev-parse','HEAD')!=plan['revision']:
            raise ValueError('Profile cache revision differs; existing source was preserved')
    else:
        subprocess.run(['git','-C',str(root),'worktree','add','--detach',str(source),plan['revision']],check=True)
    output=manifest.parent/('attempt-'+uuid.uuid4().hex[:12])
    output.mkdir(parents=True,exist_ok=False)
    env=profile_environment(source,selected['timer'])
    build.copy_local_dependencies(root,source)
    build.check_dependencies(source)
    timings=[]
    for command in ([sys.executable,'scripts/build_shmem.py'],
                    [sys.executable,'-m','pip','wheel','-v','--no-deps','--no-build-isolation','--wheel-dir',str(output),'.']):
        start=time.perf_counter()
        print(f'Building {variant} Bandwidth diagnostic; log: {output / "build.log"}',flush=True)
        with (output/'build.log').open('a',encoding='utf-8') as log:
            log.write(repr(command)+'\n');log.flush()
            completed=subprocess.run(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
        timings.append(dict(command=command,seconds=time.perf_counter()-start,exit_code=completed.returncode))
        build.write_json(output/'timings.json',timings)
        completed.check_returncode()
    wheels=list(output.glob('*.whl'))
    if len(wheels)!=1:
        raise ValueError(f'Expected exactly one complete {variant} Bandwidth wheel')
    with zipfile.ZipFile(wheels[0]) as archive:
        native=[name for name in archive.namelist() if Path(name).name.startswith('_C.') and name.endswith('.so')]
        kernels=[name for name in archive.namelist() if Path(name).name=='libdeepep_kernels.so']
        if len(native)!=1 or len(kernels)!=1:
            raise ValueError('Cannot uniquely identify Bandwidth native libraries')
        info=json.loads(archive.read('deep_ep/_build_info.json'))
        if not re.fullmatch(r'[0-9a-f]{64}',str(info.get('source_sha256',''))):
            raise ValueError('Bandwidth wheel has no valid committed-source fingerprint')
        hashes=dict(native_sha256=hashlib.sha256(archive.read(native[0])).hexdigest(),
                    kernel_sha256=hashlib.sha256(archive.read(kernels[0])).hexdigest(),source_sha256=info['source_sha256'])
    sdk=source/build.SDK
    data=dict(revision=plan['revision'],timer=selected['timer'],timer_schema=17,timer_profile='bandwidth',timer_profile_id=8,timer_bytes_per_core=1024,
              send_cores=30,recv_cores=26,logical_group=4,
              build_type='Release',high_throughput=False,**plan['dependency'],**hashes,
              source=os.path.relpath(source,root),sdk=os.path.relpath(sdk,root),
              sdk_manifest_sha256=build.sha(sdk/'deepep-sdk.json'),sdk_library_sha256=build.sha(sdk/'lib/libshmem.so'),
              wheel=wheels[0].relative_to(manifest.parent).as_posix(),wheel_sha256=build.sha(wheels[0]))
    suite.write_new_json(manifest,data)
    return verify_manifest(manifest,plan,variant,root)


def preflight_code(manifest):
    code=suite.preflight_code(manifest,30)
    old='assert not bool(n.low_latency_stage_timer_enabled)'
    if old not in code:
        raise ValueError('Shared preflight contract changed; refusing to weaken timer verification')
    assertion='assert bool(n.low_latency_stage_timer_enabled)' if manifest['timer']=='ON' else old
    return code.replace(old,assertion) + (
        '; layout=n.low_latency_stage_timer_layout(); assert layout["schema"]==17; '
        'assert layout["recording"]=="stage-bandwidth"; assert layout["slots"]==16; assert layout["fields"]==8; assert layout["slots"]*layout["fields"]*8==1024; '
        'assert n.low_latency_stage_timer_profile=="bandwidth"; assert n.low_latency_stage_timer_profile_id==8; '
        'assert n.low_latency_overlap_stage==3; '
        'from deep_ep.utils import ll_stage_timer_bandwidth as bw; '
        'assert layout["points"]==list(bw.POINTS); assert layout["bandwidth_stages"]==list(bw.STAGES); '
        'assert layout["bandwidth_fields"]==list(bw.FIELDS); assert layout["bandwidth_word_offset"]==32; '
        'assert layout["bandwidth_record_words"]==16; '
        'assert n.high_throughput_enabled is False; assert callable(n.create_low_latency_runtime); '
        'import json; '
        f'assert json.loads(Path(n.__file__).with_name("_build_info.json").read_text())["source_sha256"]=={manifest["source_sha256"]!r}; '
        'import torch, torch_npu')


def validate_matched_builds(metadata):
    """Require one source and SDK identity before any device launch."""
    off,on=metadata['off'],metadata['on']
    for key in ('revision','source_sha256','dependency_id','shmem_revision',
                'sdk_manifest_sha256','sdk_library_sha256'):
        if not off.get(key) or off[key] != on.get(key):
            raise ValueError(f'OFF/ON Bandwidth build identity differs: {key}')


def validate_loaded_ranks(directory, manifest):
    paths=sorted(Path(directory).glob('loaded_rank_*.json'))
    if len(paths)!=16:
        raise ValueError('Expected actual loaded-library records from all16 ranks')
    ranks=set()
    for path in paths:
        data=json.loads(path.read_text(encoding='utf-8'))
        rank=data.get('rank')
        if type(rank) is not int or rank not in range(16) or rank in ranks:
            raise ValueError('Missing/duplicate/invalid loaded-library rank')
        ranks.add(rank)
        libraries=data.get('libraries',{})
        native=[value for name,value in libraries.items()
                if Path(name).name.startswith('_C.') and Path(name).name.endswith('.so')]
        if native != [manifest['native_sha256']]:
            raise ValueError(f'Loaded native extension mismatch on rank{rank}')
        for filename,key in (('libdeepep_kernels.so','kernel_sha256'),('libshmem.so','sdk_library_sha256')):
            values=[value for name,value in libraries.items() if Path(name).name==filename]
            if values != [manifest[key]]:
                raise ValueError(f'Loaded {filename} mismatch on rank{rank}')


def main():
    cli=parser();args=cli.parse_args()
    try:
        plan=make_plan(args,ROOT)
    except (ValueError,OSError,subprocess.CalledProcessError) as error:
        cli.exit(2,str(error)+'\n')
    if args.plan:
        print(json.dumps(plan,indent=2));return
    release=build.acquire_lock(ROOT/STATE/'profile.lock')
    try:
        output=Path(plan['output'])
        if not args.build_only and output.exists():
            raise ValueError('Profile results already exist; use a fresh --run-prefix')
        metadata={variant:verify_manifest(selected['manifest'],plan,variant,ROOT) if args.run_only
                  else build_wheel(plan,variant,ROOT) for variant,selected in plan['builds'].items()}
        validate_matched_builds(metadata)
        environments={}
        for variant,selected in plan['builds'].items():
            data=metadata[variant]
            venv=Path(selected['venv'])
            if not venv.exists():
                subprocess.run([sys.executable,'-m','venv','--system-site-packages',str(venv)],check=True)
            if not (venv/'pyvenv.cfg').is_file() or not Path(selected['python']).is_file():
                raise ValueError('Incomplete profile venv; it was preserved')
            config=argparse.Namespace(devices=args.devices or '',send_cores=30)
            env=suite.run_environment(config,dict(revision=plan['revision']),data,ROOT)
            env.update(DEEPEP_LL_STAGE_TIMER=selected['timer'],DEEPEP_LL_TIMER_PROFILE='bandwidth',
                       DEEPEP_LL_EXPECT_TIMER_PROFILE='bandwidth')
            environments[variant]=env
            subprocess.run([selected['python'],'-I','-m','pip','install','--no-index','--no-deps','--force-reinstall',
                            str(build.resolve(Path(selected['manifest']).parent,data['wheel']))],cwd=venv,env=env,check=True)
            try:
                subprocess.run([selected['python'],'-I','-c',preflight_code(data)],cwd=venv,env=env,check=True)
            except subprocess.CalledProcessError as error:
                raise RuntimeError(f'Bandwidth {variant} preflight failed. Check native/SDK diagnostics above and '
                    f'ensure {selected["python"]} can import matching Torch/torch_npu. '
                    'A system-site-packages venv does not inherit packages installed only in an outer venv; '
                    'no Torch package was installed automatically.') from error
        if args.build_only:
            print('OFF and ON wheels built, installed into isolated venvs and Host-verified; no NPU launched.')
            return
        output.mkdir(parents=True,exist_ok=False)
        suite.write_new_json(output/'profile-plan.json',dict(plan,build_metadata=metadata,
            note='Same-source OFF dispatch performance and same-ON-wheel observe/completion active captures. Completion mode isolates each send phase with MTE3_S waits and records an empty-wait control. It changes overlap and is diagnostic, not an optimization or a pure-link bandwidth measurement; device intervals remain distinct from profiler task time.'))
        for item in plan['runs']:
            for variant in item['condition_order']:
                build_variant='off' if variant=='off' else 'on'
                idle=suite.require_idle(args.devices)
                directory=Path(item[variant+'_directory'])
                directory.parent.mkdir(parents=True,exist_ok=True)
                suite._write_new_file(Path(str(directory)+'.idle.txt'),idle)
                suite.write_new_json(Path(str(directory)+'.build.json'),metadata[build_variant])
                suite.run_logged(item[variant+'_command'],str(directory)+'.log',environments[build_variant])
                validate_loaded_ranks(directory,metadata[build_variant])
        subprocess.run(plan['export_command'],cwd=ROOT,check=True)
        report=output/'bandwidth-report.txt'
        if not report.is_file() or report.stat().st_size>65535:
            raise ValueError('Compact Bandwidth report is missing or exceeds65535 bytes')
        print(f'Send only {report} ({report.stat().st_size} bytes,1 file)',flush=True)
    finally:
        release()


if __name__=='__main__':
    main()
