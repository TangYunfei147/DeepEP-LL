"""One local launcher per host; no SSH, no HCCL process group, no pip dependency."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import project_path, results_dir, runtime_environment, workspace_root
from deployment import load_config, make_plan as deployment_plan
import signal
import socket
import subprocess
import time
from defaults import LL_WORKSPACE_BYTES

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def hosts(path):
    config = load_config(path)
    return deployment_plan(config, test='network')['ips']


def identity():
    files = [*HERE.glob('kernel/*.cpp'), *HERE.glob('host/*.cpp'), *HERE.glob('*.py'), *HERE.glob('*.sh'),
             REPO / 'csrc/runtime/session.cpp', REPO / 'csrc/runtime/session.hpp',
             REPO / 'csrc/runtime/udma_peer_masks.hpp', REPO / 'csrc/kernels/common.hpp',
             REPO / 'csrc/kernels/udma_layout.hpp', REPO / 'scripts/project_paths.py',
             REPO / 'scripts/project_env.sh', REPO / 'tests/ll_probe/paths.py', REPO / 'tests/ll_probe/deployment.py']
    return {'revision': subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
            'shmem_revision': subprocess.check_output(['git','-C',str(REPO/'third-party/shmem'),'rev-parse','HEAD'],text=True).strip(),
            'source_sha256': {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}}


def library_identity():
    libdir=project_path('third-party/shmem/install/shmem/lib')
    library=libdir/'libshmem.so'
    cache=REPO/'third-party/shmem/build/CMakeCache.txt'
    flags=[line for line in cache.read_text().splitlines() if line.startswith(('ACLSHMEM_MULTI_LINK_SUPPORT:',
           'ACLSHMEM_RELAY_SUPPORT:','ENABLE_CANN_BUILD:','SOC_TYPE:'))] if cache.exists() else []
    return {'path':str(library),'sha256':hashlib.sha256(library.read_bytes()).hexdigest(),'cmake_flags':flags}


def plan(args):
    config = load_config(args.config)
    if args.port is not None:
        config['shmem_port'] = args.port
    if args.timeout is not None:
        config['timeout'] = args.timeout
    if args.validation and args.target_run:
        raise ValueError('Validation cannot also request target-run')
    run_id = getattr(args, 'run_id', None) or os.environ.get('DEEPEP_PROBE_RUN_ID') or 'plan-only'
    base = deployment_plan(config, args.node_index, args.devices_per_host,
                           test='network', run_id=run_id, validation=args.validation)
    world = base['world']
    if not 0 < args.peer < world:
        raise ValueError('peer must be a nonzero GLOBAL rank inside the actual world')
    topology = dict(ips=base['ips'], rank_num_per_server=base['local_world_size'], world=world,
                    master_addr=base['master_addr'], master_port=base['master_port'],
                    shmem_port=base['shmem_port'])
    return dict(base, peer=args.peer,
                workspace_bytes=(2<<20) if args.validation else LL_WORKSPACE_BYTES,
                huge_pages=not (args.validation or args.no_huge_pages),
                scope='Session-only domain diagnostic; no full LL kernel',
                mte_actual_net_layer=None,
                deployment_identity=dict(run_id=run_id, topology=topology),
                route_query_scope='UDMA established QPs, NOT MTE physical route')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(HERE/'config.yaml'))
    parser.add_argument('--node-index', type=int, default=0)
    parser.add_argument('--run-id', help='Shared run identity on every target host')
    parser.add_argument('--peer', type=int, default=1, help='global rank: node_index * devices_per_host + local_device')
    parser.add_argument('--devices-per-host', type=int, default=None)
    parser.add_argument('--port', type=int, default=None)
    parser.add_argument('--timeout', type=int, default=None)
    parser.add_argument('--target-run', action='store_true', help='User target machines only: permit visibility expansion')
    parser.add_argument('--validation', action='store_true')
    parser.add_argument('--no-huge-pages', action='store_true', help='Explicit departure from ElasticBuffer default')
    parser.add_argument('--plan', action='store_true', help='CPU only, starts no ranks')
    parser.add_argument('--output', type=project_path)
    parser.add_argument('--write-build-manifest', action='store_true')
    args = parser.parse_args()
    if args.write_build_manifest:
        manifest=identity(); manifest['shmem_library']=library_identity()
        (HERE/'build/identity.json').write_text(json.dumps(manifest, indent=2)+'\n')
        return
    job = plan(args)
    if args.plan:
        print(json.dumps(job, indent=2)); return
    if not args.validation and not args.target_run:
        raise ValueError('Use --target-run on target machines; A5 validation uses --validation --devices-per-host 2')
    if job['run_id'] == 'plan-only':
        if not args.validation:
            raise ValueError('Use the shared launcher or supply the same --run-id on every host')
        args.run_id = f'validation-{time.time_ns()}'
        job = plan(args)
    args.devices_per_host = job['devices_per_host']
    args.timeout = job['timeout']
    root = runtime_environment()
    output = (args.output or root/'results'/f'network-{time.time_ns()}-node{args.node_index}').resolve()
    output.relative_to(root)
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.setdefault('SHMEM_LOG_LEVEL','INFO')
    visible = ','.join(map(str, range(args.devices_per_host))) if args.target_run else '0,1'
    for key in ('PROBE_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES','ASCEND_RT_VISIBLE_DEVICES','ASCEND_VISIBLE_DEVICES'):
        if args.validation and env.get(key, '0,1') != '0,1':
            raise ValueError(f'Validation requires {key}=0,1')
        env[key] = visible
    for key, folder in {'TMPDIR':'.tmp','TMP':'.tmp','TEMP':'.tmp','XDG_CACHE_HOME':'.cache',
                        'CUDA_CACHE_PATH':'.cache/cuda','TORCH_EXTENSIONS_DIR':'.cache/torch_extensions',
                        'TRITON_CACHE_DIR':'.cache/triton','PIP_CACHE_DIR':'.cache/pip','CCACHE_DIR':'.cache/ccache',
                        'SHMEM_LOG_PATH':'.logs/shmem','ASCEND_WORK_PATH':'.logs/ascend',
                        'ASCEND_PROCESS_LOG_PATH':'.logs/ascend'}.items():
        env[key] = str(root/folder); (root/folder).mkdir(parents=True, exist_ok=True)
    built = json.loads((HERE/'build/identity.json').read_text())
    current = identity()
    library=library_identity()
    if (built['source_sha256'] != current['source_sha256'] or built['shmem_revision'] != current['shmem_revision']
            or built['shmem_library']['sha256'] != library['sha256']):
        raise ValueError('Sources changed since build; rebuild first')
    exe = HERE/'build/network_probe'
    job.update(identity=current, build_identity=built, executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
               shmem_library=library,hostname=socket.gethostname(),
               boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
               dense_qps=env.get('DEEPEP_SHMEM_DENSE_QPS','0'))
    (output/'plan.json').write_text(json.dumps(job, indent=2)+'\n')
    # The unified launcher sends SIGTERM to this leader on timeout. Convert it
    # into normal unwinding so separately grouped device ranks are reaped.
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    previous_term = signal.signal(signal.SIGTERM, interrupted)
    processes, logs = [], []
    failed = False
    try:
        for device, rank in enumerate(job['ranks']):
            log = (output/f'rank-{rank}.log').open('w'); logs.append(log)
            cmd = [str(exe), str(rank), str(job['world']), str(device), str(args.peer), job['endpoint'],
                   str(job['workspace_bytes']), str(args.timeout), str(int(job['huge_pages']))]
            processes.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True))
        (output/'processes.json').write_text(json.dumps(dict(zip(job['ranks'],[p.pid for p in processes])),indent=2)+'\n')
        deadline = time.monotonic()+args.timeout*3+60
        while True:
            codes = [p.poll() for p in processes]
            if any(c not in (None,0) for c in codes):
                failed=True; break
            if all(c == 0 for c in codes):
                break
            if time.monotonic()>deadline:
                failed=True; break
            time.sleep(.2)
    finally:
        for p in processes:
            if p.poll() is None:
                try: os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError: pass
        deadline = time.monotonic() + 3
        for p in processes:
            try: p.wait(timeout=max(0, deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                try: os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError: pass
        for p in processes: p.wait()
        for log in logs: log.close()
        signal.signal(signal.SIGTERM, previous_term)
    records=[]
    for rank in job['ranks']:
        for line in (output/f'rank-{rank}.log').read_text(errors='replace').splitlines():
            if line.startswith('RESULT '): records.append(json.loads(line[7:]))
    complete={r['rank'] for r in records if r['kind']=='complete'}
    success=not failed and complete==set(job['ranks'])
    (output/'report.json').write_text(json.dumps({'local_success':success,'job':job,'records':records},indent=2)+'\n')
    print(f'{output}/report.json: local_success={success}; all nodes must succeed')
    if not success: raise SystemExit(1)


if __name__ == '__main__':
    main()
