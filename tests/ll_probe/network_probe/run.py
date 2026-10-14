"""One local launcher per host; no SSH, no HCCL process group, no pip dependency."""
import argparse
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import project_path, results_dir, runtime_environment, workspace_root
import signal
import socket
import subprocess
import time
from defaults import DEVICES_PER_HOST, DEFAULT_PORT, DEFAULT_TIMEOUT, LL_WORKSPACE_BYTES

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def hosts(path):
    result = []
    for line in project_path(path).read_text().splitlines():
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        if not line.startswith('- '):
            raise ValueError('config.yaml must contain only a YAML list: - IPv4')
        value = line[2:].strip().strip('\"\'')
        address = ipaddress.ip_address(value)
        if address.version != 4 or address.is_unspecified or address.is_multicast:
            raise ValueError('Use unicast bootstrap-reachable IPv4 addresses')
        result.append(str(address))
    if not result or len(set(result)) != len(result):
        raise ValueError('Host list must be nonempty and unique')
    return result


def identity():
    files = [*HERE.glob('kernel/*.cpp'), *HERE.glob('host/*.cpp'), *HERE.glob('*.py'), *HERE.glob('*.sh'),
             REPO / 'csrc/runtime/session.cpp', REPO / 'csrc/runtime/session.hpp',
             REPO / 'csrc/runtime/udma_peer_masks.hpp', REPO / 'csrc/kernels/common.hpp',
             REPO / 'csrc/kernels/udma_layout.hpp', REPO / 'scripts/project_paths.py',
             REPO / 'scripts/project_env.sh', REPO / 'tests/ll_probe/paths.py']
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
    ips = hosts(args.config)
    n = args.devices_per_host
    world = n * len(ips)
    if n < 1 or not 0 <= args.node_index < len(ips) or not 0 < args.peer < world:
        raise ValueError('Invalid devices/node-index/peer; peer is a GLOBAL rank')
    if world < 2 or world > 512 or world & (world - 1):
        raise ValueError('Production Session requires a power-of-two world in [2,512]')
    if args.validation:
        if len(ips) != 1 or n != 2 or args.target_run:
            raise ValueError('Validation must use one host, exactly devices 0,1, no --target-run')
    elif world not in (16,32,64,128):
        raise ValueError('LL domain supports EP16/32/64/128; use explicit --validation for EP2 Session-only smoke')
    if not 1024 <= args.port <= 65535 or not 10 <= args.timeout <= 3600:
        raise ValueError('Invalid port/timeout')
    return {'hosts': ips, 'devices_per_host': n, 'world': world, 'peer': args.peer,
            'endpoint': f'tcp://{ips[0]}:{args.port}', 'node_index': args.node_index,
            'ranks': list(range(args.node_index*n, (args.node_index+1)*n)),
            'workspace_bytes': (2<<20) if args.validation else LL_WORKSPACE_BYTES,
            'huge_pages': not (args.validation or args.no_huge_pages),
            'scope': 'Session-only EP2 validation' if args.validation else 'LL Session domain; no full LL kernel',
            'mte_actual_net_layer': None,
            'route_query_scope': 'UDMA established QPs, NOT MTE physical route'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(HERE/'config.yaml'))
    parser.add_argument('--node-index', type=int, default=0)
    parser.add_argument('--peer', type=int, default=1, help='global rank: node_index * devices_per_host + local_device')
    parser.add_argument('--devices-per-host', type=int, default=DEVICES_PER_HOST)
    parser.add_argument('--port', type=int, default=DEFAULT_PORT)
    parser.add_argument('--timeout', type=int, default=DEFAULT_TIMEOUT)
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
                os.killpg(p.pid, signal.SIGTERM)
        for p in processes:
            try: p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL); p.wait()
        for log in logs: log.close()
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
