"""Profile existing native probes with the shared reference sampling protocol.

Use torchrun for comm/shmem or per-rank local metadata; peer is one process. Target
kernel tasks are extracted by name; Event/Host CSV remains diagnostic only.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import ctypes
from datetime import timedelta
import hashlib
import json
import os
import subprocess
import socket
import traceback

from startup_diagnostics import StartupDiagnostics
from bandwidth_model import load_hardware, comm_edges
from profile_common import Capture, read_named_tasks, WARMUP, CAPTURE_STEPS, PROFILE_STEPS, DELAY_SHAPE

HERE = Path(__file__).resolve().parents[1]
from paths import workspace_root, project_path, runtime_environment, results_dir
ROOT = workspace_root()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind', choices=('comm', 'metadata', 'peer', 'shmem'), required=True)
    p.add_argument('--output-dir', type=project_path, required=True, help='Fresh run directory shared by ranks')
    p.add_argument('--endpoint', help='SHMEM endpoint, separate from torchrun MASTER_PORT')
    p.add_argument('--bytes', type=int, help='Payload bytes per communication edge; default 15392 without shape')
    p.add_argument('--bs', type=int, help='Tokens/rows per edge, used with --hidden-size')
    p.add_argument('--hidden-size', type=int, help='Elements per row for communication payload sizing')
    p.add_argument('--element-bytes', type=int, choices=(1,2,4,8), default=2)
    p.add_argument('--hardware-config', type=project_path, help='Editable bandwidth-reference YAML inside project')
    p.add_argument('--cores', type=int, default=8)
    p.add_argument('--lanes', type=int, default=4)
    p.add_argument('--chunk', type=int, default=4096)
    p.add_argument('--scenario', type=int, choices=(1, 2, 3, 4), default=1)
    p.add_argument('--transport',choices=('ubmem','ndr'),default='ubmem',help='Compile-time selected communication backend; no silent fallback')
    p.add_argument('--fanout', type=int, help='Scenario3 only: rank0 sends to ranks 1..N; default world-1')
    p.add_argument('--targets',type=int,nargs='+',help='Global destination ranks for scenario1/3; sender remains rank0')
    p.add_argument('--sources',type=int,nargs='+',help='Global source ranks for scenario2; receiver remains rank0')
    p.add_argument('--layer', type=int, choices=(0, 1, 3), default=3)
    p.add_argument('--mode', type=int, choices=(0, 1, 2, 3), default=3)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--hidden', type=int, default=4096)
    p.add_argument('--top-k', type=int, default=6)
    p.add_argument('--experts', type=int, default=256)
    p.add_argument('--partitions', type=int, default=16)
    p.add_argument('--warmup', type=int, default=WARMUP)
    p.add_argument('--samples', type=int, default=CAPTURE_STEPS)
    p.add_argument('--tail', type=int, default=PROFILE_STEPS)
    p.add_argument('--delay-rows', type=int, default=DELAY_SHAPE[0])
    p.add_argument('--delay-cols', type=int, default=DELAY_SHAPE[1])
    p.add_argument('--plan', action='store_true', help='CPU-only validation and command plan')
    return p


def resolve_payload(args):
    shape = args.bs is not None or args.hidden_size is not None
    if shape:
        if args.kind == 'metadata' or args.bs is None or args.hidden_size is None or args.bs <= 0 or args.hidden_size <= 0:
            raise ValueError('Communication shape requires positive --bs and --hidden-size; metadata uses --batch/--hidden')
        size=args.bs*args.hidden_size*args.element_bytes
        if args.bytes is not None and args.bytes!=size:
            raise ValueError('--bytes disagrees with bs*hidden-size*element-bytes')
    else:
        size=15392 if args.bytes is None else args.bytes
    if size<=0 or size%32:
        raise ValueError('Payload must be positive and 32-byte aligned; no implicit padding')
    args.bytes=size
    return dict(mode='shape' if shape else 'bytes',bs=args.bs,hidden_size=args.hidden_size,
                element_bytes=args.element_bytes if shape else None,bytes_per_edge=size)


def plan(args, environ):
    if args.transport!='ubmem' and args.kind!='comm':raise ValueError('NDR is available only for comm/empty/tiling probes')
    payload=resolve_payload(args)
    hardware=None
    if args.hardware_config is not None:
        args.hardware_config=project_path(args.hardware_config)
        if not args.hardware_config.is_relative_to(ROOT):raise ValueError('Hardware config must be inside project')
        hardware=load_hardware(args.hardware_config)
        if hardware["specification"].get("transport","ubmem")!=args.transport:
            raise ValueError("Hardware reference transport mismatch; NDR needs a NIC-specific model with transport: ndr")
    world = int(environ.get('WORLD_SIZE', '1'))
    rank = int(environ.get('RANK', '0'))
    device = int(environ.get('LOCAL_RANK', '0'))
    visible = environ.get('PROBE_VISIBLE_DEVICES', environ.get('ASCEND_RT_VISIBLE_DEVICES', environ.get('ASCEND_VISIBLE_DEVICES')))
    if not 0 <= rank < world or device < 0 or (visible is not None and device >= len(visible.split(','))):
        raise ValueError('Rank/device outside declared world or visible devices')
    args.output_dir = project_path(args.output_dir)
    if args.output_dir == ROOT or ROOT not in args.output_dir.parents:
        raise ValueError('Output must be below authorized root')
    if not 1 <= args.samples <= 10000 or not 0 <= args.warmup <= 10000 or not 0 <= args.tail <= args.samples:
        raise ValueError('Invalid warmup/capture/tail counts; set --tail explicitly for short validation')
    if args.delay_rows <= 0 or args.delay_cols <= 0:
        raise ValueError('Delay shape must be positive')
    if args.kind == 'peer' and (world != 1 or device != 0):
        raise ValueError('peer uses one process on device0 and additionally owns device1')
    if args.kind == 'metadata' and world not in (1, 2, 4, 8, 16, 32):
        raise ValueError('metadata sampling supports actual worlds 1/2/4/8/16/32; logical partitions are independent')
    if args.kind in ('comm', 'shmem') and not args.endpoint:
        raise ValueError('SHMEM endpoint required')
    if args.fanout is not None and (args.kind != 'comm' or args.scenario != 3):
        raise ValueError('--fanout is only valid for comm scenario3')
    if (args.targets is not None or args.sources is not None) and args.kind != 'comm':
        raise ValueError('Rank selection applies only to comm')
    if args.kind == 'comm':
        if world not in (2, 4, 8, 16, 32):
            raise ValueError('comm supports actual worlds 2/4/8/16/32')
        edges=comm_edges(world,args.scenario,args.fanout,args.targets,args.sources)
        if args.targets is not None:args.targets=sorted(args.targets)
        if args.sources is not None:args.sources=sorted(args.sources)
        selected_ranks=sorted({s if args.scenario==2 else d for s,d in edges})
        peer_mask=0 if args.scenario==4 else sum(1<<v for v in selected_ranks)
        fanout=len(selected_ranks) if args.scenario==3 else world-1
        argv = [rank, world, device, args.endpoint, args.bytes, args.cores,
                args.lanes, args.scenario, args.layer, args.warmup, args.samples, fanout, peer_mask]
        library = HERE/('communication/build/libcomm_ndr_profile.so' if args.transport=='ndr' else 'communication/build/libcomm_profile.so')
        prefix='four_ndr' if args.transport=='ndr' else 'four_comm'
        kernel = 'four_empty' if args.layer == 0 else f'{prefix}_{args.scenario}'
    elif args.kind == 'metadata':
        argv = [args.batch, args.hidden, args.top_k, args.experts, args.partitions, args.warmup, args.samples, device]
        library, kernel = HERE/'metadata/build/libmetadata_profile.so', 'four_metadata'
    elif args.kind == 'peer':
        argv = [args.bytes, args.cores, args.lanes, args.chunk, args.mode, args.warmup, args.samples]
        library, kernel = HERE/'bandwidth/build/libpeer_profile.so', 'll_transfer_probe'
    else:
        if world != 2:
            raise ValueError('Legacy shmem probe requires exactly two ranks')
        argv = [rank, device, args.endpoint, args.bytes, args.cores, args.lanes, args.chunk,
                args.mode, args.warmup, args.samples]
        library, kernel = HERE/'bandwidth/build/libshmem_profile.so', 'll_transfer_probe'
    deployment_keys = ('DEEPEP_PROBE_RUN_ID', 'DEEPEP_PROBE_TOPOLOGY', 'DEEPEP_PROBE_NODE_INDEX')
    present = [key in environ for key in deployment_keys]
    if any(present) and not all(present):
        raise ValueError('Incomplete unified deployment environment')
    topology = json.loads(environ.get('DEEPEP_PROBE_TOPOLOGY', 'null'))
    if all(present):
        if not environ['DEEPEP_PROBE_RUN_ID'].strip() or not isinstance(topology, dict):
            raise ValueError('Invalid unified deployment identity')
        if int(environ['DEEPEP_PROBE_NODE_INDEX']) < 0:
            raise ValueError('Invalid deployment node index')
    deployment = ({'deployment_identity': dict(run_id=environ.get('DEEPEP_PROBE_RUN_ID'), topology=topology)}
                  if 'DEEPEP_PROBE_RUN_ID' in environ or 'DEEPEP_PROBE_TOPOLOGY' in environ else {})
    scope = (dict(kind='local_metadata_per_rank', sampling_world=world, logical_source_rank=0,
                  logical_partitions=args.partitions, partitions_are_device_world=False,
                  includes_cross_rank_metadata_exchange=False) if args.kind == 'metadata'
             else dict(kind=args.kind, sampling_world=world))
    if args.kind == 'comm':
        scope.update(transport=args.transport,completion=('per-core per-peer QP quiet' if args.transport=='ndr' else 'local MTE drain') if args.layer==3 else 'no payload')
        scope.update(sender_ranks=sorted({s for s,d in edges}),receiver_ranks=sorted({d for s,d in edges}),
                     edge_count=len(edges),peer_mask=peer_mask)
        if args.scenario==3:scope['fanout']=fanout
    return dict(rank=rank, world=world, device=device, payload=payload, hardware_profile=hardware,
                measurement_scope=scope, **deployment, library=str(library), kernel=kernel,
                argv=['profile_probe', *map(str, argv)],
                warmup=args.warmup, capture_steps=args.samples, sample_window=args.tail,
                delay_shape=[args.delay_rows, args.delay_cols],
                reference_defaults=(args.warmup, args.samples, args.tail, args.delay_rows, args.delay_cols)
                    == (WARMUP, CAPTURE_STEPS, PROFILE_STEPS, *DELAY_SHAPE),
                rank_alignment='HCCL scalar all_reduce' if world > 1 else 'single-process identity; no distributed alignment',
                stream_ordering='Exp/all_reduce followed by device sync before native ACL stream launch',
                measurement='named_kernel_task_duration_not_end_to_end')


def main(argv=None):
    args = parser().parse_args(argv)
    visible = os.environ.get('PROBE_VISIBLE_DEVICES')
    if visible is not None:
        for variable in ('ASCEND_RT_VISIBLE_DEVICES', 'ASCEND_VISIBLE_DEVICES'):
            os.environ[variable] = visible
    identity = plan(args, os.environ)
    if args.plan:
        print(json.dumps(identity, indent=2))
        return
    if args.transport=='ndr':
        import runpy
        support=runpy.run_path(str(HERE/'communication/ndr_support.py'))
        identity['ndr_build_identity']=support['verify_runtime']()
    runtime_environment()
    if 'deployment_identity' in identity:
        identity['rank_provenance'] = dict(
            hostname=socket.gethostname(),
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            node_index=int(os.environ.get('DEEPEP_PROBE_NODE_INDEX', '0')))
    rank, world, device = (identity[key] for key in ('rank', 'world', 'device'))
    out = args.output_dir.resolve()/f'rank_{rank}'
    out.mkdir(parents=True, exist_ok=False)
    with StartupDiagnostics(out, rank) as startup:
        # Set before importing Torch, including when launched directly without run.sh.
        for key, value in {
            'TMPDIR': ROOT/'.tmp', 'TMP': ROOT/'.tmp', 'TEMP': ROOT/'.tmp',
            'XDG_CACHE_HOME': ROOT/'.cache', 'ASCEND_CACHE_PATH': ROOT/'.cache/ascend',
            'SHMEM_LOG_PATH': out/'shmem_logs', 'ASCEND_WORK_PATH': out/'ascend_logs',
            'ASCEND_PROCESS_LOG_PATH': out/'ascend_logs',
        }.items():
            value.mkdir(parents=True, exist_ok=True)
            os.environ[key] = str(value)
        os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
        startup.mark('torch_import_begin')
        import torch
        import torch.distributed as dist
        import torch_npu  # noqa: F401
        startup.mark('torch_import_done')
        startup.mark('set_device_begin')
        torch.npu.set_device(device)
        startup.mark('set_device_done')
        if world > 1:
            startup.mark('hccl_init_begin')
            dist.init_process_group('hccl', rank=rank, world_size=world, timeout=timedelta(seconds=180))
            startup.mark('hccl_init_done')
            # Initialize all contexts before SHMEM's short client retry window starts.
            startup.mark('hccl_tensor_begin')
            temp = torch.ones(1, device='npu')
            startup.mark('hccl_all_reduce_begin')
            dist.all_reduce(temp)
            startup.mark('hccl_all_reduce_submitted')
            startup.mark('hccl_sync_begin')
            torch.npu.synchronize()
            startup.mark('hccl_sync_done')
            del temp
        startup.mark('binary_hash_begin')
        libpath = Path(identity['library'])
        identity['binary_sha256'] = hashlib.sha256(libpath.read_bytes()).hexdigest()
        startup.mark('binary_hash_done')
        startup.mark('git_identity_begin')
        identity['git_revision'] = subprocess.check_output(['git', '-C', str(HERE), 'rev-parse', 'HEAD'], text=True).strip()
        identity['git_dirty'] = bool(subprocess.check_output(['git', '-C', str(HERE), 'status', '--porcelain'], text=True).strip())
        startup.mark('git_identity_done')
        startup.mark('source_hash_begin')
        identity['source_sha256'] = {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(HERE.rglob('*')) if p.is_file() and p.suffix in ('.py', '.cpp', '.h', '.sh', '.txt')
            and not {'build', 'snapshot', '__pycache__'}.intersection(p.relative_to(HERE).parts)}
        startup.mark('source_hash_done')
        startup.mark('identity_write_begin')
        identity['configuration'] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        (out/'identity.json').write_text(json.dumps(identity, indent=2))
        startup.mark('identity_write_done')
        capture = None
        errors = []
        phases = []

        @ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int)
        def hook(phase):
            nonlocal capture
            try:
                phases.append(phase)
                if phase == 0:
                    startup.stop('native_validation_done')
                    capture = Capture(out/'trace', args.delay_rows, args.delay_cols, bridge=True)
                elif phase == 1:
                    capture.start_capture()
                elif phase == 2:
                    capture.before_sample()
                elif phase == 3:
                    capture.close()
                else:
                    raise ValueError(f'Unknown native profiling phase {phase}')
                return 0
            except BaseException:
                errors.append(traceback.format_exc())
                return 1

        startup.mark('library_load_begin')
        library = ctypes.CDLL(str(libpath), mode=ctypes.RTLD_LOCAL)
        startup.mark('library_load_done')
        library.ll_probe_run.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p), type(hook)]
        library.ll_probe_run.restype = ctypes.c_int
        encoded = [arg.encode() for arg in identity['argv']]
        cargv = (ctypes.c_char_p*len(encoded))(*encoded)
        # C++ validation and CSV diagnostics are kept per rank, separate from Python logs.
        libc = ctypes.CDLL(None)
        saved = [os.dup(1), os.dup(2)]
        rc = -1
        try:
            with (out/'event_diagnostics.csv').open('w') as csvfile, (out/'native.log').open('w') as logfile:
                sys.stdout.flush(); sys.stderr.flush(); libc.fflush(None)
                os.dup2(csvfile.fileno(), 1); os.dup2(logfile.fileno(), 2)
                try:
                    startup.mark('native_init_validation_begin')
                    rc = library.ll_probe_run(len(encoded), cargv, hook)
                finally:
                    libc.fflush(None)
                    os.dup2(saved[0], 1); os.dup2(saved[1], 2)
        finally:
            for fd in saved:
                os.close(fd)
            if capture is not None and not capture.closed:
                capture.close(success=False)
        if rc or errors:
            (out/'failure.json').write_text(json.dumps(dict(returncode=rc, errors=errors, phases=phases), indent=2))
            raise RuntimeError(f'Native validation/profiling failed: {out}/native.log')
        if phases != [0, 1]+[2]*args.samples+[3]:
            raise RuntimeError(f'Unexpected capture phases: {phases}')
        # Legacy shmem rank1 receives only; it must not invent sender task samples.
        groups = [] if args.kind == 'shmem' and rank == 1 else read_named_tasks(out/'trace', identity['kernel'], args.samples, args.tail)
        report = dict(**identity, correctness_passed=True, kernel_statistics=groups,
                      native_log=str(out/'native.log'), event_csv_role='diagnostic_under_profiler_not_baseline',
                      task_statistics_are_not_end_to_end_latency=True)
        if world > 1:
            dist.barrier()
            dist.destroy_process_group()
        (out/'report.json').write_text(json.dumps(report, indent=2))
        print(f'PASS rank={rank}: receiver/metadata checks and named task profiling: {out}', flush=True)


if __name__ == '__main__':
    main()
