"""A5 two-rank timing diagnostic; never executes or impersonates native LL."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time
from datetime import timedelta

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
from paths import output_path, runtime_environment


def require_two_devices():
    for key in ('PROBE_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES','ASCEND_RT_VISIBLE_DEVICES','ASCEND_VISIBLE_DEVICES'):
        if os.environ.get(key) != '0,1':
            raise ValueError(f'{key} must be explicitly 0,1 for this diagnostic')


def worker(rank, args):
    import torch
    import torch_npu
    import torch.distributed as dist
    import profile_common as pc
    torch.npu.set_device(rank)
    torch.set_num_threads(1)
    dist.init_process_group('hccl', rank=rank, world_size=2, timeout=timedelta(seconds=180))
    st = runpy.run_path(str(REPO/'tests/elastic/test_ep_low_latency.py'), run_name='ll_parse_probe_library')
    ns = st['measure'].__globals__
    original_parse = ns['read_measurement']
    records, pending, cases = [], [], []
    active = {}
    base_capture = pc.Capture
    if args.schedule=='deferred_all':
        original_handler=torch_npu.profiler.tensorboard_trace_handler
        torch_npu.profiler.tensorboard_trace_handler=lambda directory: original_handler(directory,analyse_flag=False)

    class TimedCapture(base_capture):
        def __init__(self, *values, **kwargs):
            t=time.perf_counter(); super().__init__(*values, **kwargs)
            active['profiler_init_seconds']=time.perf_counter()-t

        def close(self, success=True):
            was_closed=self.closed
            t=time.perf_counter(); super().close(success)
            if not was_closed:
                active['close_sync_export_seconds']=time.perf_counter()-t

    pc.Capture = TimedCapture

    def capture(*values, **kwargs):
        t=time.perf_counter()
        result=pc.run_profiled(*values, **kwargs)
        active['capture_including_export_seconds']=time.perf_counter()-t
        return result

    def parse(*values, **kwargs):
        t=time.perf_counter()
        result=original_parse(*values, **kwargs)
        active['parse_gather_seconds']=time.perf_counter()-t
        for group in result['trace_reports_by_rank']:
            if not group or any(len(g['kernel_us'])!=args.samples for g in group):
                raise AssertionError('Missing per-rank profiler samples')
        return result

    ns['run_profiled']=capture
    ns['read_measurement']=parse
    library=REPO/'tests/ll_probe/bandwidth/build/libpeer_profile.so'
    lib=ctypes.CDLL(str(library))
    launch=lib.ll_transfer_probe_do
    launch.argtypes=[ctypes.c_uint32,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_uint64,
                     ctypes.c_uint32,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_uint32]
    launch.restype=None

    def copy(source):
        target=torch.empty_like(source)
        launch(8,torch.npu.current_stream().npu_stream,source.data_ptr(),target.data_ptr(),source.numel(),4096,4,1,3)
        return target

    def check(source, target):
        torch.npu.synchronize()
        st['agree_check'](lambda: torch.testing.assert_close(target.cpu(),source.cpu(),rtol=0,atol=0),args,rank)

    def save_case(case):
        if rank==0:
            path=args.output/f'case_{case["case_id"]:05d}'/'result.json'
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(case,indent=2)+'\n')
        dist.barrier()

    started=time.perf_counter()
    for batch in args.tokens:
        for dtype in args.dtypes:
            case_id=len(cases)
            case={'case_id':case_id,'B':batch,'dtype_label':dtype,'results':{}}
            # Local-copy proxy only. Labels affect byte size, not BF16/FP8 arithmetic.
            source=torch.full((batch*args.hidden*(2 if dtype=='bf16' else 1),),17+rank,dtype=torch.uint8,device='npu')
            initial=copy(source); check(source,initial)
            for _ in range(2):
                checked=copy(initial); check(initial,checked)
            for operation in ('dispatch','combine'):
                if operation=='combine':
                    initial=copy(source); check(source,initial)
                input_tensor=source if operation=='dispatch' else initial
                invoke=lambda: copy(input_tensor)
                active={'case_id':case_id,'operation':operation,'offline_analysis_seconds':0.0}
                records.append(active)
                folder=args.output/f'case_{case_id:05d}'/operation/f'rank_{rank:03d}'
                folder.mkdir(parents=True,exist_ok=True)
                if rank==0: print(f'case={case_id} B={batch} dtype={dtype} {operation} {args.schedule}',flush=True)
                if args.schedule=='immediate':
                    case['results'][operation]=st['measure'](invoke,args,operation,folder,kernel_override='ll_transfer_probe')
                else:
                    output=capture(invoke,folder,args.warmup,args.num_iters,args.delay_rows,args.delay_cols)
                    del output
                    pending.append((case,operation,folder,active))
            cases.append(case)
            if args.schedule=='immediate': save_case(case)
            del source,initial,checked,input_tensor,invoke
            st['completion_boundary'](args)
    capture_phase_end=time.perf_counter()
    for case,operation,folder,record in pending:
        active=record
        if args.schedule=='deferred_all':
            t=time.perf_counter()
            torch_npu.profiler.profiler.analyse(str(folder),max_process_number=1)
            active['offline_analysis_seconds']=time.perf_counter()-t
        case['results'][operation]=parse(args,operation,folder,kernel_override='ll_transfer_probe')
        if operation=='combine': save_case(case)
    dist.barrier(); torch.npu.synchronize()
    end=time.perf_counter()
    report={'rank':rank,'schedule':args.schedule,'case_count':len(cases),'records':records,
            'worker_seconds':end-started,'capture_phase_seconds':capture_phase_end-started,
            'deferred_tail_seconds':end-capture_phase_end,
            'case_ids':[c['case_id'] for c in cases],
            'correctness_passed':True,'all_profiles_parsed':all(len(c['results'])==2 for c in cases),
            'world':2,'warmup':args.warmup,'samples':args.samples,'tail':args.perf_tail,
            'delay_shape':[args.delay_rows,args.delay_cols],'hidden':args.hidden,
            'tokens':args.tokens,'dtypes':args.dtypes,
            'library_sha256':hashlib.sha256(library.read_bytes()).hexdigest(),
            'scope':'local-copy proxy + HCCL alignment; NOT native LL or target performance',
            'torch':torch.__version__,'torch_npu':torch_npu.__version__}
    (args.output/f'rank_{rank}.json').write_text(json.dumps(report,indent=2)+'\n')
    dist.destroy_process_group()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parse-schedule-probe',action='store_true',required=True)
    p.add_argument('--schedule',choices=('immediate','deferred','deferred_all'),required=True)
    p.add_argument('--output',type=output_path,required=True)
    p.add_argument('--tokens',nargs='+',type=int,default=[1,2,4,8,16,24,32,48,64,96,128,192,256])
    p.add_argument('--dtypes',nargs='+',choices=('bf16','fp8'),default=['bf16','fp8'])
    p.add_argument('--hidden',type=int,default=3584)
    p.add_argument('--warmup',type=int,default=10)
    p.add_argument('--samples',type=int,default=100)
    p.add_argument('--tail',type=int,default=50)
    p.add_argument('--delay-rows',type=int,default=64)
    p.add_argument('--delay-cols',type=int,default=64)
    p.add_argument('--port',type=int,default=19891)
    args=p.parse_args(argv)
    require_two_devices(); runtime_environment()
    if args.delay_rows*args.delay_cols>1024*1024 or min(args.delay_rows,args.delay_cols)<=0:
        p.error('A5 diagnostic only: small positive Exp buffer required')
    if args.hidden<=0 or args.hidden%32 or any(b<=0 or b>256 for b in args.tokens): p.error('Invalid proxy size')
    if args.warmup<0 or args.samples<1 or not 0<=args.tail<=args.samples: p.error('Invalid cadence')
    if not 1024<=args.port<=65535: p.error('Invalid port')
    args.output.mkdir(parents=True,exist_ok=False)
    args.group_backend='hccl'; args.trace_ops=False; args.num_iters=args.samples; args.perf_tail=args.tail
    os.environ.update(MASTER_ADDR='127.0.0.1',MASTER_PORT=str(args.port))
    import torch.multiprocessing as mp
    mp.spawn(worker,args=(args,),nprocs=2,join=True)


if __name__=='__main__': main()
