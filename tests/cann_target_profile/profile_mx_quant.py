"""Single-NPU BF16 -> E4M3/E8M0 MX quantization: 50 warmup, 100 calls, tail50.

No distributed initialization, Exp padding, communication, or per-call synchronization.
Only the quantization operator is issued inside the capture loop.
"""
import argparse
import csv
from decimal import Decimal
import json
import math
import os
from pathlib import Path
import statistics


def stats(values):
    ordered=sorted(values)
    def q(p):
        x=(len(ordered)-1)*p
        low=math.floor(x);high=math.ceil(x)
        return ordered[low]+(ordered[high]-ordered[low])*(x-low)
    return dict(samples=len(values),mean_us=statistics.mean(values),p99_us=q(.99),
                p90_us=q(.90),p50_us=q(.50),min_us=min(values),max_us=max(values))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--tokens',type=int,default=8)
    parser.add_argument('--hidden',type=int,default=3584)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    out=args.output_dir if args.output_dir.is_absolute() else root/args.output_dir
    out.mkdir(parents=True,exist_ok=False)
    for name,leaf in (('ASCEND_WORK_PATH','ascend'),('ASCEND_PROCESS_LOG_PATH','ascend'),
                      ('ASCEND_CACHE_PATH','cache'),('TORCH_EXTENSIONS_DIR','cache/torch'),
                      ('TMPDIR','tmp'),('SHMEM_LOG_PATH','shmem')):
        path=out/leaf;path.mkdir(parents=True,exist_ok=True);os.environ[name]=str(path)
    for name in ('CUDA_VISIBLE_DEVICES','ASCEND_RT_VISIBLE_DEVICES','ASCEND_VISIBLE_DEVICES','PROBE_VISIBLE_DEVICES'):
        os.environ[name]='0'
    import torch
    import torch_npu
    torch_npu.npu.set_device(0)
    torch.manual_seed(0)
    x=(torch.randn(args.tokens,args.hidden)*0.05).to(device='npu:0',dtype=torch.bfloat16)
    def quant():
        # This target's schema default dst_type=296 is E4M3FN; verify below.
        return torch_npu.npu_dynamic_mx_quant(x,axis=-1,round_mode='rint',
                                             dst_type=296,block_size=32,scale_alg=0)
    for _ in range(50):
        y,scales=quant()
    torch_npu.npu.synchronize()
    assert tuple(y.shape)==(args.tokens,args.hidden) and y.dtype==torch.float8_e4m3fn,(y.shape,y.dtype)
    scale_columns=((args.hidden+31)//32+1)//2*2
    assert tuple(scales.shape)==(args.tokens,scale_columns),(scales.shape,scale_columns)
    assert scales.dtype in (torch.uint8,getattr(torch,'float8_e8m0fnu',torch.uint8)),scales.dtype
    trace=out/'profile'
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(trace)),
        profile_memory=True,with_stack=False,with_flops=False,with_modules=False,
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            export_type=torch_npu.profiler.ExportType.Text,
            l2_cache=False,data_simplification=False)):
        for _ in range(100):
            y,scales=quant()
        torch_npu.npu.synchronize()
    found=[]
    task_counts={}
    for path in trace.rglob('kernel_details.csv'):
        with path.open(encoding='utf-8-sig',newline='') as f:
            for row in csv.DictReader(f):
                name=row['Name'];task_counts[name]=task_counts.get(name,0)+1
                if 'dynamicmxquant' in (name+row.get('Type','')).replace('_','').lower():
                    found.append((Decimal(row['Start Time(us)'].strip()),float(row['Duration(us)']),name,str(path.relative_to(out))))
    assert len(found)==100,dict(expected=100,matched=len(found),tasks=task_counts)
    assert len({x[2] for x in found})==1,task_counts
    found.sort(key=lambda x:x[0])
    values=[x[1] for x in found]
    assert all(math.isfinite(v) and v>0 for v in values)
    result=dict(status='completed',input_shape=list(x.shape),input_dtype=str(x.dtype),
                output_shape=list(y.shape),output_dtype=str(y.dtype),scale_shape=list(scales.shape),
                scale_storage_dtype=str(scales.dtype),scale_logical_dtype='E8M0',block_size=32,
                operator='npu_dynamic_mx_quant',warmup=50,capture=100,tail=50,
                protocol='quant operator only; no Exp, no collective; one synchronize at capture end',
                measurement='kernel task Duration(us)',summary=stats(values[-50:]),
                kernel_us=values,kernel_name=found[0][2],task_counts=task_counts,
                raw_csv=sorted({x[3] for x in found}),torch_version=torch.__version__,
                torch_npu_version=torch_npu.__version__)
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    with (out/'samples.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['iteration','duration_us','selected_tail50'])
        w.writerows((i,v,int(i>50)) for i,v in enumerate(values,1))
    print(json.dumps({k:v for k,v in result.items() if k!='kernel_us'}),flush=True)


if __name__=='__main__':
    main()
