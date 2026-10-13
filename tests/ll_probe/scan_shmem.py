"""Two-rank SHMEM plan/launcher; --local validates code, not cross-host fabric."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import time
import socket

from paths import workspace_root, project_path, runtime_environment, results_dir
ROOT = workspace_root()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--local',action='store_true',help='Two ranks on devices 0,1 of this host')
    p.add_argument('--rank',type=int,choices=[0,1])
    p.add_argument('--device',type=int,choices=[0,1],default=0)
    p.add_argument('--endpoint',required=True,help='tcp://rank0-host:unused-port; identical on both ranks')
    p.add_argument('--bytes',type=int,nargs='+',default=[32,512,8192,131072,2097152,33554432,67108864])
    p.add_argument('--cores',type=int,nargs='+',default=[1,8,32])
    p.add_argument('--lanes',type=int,nargs='+',default=[1,4])
    p.add_argument('--chunk',type=int,default=4096)
    p.add_argument('--modes',type=int,nargs='+',default=[0,1,2,3])
    p.add_argument('--warmup',type=int,default=10)
    p.add_argument('--samples',type=int,default=50)
    p.add_argument('--peak-gbps',type=float)
    p.add_argument('--validation-only',action='store_true')
    p.add_argument('--execute',action='store_true')
    p.add_argument('--output',type=project_path)
    args=p.parse_args()
    if args.local and args.rank is not None: p.error('--local and --rank are mutually exclusive')
    if not args.local and args.rank is None: p.error('Specify --local or --rank')
    if not args.endpoint.startswith('tcp://') or ':' not in args.endpoint[6:]: p.error('Invalid endpoint')
    if any(b<=0 or b%32 or b>64<<20 for b in args.bytes): p.error('bytes: 32B multiples, up to 64MiB')
    if any(c<1 or c>32 for c in args.cores) or any(l<1 or l>4 for l in args.lanes): p.error('Invalid cores/lanes')
    if any(m not in range(4) for m in args.modes): p.error('Invalid mode')
    if args.chunk<=0 or args.chunk%32 or args.chunk>32768: p.error('Invalid chunk')
    if not 0<=args.warmup<=10000 or not 1<=args.samples<=10000: p.error('Invalid repeats')
    if args.peak_gbps is not None and (not math.isfinite(args.peak_gbps) or args.peak_gbps<=0): p.error('Invalid reference peak')
    cases=[dict(bytes=b,cores=c,lanes=l,chunk=args.chunk,mode=m)
           for b in args.bytes for c in args.cores for l in args.lanes for m in args.modes]
    plan={'topology':'same-host-validation' if args.local else 'user-cross-host',
          'hostname':socket.gethostname(),
          'direction':'rank0->rank1','transport':'SHMEM MTE symmetric mapping',
          'endpoint':args.endpoint,'cases':cases,'samples':0 if args.validation_only else args.samples,
          'warmup':0 if args.validation_only else args.warmup,'reference_peak_gbps':args.peak_gbps}
    plan.update(measurement='legacy_acl_event_diagnostic', reference_protocol=False, profiler_entry='profile_probe.py; see PROFILING.md')
    if not args.execute: print(json.dumps(plan,indent=2)); return
    if args.output is None: p.error('--execute requires fresh --output')
    output=args.output.resolve()
    if not output.is_relative_to(ROOT) or output==ROOT: p.error('Output outside authorized root')
    runtime_environment()
    output.mkdir(parents=True,exist_ok=False)
    binary=Path(__file__).resolve().parent/'build/shmem_probe'
    plan['binary_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()
    (output/'plan.json').write_text(json.dumps(plan,indent=2))
    results=[]
    for index,case in enumerate(cases):
        identities=[(0,0),(1,1)] if args.local else [(args.rank,args.device)]
        processes=[]; files=[]
        try:
            for rank,device in identities:
                stdout=(output/f'{index:04d}-rank{rank}.csv').open('w')
                stderr=(output/f'{index:04d}-rank{rank}.log').open('w')
                files.extend([stdout,stderr])
                cmd=[str(binary),str(rank),str(device),args.endpoint,
                     *(str(case[k]) for k in ['bytes','cores','lanes','chunk','mode']),
                     str(plan['warmup']),str(plan['samples'])]
                processes.append((rank,subprocess.Popen(cmd,stdout=stdout,stderr=stderr)))
            deadline=time.monotonic()+120
            while any(proc.poll() is None for _,proc in processes):
                if any(proc.poll() not in (None,0) for _,proc in processes): raise RuntimeError(f'Case {index} rank failed')
                if time.monotonic()>deadline: raise TimeoutError(f'Case {index} exceeded 120 seconds')
                time.sleep(0.1)
        finally:
            for _,proc in processes:
                if proc.poll() is None: proc.kill()
                proc.wait()
            for stream in files: stream.close()
        report={**case,'samples':plan['samples'],'local_ranks':[],'local_checks':'passed'}
        for rank,proc in processes:
            log=(output/f'{index:04d}-rank{rank}.log').read_text()
            if proc.returncode or f'PASS: rank={rank}' not in log or 'normal cleanup' not in log:
                raise RuntimeError(f'Case {index} rank {rank} did not exit cleanly')
            report['local_ranks'].append(rank)
            if rank==0:
                rows=list(csv.DictReader((output/f'{index:04d}-rank0.csv').read_text().splitlines()))
                if len(rows)!=plan['samples']: raise ValueError('Missing sender samples')
                event=[float(row['event_us']) for row in rows]
                host=[float(row['host_us']) for row in rows]
                if any(t<=0 or not math.isfinite(t) for t in event+host): raise ValueError('Invalid timing')
                if event:
                    mean=statistics.mean(event)
                    report.update(event_mean_us=mean,event_median_us=statistics.median(event),host_mean_us=statistics.mean(host))
                    if case['mode']>=2:
                        gbps=case['bytes']/(mean*1000); report['wire_gbps']=gbps
                        if args.peak_gbps is not None: report['attainment']=gbps/args.peak_gbps
        results.append(report)
        print(f'case {index+1}/{len(cases)} local ranks exited cleanly',flush=True)
    # Cross-host success requires rank1's independently completed report too.
    (output/'summary.json').write_text(json.dumps({'status':'local-ranks-passed','plan':plan,'results':results},indent=2))

if __name__=='__main__': main()
