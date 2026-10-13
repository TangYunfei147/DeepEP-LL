"""Validate two-card subcases; never claim N-way contention on two cards."""
import csv
import json
import math
from pathlib import Path
import subprocess
import time
import os
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--comm-only',action='store_true',help='Run one communication regression after Host-only changes')
args=parser.parse_args()
HERE=Path(__file__).resolve().parent
import tempfile
OUT=Path(tempfile.mkdtemp(prefix='four-layers-',dir='/home/tyf/Q4_remote_test/project_control/results'))
print(str(OUT),flush=True)
if not args.comm_only:
    subprocess.run(['bash',str(HERE/'run.sh'),str(HERE/'build/metadata_probe')],check=True,
                   stdout=(OUT/'metadata.log').open('w'),stderr=subprocess.STDOUT,timeout=90)
cases=[(4,3)] if args.comm_only else [(1,0),(1,1)]+[(key,3) for key in range(1,5)]
for index,(key,layer) in enumerate(cases):
    processes=[];streams=[]
    try:
        for rank in [0,1]:
            stdout=(OUT/f'{index}-rank{rank}.csv').open('w');stderr=(OUT/f'{index}-rank{rank}.log').open('w')
            streams.extend([stdout,stderr])
            cmd=['bash',str(HERE/'run.sh'),str(HERE/'build/comm_probe'),str(rank),'2',str(rank),
                 f'tcp://127.0.0.1:{19740+index}','15392','8','4',str(key),str(layer),'0','2']
            env=dict(os.environ)
            if rank==0:env['FOUR_READY_FILE']=str(OUT/f'{index}-rank0.ready')
            processes.append(subprocess.Popen(cmd,stdout=stdout,stderr=stderr,env=env))
            if rank==0:
                ready_deadline=time.monotonic()+90
                while not (OUT/f'{index}-rank0.ready').exists():
                    if processes[0].poll() is not None:raise RuntimeError('rank0 initialization failed')
                    if time.monotonic()>ready_deadline:raise TimeoutError('rank0 context readiness timeout')
                    time.sleep(.1)
                # Give the bootstrap server time to start before client context creation.
                time.sleep(1)
        deadline=time.monotonic()+120
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in [None,0] for p in processes):raise RuntimeError('rank failed')
            if time.monotonic()>deadline:raise TimeoutError('rank timeout')
            time.sleep(.1)
    finally:
        for proc in processes:
            if proc.poll() is None:proc.kill()
            proc.wait()
        for stream in streams:stream.close()
    for rank,proc in enumerate(processes):
        assert proc.returncode==0
        assert 'normal cleanup' in (OUT/f'{index}-rank{rank}.log').read_text()
        rows=list(csv.DictReader((OUT/f'{index}-rank{rank}.csv').read_text().splitlines()))
        assert len(rows)==2
        assert all(float(row['event_us'])>0 and math.isfinite(float(row['event_us'])) for row in rows)
    print(f'PASS scene={key}, layer={layer}, two-card subcase',flush=True)
(OUT/'scope.json').write_text(json.dumps({'actual_ranks':2,'metadata_device':0,'status':'passed',
    'scenarios':sorted({key for key,layer in cases}),'layers_validated':[3] if args.comm_only else [0,1,2,3],'native_full_operator':'CPU plan and adapter checks only; target EP16+ needed',
    'warning':'N-way contention is not validated by the two-card degenerate subcases'},indent=2))
