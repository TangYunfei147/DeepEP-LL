"""Standalone diagnostic harness; all paths relative to this checkout."""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

HERE = Path(__file__).resolve().parent
NAMES = ['empty', 'metadata_scalar_copy', 'metadata_vector_copy', 'q_plain',
         'q_packed', 'pending_scalar', 'pending_vector', 'route_dma',
         'route_preloaded', 'metadata_header_alias']

def idle():
    text = subprocess.check_output(['npu-smi', 'info'], text=True)
    # Fail closed on a changed format or any task; never kill another job.
    ids = [line for line in text.splitlines() if 'No running processes found in NPU' in line]
    if not ids or 'Process id in container' not in text:
        raise RuntimeError('NPU occupancy unavailable: paused')
    import re
    devices = set(re.findall(r'^\|\s*(\d+)\s*\|\s*Ascend', text, re.M))
    empty = set(re.findall(r'No running processes found in NPU (\d+)', text))
    if not devices or empty != devices:
        raise RuntimeError('NPU task exists: paused')
    return text

def summarize(path):
    rows = list(csv.DictReader(path.open()))
    result = []
    if not rows:
        raise RuntimeError(f'empty results: {path}')
    if 'ticks' in rows[0]:
        for mode in sorted({int(r['mode']) for r in rows}):
            selected = [r for r in rows if int(r['sample']) >= 50 and int(r['mode']) == mode]
            # Preserve per-core observations; do not pool cores as independent runs.
            for core in sorted({int(r['core']) for r in selected}):
                rs = [r for r in selected if int(r['core']) == core]
                ticks = [int(r['ticks']) for r in rs]
                result.append(dict(mode=NAMES[mode], core=core, samples=len(rs),
                    ticks_p50=statistics.median(ticks),
                    ticks_per_repeat_p50=statistics.median(ticks)/int(rs[0]['repeats']),
                    event_us_p50=statistics.median(float(r['event_us']) for r in rs)))
    else:
        rs = [r for r in rows if int(r['sample']) >= 50]
        us = statistics.median(float(r['event_us']) for r in rs)
        result.append(dict(samples=len(rs), event_us_p50=us,
            payload_GBps=(int(rs[0]['bytes'])/us/1000 if int(rs[0]['mode']) in (2,3,4) else None),
            note='Event-based payload rate; relay read+write traffic is twice payload; no wire flags.'))
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--suite',choices=['scalar','transfer-smoke','transfer'],default='scalar')
    p.add_argument('--run-name',required=True)
    p.add_argument('--plan',action='store_true')
    a=p.parse_args()
    if Path(a.run_name).name != a.run_name or a.run_name in ('.','..'):
        p.error('run-name must be a single directory name')
    jobs=[]
    if a.suite=='scalar':
        for cores in (1,56):
            for repeats in (1,32):
                for count in (1,32):
                    jobs.append((f'c{cores}-r{repeats}-n{count}', ['scalar_probe',str(cores),str(repeats),str(count)]))
    else:
        sizes=(4096,) if a.suite=='transfer-smoke' else (4096,131072,1048576)
        cores_set=(1,) if a.suite=='transfer-smoke' else (1,28,56)
        for size in sizes:
            for cores in cores_set:
                for lanes in (1,4):
                    for direction,modes in [('tx',(0,1,2,3,4)),('rx',(3,4)),('relay',(3,))]:
                        for mode in modes:
                            jobs.append((f'{direction}-b{size}-c{cores}-l{lanes}-m{mode}',
                                ['peer_probe',str(size),str(cores),str(lanes),'4096',str(mode),'10','100',direction]))
    if a.plan:
        print(json.dumps(jobs,indent=2));return
    for name in ('PROBE_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES','ASCEND_RT_VISIBLE_DEVICES','ASCEND_VISIBLE_DEVICES'):
        if os.environ.get(name)!='0,1':raise RuntimeError(f'{name} must explicitly be 0,1')
    out=HERE/'results'/a.run_name;out.mkdir(parents=True,exist_ok=False)
    (out/'identity.json').write_text(json.dumps(dict(commit=subprocess.check_output(
        ['git','-C',str(HERE),'rev-parse','HEAD'],text=True).strip(),suite=a.suite,
        topology='record machine topology separately; not target-equivalent',
        scope='standalone primitive; not full LL, no Exp/all_reduce conditioning',
        timestamp=time.time(),warmup=10,samples=100,tail=50),indent=2))
    summary={}
    for label,args in jobs:
        (out/f'{label}.preflight.txt').write_text(idle())
        csv_path=out/f'{label}.csv'
        with csv_path.open('w') as stdout,(out/f'{label}.log').open('w') as stderr:
            subprocess.run([str(HERE/'build'/args[0]),*args[1:]],stdout=stdout,stderr=stderr,check=True,timeout=180)
        summary[label]=summarize(csv_path)
        (out/'summary.json').write_text(json.dumps(summary,indent=2))
        print(f'PASS {label}',flush=True)
    print(out)

if __name__=='__main__':main()
