"""Audit all-rank raw communication CSV/logs and compute iteration rank MAX."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'profiling'))
from bandwidth_model import comm_edges
from paths import project_path, results_dir, runtime_environment, workspace_root
import statistics

def stats(values):
    ordered=sorted(values)
    def q(p):
        x=(len(ordered)-1)*p;lo=int(x);hi=min(lo+1,len(ordered)-1)
        return ordered[lo]+(ordered[hi]-ordered[lo])*(x-lo)
    return dict(mean=statistics.mean(values),p50=q(.5),p90=q(.9),p95=q(.95),p99=q(.99),minimum=min(values),maximum=max(values))

def analyze(paths):
    ranks={};identity=None
    for path in paths:
        # User-run layout: samples.csv + run.log. Validation layout uses .log sibling.
        log=path.with_name('run.log') if path.name=='samples.csv' else path.with_suffix('.log')
        if 'receiver checks and normal cleanup' not in log.read_text():raise ValueError('Missing receiver/cleanup proof')
        rows=list(csv.DictReader(path.read_text().splitlines()))
        if not rows:raise ValueError('No timing samples')
        rank=int(rows[0]['rank'])
        if rank in ranks:raise ValueError('Duplicate rank')
        key=tuple(int(rows[0][k]) for k in ['world','scenario','layer','bytes_per_edge'])
        key += (int(rows[0].get('fanout', int(rows[0]['world'])-1)),int(rows[0].get('peer_mask',0)))
        if identity is not None and key!=identity:raise ValueError('Mismatched case')
        identity=key
        for i,row in enumerate(rows):
            if int(row['rank'])!=rank or int(row['sample'])!=i:raise ValueError('Sample identity/alignment mismatch')
            if tuple(int(row[k]) for k in ['world','scenario','layer','bytes_per_edge'])+(int(row.get('fanout',int(row['world'])-1)),int(row.get('peer_mask',0)))!=key:raise ValueError('Case changed midstream')
            if any(not math.isfinite(float(row[k])) or float(row[k])<=0 for k in ['event_us','api_us','job_join_us']):raise ValueError('Invalid timing')
        ranks[rank]=rows
    world,scene,layer,bytes_edge,fanout,peer_mask=identity
    if not 1<=fanout<world or (scene!=3 and fanout!=world-1):raise ValueError("Invalid fanout")
    if sorted(ranks)!=list(range(world)):raise ValueError('Missing ranks')
    count=len(ranks[0])
    if any(len(rows)!=count for rows in ranks.values()):raise ValueError('Unaligned samples')
    if peer_mask<0 or peer_mask>=(1<<world) or peer_mask&1 or (scene==4 and peer_mask):raise ValueError('Invalid peer mask')
    selected=[r for r in range(1,world) if peer_mask&(1<<r)] if peer_mask else None
    if scene==3 and selected is not None and len(selected)!=fanout:raise ValueError('fanout/mask mismatch')
    edges=set(comm_edges(world,scene,fanout if scene==3 and selected is None else None,
                        selected if scene in (1,3) else None,selected if scene==2 else None))
    def edge(s,d):return (s,d) in edges
    job_bytes=0
    for rank,rows in ranks.items():
        expected=bytes_edge*sum(edge(rank,d) for d in range(world)) if layer==3 else 0
        if any(int(row['rank_wire_bytes'])!=expected for row in rows):raise ValueError('Incorrect wire byte accounting')
        job_bytes+=expected
    result={'peer_mask':peer_mask,'fanout':fanout,'world':world,'scenario':scene,'layer':layer,'bytes_per_edge':bytes_edge,'job_wire_bytes':job_bytes,
            'samples':count,'exit_and_receiver_audit':'passed','statistics':{},'per_rank':{}}
    for metric in ['event_us','api_us','job_join_us']:
        maximum=[max(float(ranks[r][i][metric]) for r in ranks) for i in range(count)]
        result['statistics'][metric]={'rank_max_samples':maximum,**stats(maximum)}
    for rank,rows in ranks.items():result['per_rank'][rank]={metric:stats([float(row[metric]) for row in rows]) for metric in ['event_us','api_us','job_join_us']}
    if job_bytes:result['aggregate_wire_gbps']=job_bytes/(result['statistics']['event_us']['mean']*1000)
    result['timing_note']='Event is local stream interval; API waits local completion; job_join includes test barrier. Rank MAX is not globally timestamp-aligned makespan.'
    result.update(measurement='legacy_acl_event_diagnostic', reference_protocol=False, profiler_entry='../profile_probe.py; see ../PROFILING.md')
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('csv',nargs='+',type=project_path);p.add_argument('--output',type=project_path,required=True)
    p.add_argument('--peak-gbps',type=float);p.add_argument('--peak-description')
    args=p.parse_args();result=analyze(args.csv)
    if args.peak_gbps is not None:
        if not args.peak_description or not math.isfinite(args.peak_gbps) or args.peak_gbps<=0:p.error('Require positive peak and matching direction/topology description')
        if 'aggregate_wire_gbps' in result:result.update(attainment=result['aggregate_wire_gbps']/args.peak_gbps,reference_peak_gbps=args.peak_gbps,peak_description=args.peak_description)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2)

if __name__=='__main__':main()
