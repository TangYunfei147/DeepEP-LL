"""Compare matched OFF experiments; ready replay remains a separate diagnostic."""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics
from common import ROOT,write_json

def load_run(path,mode):
    identity=json.loads((path/'ablation_identity.json').read_text())
    if identity['mode']!=mode:raise ValueError(f'{path}: wrong mode')
    data=json.loads((path/'summary.json').read_text())
    if data['status']!='completed':raise ValueError('Incomplete run')
    if mode!='ready-replay' and identity['timer']!='OFF':raise ValueError('Latency comparison requires OFF')
    metadata={p.name:json.loads(p.read_text()) for p in path.glob('ablation_metadata_rank*.json')}
    if len(metadata)!=data['world_size']:raise ValueError('Need metadata from every rank, collect all hosts first')
    rows={}
    for r in data['results']:
        if r['correctness']!='passed':raise ValueError('Correctness must pass')
        key=json.dumps([r['config'],r['route']],sort_keys=True)
        rows[key]=r
    return identity,data,metadata,rows

def compare(paths):
    runs={mode:load_run(path,mode) for mode,path in paths.items()}
    ref=runs['baseline']
    for mode,(identity,data,metadata,rows) in runs.items():
        for field in ['commit','send_cores','recv_cores']:
            if identity[field]!=ref[0][field]:raise ValueError(f'{mode}: {field} mismatch')
        if identity.get('logical_group')!=ref[0].get('logical_group'):raise ValueError('Logical group mismatch')
        for field in ['world_size','case_count','result_count']:
            if data[field]!=ref[1][field]:raise ValueError(f'{mode}: {field} mismatch')
        if metadata!=ref[2]:raise ValueError(f'{mode}: valid routing metadata mismatch')
        if set(rows)!=set(ref[3]):raise ValueError('Case/route mismatch')
        # Explicitly compare sampling/seed configuration; do not mix unrelated captures.
        for field in ['seed','warmup','num_iters','perf_tail','delay_rows']:
            if data['config'].get(field)!=ref[1]['config'].get(field):raise ValueError(f'{mode}: {field} mismatch')
    output=[]
    for key in ref[3]:
        times={mode:run[3][key]['dispatch']['max_rank_mean_kernel_us'] for mode,run in runs.items() if mode!='ready-replay'}
        output.append(dict(case=json.loads(key),kernel_us=times,
            input_content_effect_us=times['baseline']-times['fixed-payload'],
            preload_delta_us=times['fixed-payload']-times['preload'],
            prepack_delta_us=times['preload']-times['prepack'],
            combined_payload_delta_us=times['fixed-payload']-times['prepack']))
    return output

def replay_report(path):
    spec=importlib.util.spec_from_file_location('ab_timer',ROOT/'deep_ep/utils/ll_stage_timer.py')
    timer=importlib.util.module_from_spec(spec);spec.loader.exec_module(timer)
    groups={}
    for file in path.rglob('*.json'):
        data=json.loads(file.read_text())
        if not isinstance(data,dict) or 'raw' not in data or data.get('operation')!='dispatch':continue
        decoded=timer.parse_records(data['raw'],'dispatch',data.get('frequency_hz'))
        if decoded['schema']!=14:raise ValueError('Replay needs schema14')
        records=[r for r in decoded['records'] if r['stage']=='ABLATION_READY_RECV']
        if len(records)!=56 or any(r['calls']!=1 for r in records):raise ValueError('Incomplete replay envelope')
        meta=data['metadata'];case=json.dumps(meta['config'],sort_keys=True)
        item=dict(rank=meta['rank'],iteration=meta['iteration'],max_core_ticks=max(r['total_ticks'] for r in records))
        if decoded['frequency_hz'] is not None:item['max_core_us']=item['max_core_ticks']*1e6/decoded['frequency_hz']
        groups.setdefault(case,[]).append(item)
    if not groups:raise ValueError('No replay diagnostic captures found')
    return groups

def main():
    p=argparse.ArgumentParser()
    for name in ['baseline','fixed-payload','preload','prepack']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--ready-replay',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    paths={n:getattr(a,n.replace('-','_')) for n in ['baseline','fixed-payload','preload','prepack']}
    if a.ready_replay:paths['ready-replay']=a.ready_replay
    if a.output.exists():raise ValueError('Output exists')
    report=dict(latency=compare(paths),interpretation='Matched max-rank mean kernel durations; deltas are empirical, not additive hardware costs.')
    if a.ready_replay:report['ready_replay']=replay_report(a.ready_replay)
    write_json(a.output,report)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
