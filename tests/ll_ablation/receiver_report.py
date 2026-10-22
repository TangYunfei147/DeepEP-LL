"""Paired receive replay durations and executed-operation checks; never sum savings."""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import csv
from common import ROOT,write_json

CASES={'baseline':('ABLATION_READY_RECV','REPLAY_BASE_FLAGS_PLUS1','REPLAY_BASE_BYTES_PLUS1'),
       'skip-flags':('ABLATION_SKIP_FLAGS','REPLAY_SKIP_FLAGS_PLUS1','REPLAY_SKIP_BYTES_PLUS1'),
       'no-payload-output':('ABLATION_NO_PAYLOAD','REPLAY_NOOUT_FLAGS_PLUS1','REPLAY_NOOUT_BYTES_PLUS1')}
ERRORS={'baseline':'REPLAY_BASE_ERRORS_PLUS1','skip-flags':'REPLAY_SKIP_ERRORS_PLUS1','no-payload-output':'REPLAY_NOOUT_ERRORS_PLUS1'}

def decode_sample(decoded):
    if decoded['schema']!=15:raise ValueError('Receiver breakdown requires schema15')
    cores={}
    for row in decoded['records']:cores.setdefault(row['core'],{})[row['stage']]=row
    if set(cores)!=set(range(56)):raise ValueError('Missing receiver cores')
    result=[]
    for core,stages in sorted(cores.items()):
        arms={}
        for name,(duration,flags,output) in CASES.items():
            if any(stage not in stages for stage in (duration,flags,output,'REPLAY_ORDER_PLUS1',ERRORS[name])):
                raise ValueError('Incomplete operation counters or replay envelope')
            if stages[duration]['calls']!=1:raise ValueError('Replay envelope must occur once')
            if stages[ERRORS[name]]['calls']!=1 or stages[ERRORS[name]]['total_ticks']:
                raise ValueError(f'{name}: per-arm payload/scale verification failed')
            f=stages[flags]['calls']-1;b=stages[output]['calls']-1
            if min(f,b)<0 or stages[flags]['total_ticks'] or stages[output]['total_ticks']:
                raise ValueError('Invalid replay operation counter')
            arms[name]=dict(ticks=stages[duration]['total_ticks'],flag_batches=f,payload_bytes=b)
        base,skip,noout=[arms[k] for k in CASES]
        if skip['flag_batches']!=0:raise ValueError('Skip-flags still issued flag probes')
        if noout['payload_bytes']!=0:raise ValueError('No-output still issued payload/scales writes')
        if base['payload_bytes']!=skip['payload_bytes'] or base['flag_batches']!=noout['flag_batches']:
            raise ValueError('Arms did not process matching tasks')
        if bool(base['flag_batches'])!=bool(base['payload_bytes']):raise ValueError('Invalid active receiver counts')
        order=stages['REPLAY_ORDER_PLUS1']['calls']-1
        if order not in (0,1,2):raise ValueError('Invalid replay order')
        result.append(dict(core=core,order=order,arms=arms))
    if len({r['order'] for r in result})!=1:raise ValueError('Core replay order differs')
    return result

def report(directory):
    identity=json.loads((directory/'ablation_identity.json').read_text())
    if identity['mode']!='ready-breakdown' or identity['timer']!='ON':raise ValueError('Wrong experiment identity')
    summary=json.loads((directory/'summary.json').read_text())
    if summary['status']!='completed' or summary['world_size']!=16:raise ValueError('Incomplete EP16 run')
    if any(r['correctness']!='passed' for r in summary['results']):raise ValueError('Correctness did not pass')
    spec=importlib.util.spec_from_file_location('receiver_timer',ROOT/'deep_ep/utils/ll_stage_timer.py')
    timer=importlib.util.module_from_spec(spec);spec.loader.exec_module(timer)
    samples={};seen=set()
    for file in directory.rglob('*.json'):
        obj=json.loads(file.read_text())
        if not isinstance(obj,dict) or 'raw' not in obj or obj.get('operation')!='dispatch':continue
        meta=obj['metadata']
        if meta.get('correctness')!='passed':raise ValueError('Capture lacks correctness evidence')
        case=json.dumps([meta['config'],meta.get('case_id'),meta.get('route'),meta.get('seed')],sort_keys=True)
        key=(case,meta['rank'],meta['iteration'])
        if key in seen:raise ValueError('Duplicate capture')
        seen.add(key)
        decoded=timer.parse_records(obj['raw'],'dispatch',obj.get('frequency_hz'))
        if not decoded['frequency_hz']:raise ValueError('Frequency required for microsecond comparison')
        factor=1e6/decoded['frequency_hz'];cores=decode_sample(decoded)
        for core in cores:
            for arm in core['arms'].values():arm['us']=arm['ticks']*factor
        durations={name:max(c['arms'][name]['us'] for c in cores) for name in CASES}
        samples.setdefault(case,[]).append(dict(rank=meta['rank'],iteration=meta['iteration'],
            order=cores[0]['order'],max_core_us=durations,cores=cores,
            skip_flags_delta_us=durations['baseline']-durations['skip-flags'],
            no_output_delta_us=durations['baseline']-durations['no-payload-output']))
    if not samples:raise ValueError('No receiver breakdown captures')
    aggregate={}
    for case,rows in samples.items():
        if {r['rank'] for r in rows}!=set(range(16)):raise ValueError('Collect all16rank captures first')
        iterations={r['iteration'] for r in rows}
        for rank in range(16):
            owned=[r for r in rows if r['rank']==rank]
            if {r['iteration'] for r in owned}!=iterations:raise ValueError('Missing rank/iteration capture')
            counts=[sum(r['order']==i for r in owned) for i in range(3)]
            if min(counts)==0 or max(counts)-min(counts)>1:raise ValueError('Replay orders are not balanced')
        per_iteration=[]
        for it in sorted(iterations):
            owned=[r for r in rows if r['iteration']==it]
            values={name:max(r['max_core_us'][name] for r in owned) for name in CASES}
            per_iteration.append(dict(iteration=it,us=values,
                skip_flags_delta_us=values['baseline']-values['skip-flags'],
                no_output_delta_us=values['baseline']-values['no-payload-output']))
        aggregate[case]=dict(per_iteration=per_iteration,
            median_max_duration_us={name:statistics.median(r['us'][name] for r in per_iteration) for name in CASES},
            median_skip_flags_delta_us=statistics.median(r['skip_flags_delta_us'] for r in per_iteration),
            median_no_output_delta_us=statistics.median(r['no_output_delta_us'] for r in per_iteration),
            per_rank=[dict(rank=rank,median_duration_us={name:statistics.median(
                r['max_core_us'][name] for r in rows if r['rank']==rank) for name in CASES}) for rank in range(16)])
    return dict(identity=identity,summary=aggregate,samples=samples,
        interpretation='Paired warm replay arms in one wheel; restore/sync excluded, pending reset/cleanup/drain included. Deltas overlap and must not be added. Old40us is not the new baseline.')

def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    core_file=a.output.with_suffix('.cores.csv');sample_file=a.output.with_suffix('.rank_samples.csv')
    if any(p.exists() for p in (a.output,core_file,sample_file)):raise ValueError('Output exists')
    result=report(a.directory)
    core_rows=[];rank_rows=[]
    for case,rows in result.pop('samples').items():
        for row in rows:
            base=dict(case=case,rank=row['rank'],iteration=row['iteration'],order=row['order'])
            rank_rows.append(dict(**base,**row['max_core_us'],skip_flags_delta_us=row['skip_flags_delta_us'],no_output_delta_us=row['no_output_delta_us']))
            for core in row['cores']:
                values={name.replace('-','_')+'_'+key:value for name,arm in core['arms'].items() for key,value in arm.items()}
                core_rows.append(dict(**base,core=core['core'],**values))
    for file,rows in ((core_file,core_rows),(sample_file,rank_rows)):
        with file.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result['detail_files']={'per_core':core_file.name,'rank_samples':sample_file.name}
    write_json(a.output,result)
    print(json.dumps(result['summary'],indent=2));print(a.output)

if __name__=='__main__':main()
