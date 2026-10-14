"""Aggregate validated LL stage traces per rank/core/stage; retain raw ticks."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('ll_stage_timer_parser',ROOT/'deep_ep/utils/ll_stage_timer.py')
timer=importlib.util.module_from_spec(spec);spec.loader.exec_module(timer)


def stats(values):
    values=sorted(values)
    def q(p):
        x=(len(values)-1)*p;lo=int(x);hi=min(lo+1,len(values)-1)
        return values[lo]+(values[hi]-values[lo])*(x-lo)
    return dict(samples=len(values),mean=statistics.mean(values),minimum=min(values),maximum=max(values),
                p50=q(.5),p90=q(.9),p99=q(.99))


def analyze(paths, frequency_hz=None):
    configurations={};ranks={};groups={};identity=None;stage_max={}
    for path in paths:
        saved=json.loads(path.read_text())
        meta=saved['metadata']
        if meta.get('correctness')!='passed':raise ValueError('Missing instrumented correctness evidence')
        report=timer.parse_records(saved['raw'],saved['operation'],frequency_hz if frequency_hz is not None else saved['frequency_hz'])
        common=(meta['world_size'],meta['config'],report['active_aiv'],report['boundary_mode'],report['frequency_hz'],
                meta.get('run_id'),meta.get('seed'),meta.get('case_id'))
        if identity is not None and common!=identity:raise ValueError('Mixed configurations or timer modes')
        identity=common
        op=report['operation'];rank=meta['rank'];iteration=meta['iteration']
        if not 0<=rank<meta['world_size'] or type(iteration) is not int or iteration<0:raise ValueError('Invalid rank/iteration')
        key=(op,rank)
        if iteration in ranks.setdefault(key,set()):raise ValueError('Duplicate rank/iteration')
        ranks[key].add(iteration)
        configurations[op]=meta['config']
        unit='us' if report['frequency_hz'] is not None else 'ticks'
        for row in report['records']:
            key=(op,rank,row['core'],row['role'],row['stage'])
            duration=row['total_us'] if unit=='us' else row['total_ticks']
            groups.setdefault(key,[]).append((duration,row['calls']))
            stage_key=(op,rank,row['stage'])
            per_iteration=stage_max.setdefault(stage_key,{})
            per_iteration[iteration]=max(per_iteration.get(iteration,0),duration)
    if identity is None:raise ValueError('No traces')
    for op in configurations:
        if {rank for kind,rank in ranks if kind==op}!=set(range(identity[0])):
            raise ValueError(f'Missing ranks for {op}')
        reference=ranks[(op,0)]
        if reference!=set(range(len(reference))) or any(ranks[(op,r)]!=reference for r in range(identity[0])):
            raise ValueError('Missing or misaligned diagnostic iterations')
    rows=[]
    for (op,rank,core,role,stage), values in sorted(groups.items()):
        rows.append(dict(operation=op,rank=rank,core=core,role=role,stage=stage,unit=unit,
                         **stats([v for v,n in values]),mean_calls=statistics.mean(n for v,n in values)))
    return dict(schema=timer.SCHEMA,world_size=identity[0],boundary_mode=identity[3],frequency_hz=identity[4],
                configurations=configurations,per_rank_core_stage=rows,
                per_rank_stage_max=[dict(operation=op,rank=rank,stage=stage,unit=unit,**stats(list(values.values())))
                                    for (op,rank,stage),values in sorted(stage_max.items())],
                note='Per-call diagnostic stage totals; nested/parallel stages cannot be added. No cross-rank timestamp subtraction.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path,help='One case/mode/ll_stage_timer directory containing every rank')
    parser.add_argument('--frequency-hz',type=float)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    source=args.directory if args.directory.is_absolute() else ROOT/args.directory
    output=args.output_dir if args.output_dir.is_absolute() else ROOT/args.output_dir
    report=analyze(sorted(source.rglob('sample_*.json')),args.frequency_hz)
    output.mkdir(parents=True,exist_ok=False)
    (output/'summary.json').write_text(json.dumps(report,indent=2))
    rows=report['per_rank_core_stage']
    with (output/'stages.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(output)


if __name__=='__main__':main()
