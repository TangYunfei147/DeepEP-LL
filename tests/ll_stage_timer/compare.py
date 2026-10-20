"""Compare matching LL reports; never treat differing configurations as a speedup."""
import argparse
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DIFFERENCES={'implementation','transport','rdma_compiled','source_revision','source_dirty','logical_group'}
REQUIRED={'implementation','transport','world_size','logical_group','active_aiv','core_protocol',
          'case','warmup','num_iters','perf_tail','delay_rows','delay_cols','stage_capture','expert_capture',
          'timer_mode','stage_timer','stage_timer_schema','frequency_hz','seed','route','use_1g_huge_pages',
          'comparison_schema','sampling_protocol','perf_enabled'}


def check_identity(left,right):
    if not isinstance(left,dict) or not isinstance(right,dict) or not REQUIRED<=left.keys() or not REQUIRED<=right.keys():
        raise ValueError('Missing comparison identity; recollect with the shared launcher and matching wheels')
    for item in (left,right):
        if item['comparison_schema']!=1 or any(type(item[key]) is not int or item[key]<=0
                                               for key in ('world_size','logical_group','active_aiv','core_protocol')):
            raise ValueError('Unknown or invalid runtime comparison identity')
    differences=set(DIFFERENCES)
    if all(7001 <= item['core_protocol'] <= 7055 or 8001 <= item['core_protocol'] <= 8055 for item in (left,right)):
        for item in (left,right):
            if (item.get('send_cores',0)+item.get('recv_cores',0)!=56 or item['active_aiv']!=56
                    or item['core_protocol']!=(item['core_protocol']//1000)*1000+item.get('send_cores',0)):
                raise ValueError('Invalid configured core split identity')
        for item in (left,right):
            shared = item['core_protocol']//1000 == 8
            expected = 56 if shared else item['recv_cores']
            if item.get('dispatch_recv_cores',expected)!=expected or item['stage_timer_schema']!=(6 if shared else 5):
                raise ValueError('Invalid receive participation/schema identity')
        differences.update(('send_cores','recv_cores','dispatch_recv_cores','core_protocol','stage_timer_schema'))
    mismatches=[key for key in sorted(left.keys()|right.keys()) if key not in differences and left.get(key)!=right.get(key)]
    if mismatches: raise ValueError('Incomparable settings: '+', '.join(mismatches))


def compare(left,right,kind):
    def indexed(report):
        if kind=='perf':
            result={}
            if report.get('status')!='completed': raise ValueError('Incomplete performance run')
            for case in report['results']:
                if case.get('correctness')!='passed': raise ValueError('Performance case lacks correctness evidence')
                for operation in ('dispatch','combine'):
                    for rank in case.get(operation,{}).get('ranks',[]):
                        key=(case['case_id'],operation,rank['rank'])
                        if key in result: raise ValueError('Duplicate performance key')
                        result[key]=(rank['mean_us'],case.get('comparison'))
            return result
        collection='per_rank_core_stage' if kind=='stage' else 'per_expert'
        keys=('operation','rank','core','stage') if kind=='stage' else ('executing_rank','expert_id','expert_rank','stage')
        result={}
        for row in report.get(collection,[]):
            key=tuple(row[k] for k in keys)
            if key in result: raise ValueError('Duplicate diagnostic key')
            if row['unit']!='us': raise ValueError('Use microseconds for comparison')
            result[key]=(row['mean'],report.get('comparison'))
        return result
    l,r=indexed(left),indexed(right)
    if not l or not r: raise ValueError('No matching measurement data in one report')
    if not l.keys()&r.keys(): raise ValueError('No overlapping measurement keys')
    if kind!='perf': check_identity(left.get('comparison'),right.get('comparison'))
    rows=[]
    for key in sorted(l.keys()|r.keys()):
        if key in l and key in r: check_identity(l[key][1],r[key][1])
        a=l[key][0] if key in l else None;b=r[key][0] if key in r else None
        role_changed = (kind=='stage' and key in l and key in r
                        and l[key][1].get('send_cores') is not None and r[key][1].get('send_cores') is not None
                        and (key[2] < l[key][1]['send_cores']) != (key[2] < r[key][1]['send_cores']))
        partition_changed = (kind=='stage' and key[0]=='dispatch' and key in l and key in r
                             and (l[key][1]['core_protocol']//1000==8)!=(r[key][1]['core_protocol']//1000==8))
        rows.append(dict(key=json.dumps(key),baseline_mean_us=a,candidate_mean_us=b,
                         baseline_send_cores=l[key][1].get("send_cores") if key in l else None,
                         candidate_send_cores=r[key][1].get("send_cores") if key in r else None,
                         baseline_recv_cores=l[key][1].get("recv_cores") if key in l else None,
                         candidate_recv_cores=r[key][1].get("recv_cores") if key in r else None,
                         baseline_group=l[key][1]['logical_group'] if key in l else None,
                         candidate_group=r[key][1]['logical_group'] if key in r else None,
                         candidate_over_baseline=b/a if a is not None and a>0 and b is not None and not role_changed and not partition_changed else None,
                         status='work_partition_changed' if partition_changed else 'role_changed' if role_changed else ('matched' if key in l and key in r else 'missing_on_one_side')))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('baseline',type=Path);p.add_argument('candidate',type=Path)
    p.add_argument('--kind',choices=('perf','stage','expert'),required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    resolve=lambda path:path if path.is_absolute() else ROOT/path
    rows=compare(json.loads(resolve(args.baseline).read_text()),json.loads(resolve(args.candidate).read_text()),args.kind)
    output=resolve(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(f'Compared matching settings -> {output}; missing stages remain blank, never zero-filled')


if __name__=='__main__':main()
