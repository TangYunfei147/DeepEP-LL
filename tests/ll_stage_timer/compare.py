"""Compare matching LL reports; never treat differing configurations as a speedup."""
import argparse
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
# Cross-algorithm A/B may intentionally change source. Within one staged
# diagnosis, diagnose.py separately requires source_sha256 equality for OFF/ON.
PREFIX_FIELDS=('small_b_prefix_cores','effective_small_b_prefix_cores')
DIFFERENCES={'implementation','transport','rdma_compiled','source_revision','source_dirty','source_sha256','logical_group','overlap_stage',*PREFIX_FIELDS}
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
        for field in PREFIX_FIELDS:
            value=item.get(field)
            if value is not None and (type(value) is not int or value < (1 if field.startswith('effective_') else 0)
                                      or value > item.get('recv_cores',55)):
                raise ValueError('Invalid prefix producer identity')
        if item.get('effective_small_b_prefix_cores') is not None and item.get('small_b_prefix_cores') is None:
            raise ValueError('Effective prefix producer identity lacks its recorded build setting')
    if any(isinstance(s,int) and s>=13 for s in (left['stage_timer_schema'],right['stage_timer_schema'])) and left['stage_timer_schema']!=right['stage_timer_schema'] and any(
        item['stage_timer'] or item['stage_capture'] or item['expert_capture'] for item in (left,right)):
        raise ValueError('Incomparable diagnostic semantics: different stage schemas')
    differences=set(DIFFERENCES)
    # When both binaries and launches have timers OFF, timer layout/profile
    # metadata describes an unused capability, not a measured difference.
    inactive=all(not any(item.get(key,False) for key in ('stage_timer','stage_capture','expert_capture','profile_capture','profile_capture_requested','expert_timer_requested')) for item in (left,right))
    if inactive:
        differences.update(('stage_timer_schema','stage_timer_slots','stage_timer_fields',
                            'stage_timer_points','stage_timer_recording','timer_profile','timer_profile_id',
                            'profile_capture','profile_capture_requested','expert_timer_requested'))
    if all(19001<=item['core_protocol']<=19055 for item in (left,right)):
        for item in (left,right):
            send=item.get('send_cores',0);recv=item.get('recv_cores',0)
            if send+recv!=56 or min(send,recv)<=0 or item['core_protocol']!=19000+send or item['active_aiv']!=56:
                raise ValueError('Invalid configured core split identity')
        differences.update(('send_cores','recv_cores','dispatch_recv_cores','core_protocol','stage_timer_schema'))

    if all(7001 <= item['core_protocol'] <= 7055 or 8001 <= item['core_protocol'] <= 8055 or 9001 <= item['core_protocol'] <= 9055 or 10001 <= item['core_protocol'] <= 10055 or 14001 <= item['core_protocol'] <= 14055 for item in (left,right)):
        for item in (left,right):
            if (item.get('send_cores',0)+item.get('recv_cores',0)!=56 or item['active_aiv']!=56
                    or item['core_protocol']!=(item['core_protocol']//1000)*1000+item.get('send_cores',0)):
                raise ValueError('Invalid configured core split identity')
        for item in (left,right):
            shared = item['core_protocol']//1000 >= 8
            expected = 56 if shared else item['recv_cores']
            if item.get('dispatch_recv_cores',expected)!=expected or item['stage_timer_schema'] not in {7:(5,),8:(6,),9:(7,),10:(7,9,10),14:(11,12,13,14,15)}[item['core_protocol']//1000]:
                raise ValueError('Invalid receive participation/schema identity')
        if (left['core_protocol']//1000 in (9,10,14))!=(right['core_protocol']//1000 in (9,10,14)):
            if any(item['stage_timer'] or item['stage_capture'] or item['expert_capture'] for item in (left,right)):
                raise ValueError('Incomparable diagnostic semantics: segment and legacy timers')
        if any(item['stage_timer_schema'] in (9,10,11,12,13,14,15) for item in (left,right)) and left['stage_timer_schema']!=right['stage_timer_schema'] and any(
                item['stage_timer'] or item['stage_capture'] or item['expert_capture'] for item in (left,right)):
            raise ValueError('Incomparable diagnostic semantics: different stage schemas')
        differences.update(('send_cores','recv_cores','dispatch_recv_cores','core_protocol','stage_timer_schema'))
    mismatches=[key for key in sorted(left.keys()|right.keys()) if key not in differences and left.get(key)!=right.get(key)]
    if mismatches: raise ValueError('Incomparable settings: '+', '.join(mismatches))


def prefix_partition_status(left,right):
    """A per-core stage ratio requires the same recorded producer partition.

    Two old captures remain comparable as before. One old and one new capture
    cannot establish equal P; never assign the old capture today's default.
    Raw 0 and explicit R may describe the same recorded effective partition.
    """
    known=[item.get('small_b_prefix_cores') is not None for item in (left,right)]
    if known[0] != known[1]:
        return 'prefix_configuration_unknown'
    if not any(known):
        return None
    if ('effective_small_b_prefix_cores' in left) != ('effective_small_b_prefix_cores' in right):
        return 'prefix_configuration_unknown'
    if left.get('effective_small_b_prefix_cores') != right.get('effective_small_b_prefix_cores'):
        return 'prefix_partition_changed'
    if 'effective_small_b_prefix_cores' not in left and left['small_b_prefix_cores'] != right['small_b_prefix_cores']:
        return 'prefix_configuration_unknown'
    return None


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
    if kind!='perf':
        if (left.get('schema')==15 or right.get('schema')==15) and left.get('schema')!=right.get('schema'):
            raise ValueError('Incomparable diagnostic semantics: different stage schemas')
        if left.get('timer_profile','full') != right.get('timer_profile','full'):
            raise ValueError('Incomparable diagnostic semantics: different timer profiles')
        check_identity(left.get('comparison'),right.get('comparison'))
    rows=[]
    for key in sorted(l.keys()|r.keys()):
        if key in l and key in r: check_identity(l[key][1],r[key][1])
        a=l[key][0] if key in l else None;b=r[key][0] if key in r else None
        role_changed = (kind=='stage' and key in l and key in r
                        and l[key][1].get('send_cores') is not None and r[key][1].get('send_cores') is not None
                        and (key[2] < l[key][1]['send_cores']) != (key[2] < r[key][1]['send_cores']))
        partition_changed = (kind=='stage' and key[0]=='dispatch' and key in l and key in r
                             and ((l[key][1]['core_protocol']//1000>=8)!=(r[key][1]['core_protocol']//1000>=8)
                                  or (l[key][1]['core_protocol']//1000==10)!=(r[key][1]['core_protocol']//1000==10)))
        prefix_status = (prefix_partition_status(l[key][1],r[key][1])
                         if kind=='stage' and key[0]=='dispatch' and key in l and key in r else None)
        rows.append(dict(key=json.dumps(key),baseline_mean_us=a,candidate_mean_us=b,
                         baseline_send_cores=l[key][1].get("send_cores") if key in l else None,
                         candidate_send_cores=r[key][1].get("send_cores") if key in r else None,
                         baseline_recv_cores=l[key][1].get("recv_cores") if key in l else None,
                         candidate_recv_cores=r[key][1].get("recv_cores") if key in r else None,
                         baseline_small_b_prefix_cores=l[key][1].get('small_b_prefix_cores') if key in l else None,
                         candidate_small_b_prefix_cores=r[key][1].get('small_b_prefix_cores') if key in r else None,
                         baseline_effective_small_b_prefix_cores=l[key][1].get('effective_small_b_prefix_cores') if key in l else None,
                         candidate_effective_small_b_prefix_cores=r[key][1].get('effective_small_b_prefix_cores') if key in r else None,
                         baseline_group=l[key][1]['logical_group'] if key in l else None,
                         candidate_group=r[key][1]['logical_group'] if key in r else None,
                         candidate_over_baseline=b/a if a is not None and a>0 and b is not None and not role_changed and not partition_changed and not prefix_status else None,
                         status=prefix_status or ('work_partition_changed' if partition_changed else 'role_changed' if role_changed else ('matched' if key in l and key in r else 'missing_on_one_side'))))
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
