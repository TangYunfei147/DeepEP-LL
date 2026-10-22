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
REPORT_VERSION=2
ARM_NAMES=tuple(CASES)
AUDIT_WARNINGS=[
    dict(code='conditioned_replay_not_live_receive',message='All packets are ready and duplicate relay is suppressed. These envelopes do not measure live receive/forward or complete dispatch.'),
    dict(code='position_and_predecessor_confounded',message='Three cyclic orders balance position but not predecessor: later baseline always follows no-output, skip-flags follows baseline, and no-output follows skip-flags. Balanced means do not remove treatment-dependent carryover.'),
    dict(code='validation_changes_next_arm_state',message='Later arms follow full packet/output verification and UB lane overwrites; the first follows real dispatch. Restoration and verification are outside timing but change the measured state.'),
    dict(code='no_cross_rank_replay_start_barrier',message='SyncAll aligns AIV on one rank only. A maximum core duration is an interval-length proxy, not a cross-rank start-to-finish makespan.'),
    dict(code='different_maximum_owners',message='Legacy deltas subtract independently selected maxima, potentially from different rank/core pairs. Same-core paired differences are also supplied; neither isolates a component cost.'),
    dict(code='verification_scope',message='Counters attest instrumented control paths, not physical traffic. Payload/scale checks compare against retained packets; no-output checks preservation of existing output. They do not establish equivalent cache/pipeline state or per-arm metadata truth.'),
    dict(code='unrecorded_layout_and_consumption_sequence',message='Schema15 does not record per-arm full/compact layout choices or task consumption sequences. Correct payloads and equal output-byte totals do not prove identical input layout, priority or receive order.'),
    dict(code='do_not_attribute_or_add_deltas',message='Deleting work also changes dependencies, overlap, waits and counter work. Replay deltas must not be added or used as production speedup estimates.'),
    dict(code='legacy_route_seed_not_verified',message='The capture seed is declared configuration, not proof of the effective route after correctness repeats. A route digest/effective seed was not recorded in legacy captures; no effective seed is inferred.'),
]


def describe(values):
    """Descriptive observations only: no independent-sample or causal assumption."""
    values=list(values)
    if not values:return dict(count=0,mean=None,median=None,min=None,max=None,positive_count=0)
    return dict(count=len(values),mean=statistics.mean(values),median=statistics.median(values),
                min=min(values),max=max(values),positive_count=sum(v>0 for v in values))


def arm_positions(order):
    return {name:(index-order)%len(ARM_NAMES)+1 for index,name in enumerate(ARM_NAMES)}


def maximum_owner(rows,name):
    """Keep the identity of the maximum; report ties instead of hiding them."""
    candidates=[(core['arms'][name]['us'],row['rank'],core['core']) for row in rows for core in row['cores']]
    duration=max(value[0] for value in candidates)
    tied=sorted((rank,core) for value,rank,core in candidates if value==duration)
    rank,core=tied[0]
    return dict(us=duration,rank=rank,core=core,tie_count=len(tied),tie_selection='lowest_rank_then_core')


def paired_core_summary(rows):
    # Empty receivers have only envelope overhead and cannot identify task costs.
    active=[core for row in rows for core in row['cores'] if core['arms']['baseline']['payload_bytes']>0]
    return dict(unit='us',active_core_samples=len(active),
        skip_flags=describe(c['arms']['baseline']['us']-c['arms']['skip-flags']['us'] for c in active),
        no_output=describe(c['arms']['baseline']['us']-c['arms']['no-payload-output']['us'] for c in active),
        interpretation='Within the same rank, iteration and core. Pooled correlated observations; still subject to position/predecessor and pipeline effects.')

def decode_sample(decoded):
    if decoded['schema']!=15:raise ValueError('Receiver breakdown requires schema15')
    intervals={(row['core'],row['stage']):row for row in decoded.get('intervals',[])}
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
            interval=intervals.get((core,duration),{})
            arms[name]=dict(ticks=stages[duration]['total_ticks'],flag_batches=f,payload_bytes=b,
                first_tick=interval.get('first_tick'),last_tick=interval.get('last_tick'))
        base,skip,noout=[arms[k] for k in CASES]
        if skip['flag_batches']!=0:raise ValueError('Skip-flags still issued flag probes')
        if noout['payload_bytes']!=0:raise ValueError('No-output still issued payload/scales writes')
        if base['payload_bytes']!=skip['payload_bytes'] or base['flag_batches']!=noout['flag_batches']:
            raise ValueError('Arms did not process matching tasks')
        if bool(base['flag_batches'])!=bool(base['payload_bytes']):raise ValueError('Invalid active receiver counts')
        packet_bytes=3584+28*4 # Admitted prequantized FP8 shape, including scale packs.
        if base['payload_bytes']%packet_bytes:raise ValueError('Invalid per-packet output byte count')
        packets=base['payload_bytes']//packet_bytes
        if base['flag_batches']!=(packets+15)//16:
            raise ValueError('Replay was not fully ready: unexpected extra probe batches')
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
    expected_iters=summary.get('config',{}).get('ll_timer_iters')
    if type(expected_iters) is not int or expected_iters<3:raise ValueError('Missing diagnostic iteration configuration')
    if any(r['correctness']!='passed' for r in summary['results']):raise ValueError('Correctness did not pass')
    spec=importlib.util.spec_from_file_location('receiver_timer',ROOT/'deep_ep/utils/ll_stage_timer.py')
    timer=importlib.util.module_from_spec(spec);spec.loader.exec_module(timer)
    samples={};seen=set()
    for file in directory.rglob('*.json'):
        obj=json.loads(file.read_text())
        if not isinstance(obj,dict) or 'raw' not in obj or obj.get('operation')!='dispatch':continue
        meta=obj['metadata']
        if meta.get('correctness')!='passed':raise ValueError('Capture lacks correctness evidence')
        comparison=meta.get('comparison') or {}
        if comparison.get('ablation_mode')!=5 or comparison.get('ablation_wheel_sha256')!=identity['wheel_sha256']:
            raise ValueError('Capture wheel/mode identity differs from run')
        case=json.dumps([meta['config'],meta.get('case_id'),meta.get('route'),meta.get('seed')],sort_keys=True)
        key=(case,meta['rank'],meta['iteration'])
        if key in seen:raise ValueError('Duplicate capture')
        seen.add(key)
        decoded=timer.parse_records(obj['raw'],'dispatch',obj.get('frequency_hz'))
        if decoded['send_cores']!=identity['send_cores'] or decoded['boundary_mode']!='observe':
            raise ValueError('Capture split or timing mode differs')
        if not decoded['frequency_hz']:raise ValueError('Frequency required for microsecond comparison')
        factor=1e6/decoded['frequency_hz'];cores=decode_sample(decoded)
        for core in cores:
            for arm in core['arms'].values():arm['us']=arm['ticks']*factor
        durations={name:max(c['arms'][name]['us'] for c in cores) for name in CASES}
        samples.setdefault(case,[]).append(dict(rank=meta['rank'],iteration=meta['iteration'],
            order=cores[0]['order'],positions=arm_positions(cores[0]['order']),max_core_us=durations,cores=cores,
            frequency_hz=decoded['frequency_hz'],source_file=file.relative_to(directory).as_posix(),
            declared_seed=meta.get('requested_seed',comparison.get('requested_seed',meta.get('seed'))),
            effective_route_seed=meta.get('actual_route_seed',comparison.get('actual_route_seed')),
            route_digest=meta.get('route_digest',comparison.get('route_digest')),
            max_core_winners={name:maximum_owner([dict(rank=meta['rank'],cores=cores)],name) for name in CASES},
            skip_flags_delta_us=durations['baseline']-durations['skip-flags'],
            no_output_delta_us=durations['baseline']-durations['no-payload-output']))
    if not samples:raise ValueError('No receiver breakdown captures')
    if len(samples)!=summary['result_count']:raise ValueError('Missing capture cases')
    aggregate={}
    for case,rows in samples.items():
        rows.sort(key=lambda row:(row['iteration'],row['rank']))
        for field in ('effective_route_seed','route_digest'):
            recorded={row[field] for row in rows if row[field] is not None}
            if len(recorded)>1:raise ValueError(f'Conflicting recorded {field} within the same capture case')
        if {r['rank'] for r in rows}!=set(range(16)):raise ValueError('Collect all16rank captures first')
        iterations={r['iteration'] for r in rows}
        if iterations!=set(range(expected_iters)):raise ValueError('Missing diagnostic iterations')
        for rank in range(16):
            owned=[r for r in rows if r['rank']==rank]
            if {r['iteration'] for r in owned}!=iterations:raise ValueError('Missing rank/iteration capture')
            counts=[sum(r['order']==i for r in owned) for i in range(3)]
            if min(counts)==0 or max(counts)-min(counts)>1:raise ValueError('Replay orders are not balanced')
        per_iteration=[]
        for it in sorted(iterations):
            owned=[r for r in rows if r['iteration']==it]
            orders={r['order'] for r in owned}
            if len(orders)!=1:raise ValueError(f'Cross-rank replay order differs at iteration {it}')
            order=next(iter(orders))
            actual_bytes=sum(c['arms']['baseline']['payload_bytes'] for r in owned for c in r['cores'])
            if actual_bytes!=16*256*8*(3584+28*4):raise ValueError('Global received-route count mismatch')
            values={name:max(r['max_core_us'][name] for r in owned) for name in CASES}
            per_iteration.append(dict(iteration=it,order=order,positions=arm_positions(order),us=values,
                execution_order=[ARM_NAMES[(order+step)%3] for step in range(3)],
                max_duration_owners={name:maximum_owner(owned,name) for name in CASES},
                same_core_paired_deltas_us=paired_core_summary(owned),
                skip_flags_delta_us=values['baseline']-values['skip-flags'],
                no_output_delta_us=values['baseline']-values['no-payload-output']))
        by_position={name:{str(position):dict(
            duration_us=describe(row['us'][name] for row in per_iteration if row['positions'][name]==position),
            iterations=[row['iteration'] for row in per_iteration if row['positions'][name]==position],
            predecessor='real-dispatch' if position==1 else ARM_NAMES[(ARM_NAMES.index(name)-1)%3])
            for position in range(1,4)} for name in CASES}
        by_order={str(order):dict(execution_order=[ARM_NAMES[(order+step)%3] for step in range(3)],
            iterations=[r['iteration'] for r in per_iteration if r['order']==order],
            duration_us={name:describe(r['us'][name] for r in per_iteration if r['order']==order) for name in CASES},
            skip_flags_delta_us=describe(r['skip_flags_delta_us'] for r in per_iteration if r['order']==order),
            no_output_delta_us=describe(r['no_output_delta_us'] for r in per_iteration if r['order']==order))
            for order in range(3)}
        aggregate[case]=dict(per_iteration=per_iteration,
            median_max_duration_us={name:statistics.median(r['us'][name] for r in per_iteration) for name in CASES},
            mean_max_duration_us={name:statistics.mean(r['us'][name] for r in per_iteration) for name in CASES},
            median_skip_flags_delta_us=statistics.median(r['skip_flags_delta_us'] for r in per_iteration),
            median_no_output_delta_us=statistics.median(r['no_output_delta_us'] for r in per_iteration),
            position_strata=by_position,order_strata=by_order,
            same_core_paired_deltas_us=paired_core_summary(rows),
            rank_max_paired_deltas_us=dict(
                skip_flags=describe(r['skip_flags_delta_us'] for r in rows),
                no_output=describe(r['no_output_delta_us'] for r in rows),
                interpretation='Paired rank/iteration, independently selected core maxima; not same-core costs.'),
            route_identity=dict(declared_seeds=sorted({r['declared_seed'] for r in rows},key=str),
                effective_route_seed_status='recorded_not_independently_verified' if all(r['effective_route_seed'] is not None for r in rows) else 'not_recorded_in_all_captures',
                effective_route_seeds=sorted({r['effective_route_seed'] for r in rows if r['effective_route_seed'] is not None},key=str),
                route_digest_status='recorded' if all(r['route_digest'] for r in rows) else 'not_recorded_in_all_captures',
                route_digests=sorted({r['route_digest'] for r in rows if r['route_digest']})),
            per_rank=[dict(rank=rank,median_duration_us={name:statistics.median(
                r['max_core_us'][name] for r in rows if r['rank']==rank) for name in CASES},
                same_core_paired_deltas_us=paired_core_summary([r for r in rows if r['rank']==rank])) for rank in range(16)])
    return dict(report_version=REPORT_VERSION,identity=identity,summary=aggregate,samples=samples,
        audit_warnings=AUDIT_WARNINGS,
        measurement=dict(unit='us',maximum='maximum independently measured rank/core interval length per arm and iteration',
            positions='one-based, derived from recorded order; never inferred from iteration modulo3',
            variation='Descriptive summaries of correlated captures, not confidence intervals.',
            inference='Exploratory conditioned replay; does not identify component costs, causal optimization priority or production speedup.'),
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
            base=dict(case=case,rank=row['rank'],iteration=row['iteration'],order=row['order'],
                frequency_hz=row['frequency_hz'],source_file=row['source_file'],
                **{name.replace('-','_')+'_position':position for name,position in row['positions'].items()})
            rank_rows.append(dict(**base,**row['max_core_us'],
                **{name.replace('-','_')+'_max_core':winner['core'] for name,winner in row['max_core_winners'].items()},
                skip_flags_delta_us=row['skip_flags_delta_us'],no_output_delta_us=row['no_output_delta_us']))
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
