"""Explicit payload accounting and ideal bandwidth bounds; no hardware probing."""
import hashlib
import math
from pathlib import Path
try:
    from .profile_common import describe
except ImportError:
    from profile_common import describe


def load_hardware(path):
    # Only this runtime YAML is loaded; sibling reference files are documentation.
    import yaml
    raw=Path(path).read_bytes()
    data=yaml.safe_load(raw)
    if not isinstance(data,dict) or data.get('schema_version')!=1 or not isinstance(data.get('model'),dict):
        raise ValueError('Hardware YAML requires schema_version: 1 and model mapping')
    allowed={'per_rank_tx_GBps','per_rank_rx_GBps','intra_host_pair_GBps','inter_host_pair_GBps',
             'per_host_cross_tx_GBps','per_host_cross_rx_GBps','aggregate_GBps'}
    if set(data['model'])-allowed:raise ValueError('Unknown bandwidth model field')
    for key,value in data['model'].items():
        if value is not None and (type(value) not in (int,float) or not math.isfinite(value) or value<=0):
            raise ValueError('Bandwidth capacities must be positive finite GB/s or null: '+key)
    if not any(value is not None for value in data['model'].values()):raise ValueError('Empty bandwidth model')
    return dict(sha256=hashlib.sha256(raw).hexdigest(), specification=data)


def comm_edges(world, scenario, fanout=None):
    n=world-1 if fanout is None else fanout
    if world not in (2,4,8,16,32) or scenario not in (1,2,3,4) or not 1<=n<world:
        raise ValueError('Invalid communication graph')
    if scenario!=3 and fanout is not None:raise ValueError('fanout applies only to scenario3')
    return [(s,d) for s in range(world) for d in range(world) if s!=d and (
        (scenario==1 and s==0 and d==1) or (scenario==2 and s!=0 and d==0) or
        (scenario==3 and s==0 and 1<=d<=n) or scenario==4)]


def volume(world, cfg, local_world=None):
    edges=comm_edges(world,cfg['scenario'],cfg.get('fanout'))
    size=cfg['bytes']; tx=[0]*world;rx=[0]*world
    for s,d in edges:tx[s]+=size;rx[d]+=size
    local=cross=None
    if local_world is not None:
        if type(local_world) is not int or local_world<1 or world%local_world:raise ValueError('Invalid local world')
        local=sum(size for s,d in edges if s//local_world==d//local_world)
        cross=len(edges)*size-local
    return dict(bytes_per_edge=size,edge_count=len(edges),job_sent_bytes=len(edges)*size,
                sent_bytes_by_rank=tx,received_bytes_by_rank=rx,intra_host_sent_bytes=local,
                inter_host_sent_bytes=cross,shape=(dict(bs=cfg.get('bs'),hidden_size=cfg.get('hidden_size'),
                element_bytes=cfg.get('element_bytes')) if cfg.get('bs') is not None else None),payload_is_pattern_bytes=True),edges


def theoretical_time(vol,edges,local_world,hardware):
    if hardware is None:return dict(time_us=None,status='no_reference_config',constraints=[])
    spec=hardware['specification'];model=spec['model'];limits=[]
    def add(name,bytes_,capacity):
        if capacity is not None and bytes_:
            limits.append(dict(resource=name,bytes=bytes_,capacity_GBps=capacity,time_us=bytes_/(capacity*1000)))
    add('max_rank_tx',max(vol['sent_bytes_by_rank']),model.get('per_rank_tx_GBps'))
    add('max_rank_rx',max(vol['received_bytes_by_rank']),model.get('per_rank_rx_GBps'))
    add('aggregate_fabric',vol['job_sent_bytes'],model.get('aggregate_GBps'))
    topology_fields=('intra_host_pair_GBps','inter_host_pair_GBps','per_host_cross_tx_GBps','per_host_cross_rx_GBps')
    if any(model.get(k) is not None for k in topology_fields) and local_world is None:
        raise ValueError('Per-host/pair model needs recorded deployment topology')
    if local_world is not None:
        cross_tx={};cross_rx={}
        for s,d in edges:
            same=s//local_world==d//local_world
            key='intra_host_pair_GBps' if same else 'inter_host_pair_GBps'
            add(f'edge_{s}_{d}',vol['bytes_per_edge'],model.get(key))
            if not same:
                cross_tx[s//local_world]=cross_tx.get(s//local_world,0)+vol['bytes_per_edge']
                cross_rx[d//local_world]=cross_rx.get(d//local_world,0)+vol['bytes_per_edge']
        add('max_host_cross_tx',max(cross_tx.values(),default=0),model.get('per_host_cross_tx_GBps'))
        add('max_host_cross_rx',max(cross_rx.values(),default=0),model.get('per_host_cross_rx_GBps'))
    if not limits:raise ValueError('No configured capacity applies to this traffic')
    bound=max(x['time_us'] for x in limits)
    return dict(time_us=bound,equivalent_job_peak_GBps=vol['job_sent_bytes']/(bound*1000),status='ideal_capacity_lower_bound',
                reference_label=spec.get('label','configured reference'),constraints=limits,
                unknown_limits=[k for k in topology_fields if model.get(k) is None],
                note='No startup, protocol, serialization, cache or contention latency is modeled; this is not a measured link peak.')


def bandwidth_statistics(times_us,vol,theory):
    if not times_us or any(t<=0 or not math.isfinite(t) for t in times_us):raise ValueError('Positive durations required')
    latency=describe(times_us);sent=vol['job_sent_bytes']
    def quantiles(values):
        stats=describe(values)
        return {k.removesuffix('_us'):v for k,v in stats.items()}
    bw=[sent/(t*1000) for t in times_us]
    reference=theory['time_us']
    utilization=None if reference is None else [reference/t*100 for t in times_us]
    rows=[]
    for q in (50,90,99):
        t=latency[f'p{q}_us']
        rows.append(dict(latency_percentile=f'P{q}',time_us=t,effective_GBps=sent/(t*1000),
                         reference_utilization_percent=None if reference is None else reference/t*100))
    return dict(latency_us=latency,bandwidth_GBps=quantiles(bw),
                reference_utilization_percent=None if utilization is None else quantiles(utilization),
                bandwidth_samples_GBps=bw,utilization_samples_percent=utilization,
                theoretical_time_us=reference,at_latency_percentiles=rows,
                reference_utilization_from_mean_time_percent=None if reference is None else reference/latency['mean_us']*100,
                effective_GBps_from_mean_time=sent/(latency['mean_us']*1000),
                units='GB/s=10^9 bytes/s; us=10^-6 seconds; utilization is relative to the selected ideal bound',
                percentile_method='linear interpolation on sorted samples; bandwidth percentiles are not inverse latency percentiles',
                exceeds_reference=bool(utilization and max(utilization)>100))
